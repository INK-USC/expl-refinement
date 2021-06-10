import numpy as np, torch, torch.nn as nn, json
from Model.bert import *
from Model.data_utils import *

# Phrase Matching by enumerating all spans
class Phrase_Matcher(nn.Module):
    def __init__(self, args):
        super(Phrase_Matcher, self).__init__()
        self.args = args
        self.dropout = nn.Dropout(p=self.args['dropout'])
        self.encoder_components = self.args['encoder'].split('-')
        if 'bert' in self.encoder_components:
            self.bert = BERT(self.args['cuda_device'], self.args['fine_tune_bert'])
            self.curr_dim = 768
        if 'lstm' in self.encoder_components:
            self.lstm = nn.LSTM(self.curr_dim, self.args['lstm_hid'], 1, bidirectional=True, batch_first=True)
            if 'boun' in self.encoder_components: # use 2 boundaries as encodings: 4*hidden size
                self.curr_dim = self.args['lstm_hid'] * 4
            else:
                self.curr_dim = self.args['lstm_hid'] * 2
        if 'attn' in self.encoder_components:
            self.attn_M = nn.Parameter(torch.Tensor(self.curr_dim, self.args['attn_hid']))
            nn.init.uniform_(self.attn_M, -np.sqrt(1/self.curr_dim), np.sqrt(1/self.curr_dim))
            self.attn_v = nn.Parameter(torch.Tensor(self.args['attn_hid']))
            nn.init.uniform_(self.attn_v, -np.sqrt(1/self.args['attn_hid']), np.sqrt(1/self.args['attn_hid']))
            self.softmax = nn.Softmax(dim=0)
        
        if self.args['output_layer'] == 'cosine':
            self.cosine_D = nn.Parameter(torch.Tensor(1, self.curr_dim))
            nn.init.uniform_(self.cosine_D, -np.sqrt(1/self.curr_dim), np.sqrt(1/self.curr_dim))
            self.out_layer = nn.CosineSimilarity(dim=1)
        elif self.args['output_layer'] == 'linear':
            self.out_layer = nn.Linear(self.curr_dim*3, 1)
        elif self.args['output_layer'] == 'bilinear':
            self.out_layer = nn.Bilinear(self.curr_dim, self.curr_dim, 1)
        elif self.args['output_layer'] == 'baseline':
            self.out_layer = nn.CosineSimilarity(dim=1)
        
        self.to(self.args['cuda_device'])
    
    def dump_state_dict(self):
        model_params = self.state_dict() # returned is a copy
        for k in model_params:
            model_params[k] = model_params[k].cpu()
        
        return model_params
    
    def encode_phrases(self, sents, spans):
        assert self.encoder_components[0] in ['bert'], "invalid base encoder"
        phrase_encodings = []
        
        if self.encoder_components[0] == 'bert':
            sent_encoding = self.bert(sents)
        
        # if input is a single phrase, return the encoding for the "[CLS]" token
        if self.args['train_mode'] == 'phrase':
            return sent_encoding[:,0,:]
        
        for c in self.encoder_components[1:]:
            if c == 'lstm':
                sent_encoding, _ = self.lstm(self.dropout(sent_encoding))
            elif c == 'boun':
                for s in spans:
                    phrase_encoding = torch.cat([sent_encoding[s[0],s[1],:], sent_encoding[s[0],s[2]-1,:]])
                    phrase_encodings.append(phrase_encoding)
                return torch.stack(phrase_encodings)
            elif c == 'mean':
                for s in spans:
                    phrase = sent_encoding[s[0],s[1]:s[2],:]
                    phrase_encoding = phrase.mean(0)
                    phrase_encodings.append(phrase_encoding)
                return torch.stack(phrase_encodings)
            elif c == 'attn':
                pre_attn = torch.tanh(self.dropout(sent_encoding).bmm(self.attn_M.unsqueeze(0).repeat(self.curr_batch_size,1,1))).bmm(self.attn_v.unsqueeze(0).unsqueeze(-1).repeat(self.curr_batch_size,1,1)).squeeze(-1)
                for s in spans:
                    phrase = sent_encoding[s[0],s[1]:s[2],:]
                    if s[2] - s[1] == 1:
                        phrase_encoding = phrase.squeeze(0)
                    else:
                        phrase_attn = pre_attn[s[0],s[1]:s[2]]
                        attn_scores = self.softmax(phrase_attn)
                        phrase_encoding = (attn_scores.unsqueeze(-1) * phrase).sum(0)
                    phrase_encodings.append(phrase_encoding)
                return torch.stack(phrase_encodings)
            elif c == 'baseline':
                for s in spans:
                    phrase = sent_encoding[s[0],s[1]:s[2],:]
                    phrase_encoding = phrase.sum(0)
                    # phrase_encoding += ((sent_encoding[s[0],0,:] + sent_encoding[s[0],-1,:]) * (s[2]-s[1]) / sent_encoding.shape[1]-2)
                    phrase_encodings.append(phrase_encoding)
                return torch.stack(phrase_encodings)
            else:
                assert False, "invalid encoder"
        
        assert False, "imcomplete encoder"
    
    def predict(self, selected_query_phrase_encodings, selected_target_phrase_encodings):
        # need to standardize output in the range of [0,1]
        if self.args['output_layer'] == 'cosine':
            output = self.out_layer(selected_query_phrase_encodings * self.cosine_D, selected_target_phrase_encodings * self.cosine_D)
            return (output + 1) / 2
        elif self.args['output_layer'] == 'linear':
            output = self.out_layer(self.dropout(torch.cat([selected_query_phrase_encodings, selected_target_phrase_encodings, selected_query_phrase_encodings*selected_target_phrase_encodings], 1))).squeeze(-1)
            return torch.sigmoid(output)
        elif self.args['output_layer'] == 'bilinear':
            output = self.out_layer(self.dropout(selected_query_phrase_encodings), self.dropout(selected_target_phrase_encodings)).squeeze(-1)
            return torch.sigmoid(output)
        elif self.args['output_layer'] == 'baseline':
            output = self.out_layer(selected_query_phrase_encodings, selected_target_phrase_encodings)
            return (output + 1) / 2
    
    def forward_paragraph(self, batch):
        self.curr_batch_size = len(batch)
        query_sents = padding([r[0] for r in batch])
        query_spans = [r[1] for r in batch]
        target_sents = padding([r[2] for r in batch])
        target_spans = [r[3] for r in batch]
        
        num_spans_query = [len(r) for r in query_spans]
        query_spans_flat = [[i, r[0], r[1]] for i in range(len(query_spans)) for r in query_spans[i]]
        num_spans_target = [len(r) for r in target_spans]
        target_spans_flat = [[i, r[0], r[1]] for i in range(len(target_spans)) for r in target_spans[i]]
        
        query_phrase_encodings = self.encode_phrases(query_sents, query_spans_flat)
        target_phrase_encodings = self.encode_phrases(target_sents, target_spans_flat)
        
        query_select_idx, target_select_idx = [], []
        for i in range(len(query_spans_flat)):
            sent_idx = query_spans_flat[i][0]
            query_select_idx += [i] * num_spans_target[query_spans_flat[i][0]]
            target_spans_i_start_pos = sum([num_spans_target[j] for j in range(sent_idx)])
            target_spans_i_end_pos = target_spans_i_start_pos + num_spans_target[sent_idx]
            target_select_idx += range(target_spans_i_start_pos, target_spans_i_end_pos)
        
        selected_query_phrase_encodings = query_phrase_encodings[query_select_idx,:]
        selected_target_phrase_encodings = target_phrase_encodings[target_select_idx,:]
        
        scores = self.predict(selected_query_phrase_encodings, selected_target_phrase_encodings)
        
        all_scores, curr_idx = [], 0
        for sq, st in zip(query_spans, target_spans):
            curr_scores = []
            for s in sq:
                curr_scores.append(scores[curr_idx:(curr_idx+len(st))])
                curr_idx += len(st)
            all_scores.append(curr_scores)
        
        return all_scores
    
    def forward_sent(self, batch):
        batch_1 = batch[0]
        query_sents = [batch_1[0]]
        query_spans = batch_1[1]
        target_sents = padding(batch_1[2])
        target_spans = batch_1[3]
        
        query_spans_flat = [[0, r[0], r[1]] for r in query_spans]
        num_spans_target = [len(r) for r in target_spans]
        target_spans_flat = [[i, r[0], r[1]] for i in range(len(target_spans)) for r in target_spans[i]]
        
        self.curr_batch_size = 1
        query_phrase_encodings = self.encode_phrases(query_sents, query_spans_flat)
        self.curr_batch_size = len(target_sents)
        target_phrase_encodings = self.encode_phrases(target_sents, target_spans_flat)
        
        query_select_idx, target_select_idx = [], []
        for i in range(len(query_spans_flat)):
            sent_idx = query_spans_flat[i][0]
            query_select_idx += [i] * sum(num_spans_target)
            target_select_idx += range(sum(num_spans_target))
        
        selected_query_phrase_encodings = query_phrase_encodings[query_select_idx,:]
        selected_target_phrase_encodings = target_phrase_encodings[target_select_idx,:]
        
        scores = self.predict(selected_query_phrase_encodings, selected_target_phrase_encodings)
        
        all_scores, curr_idx = [], 0
        for sq in query_spans:
            curr_scores = []
            for st in target_spans:
                curr_scores.append(scores[curr_idx:(curr_idx+len(st))])
                curr_idx += len(st)
            all_scores.append(curr_scores)
        
        return all_scores
    
    def forward_phrase(self, batch):
        self.curr_batch_size = len(batch)
        query_phrases = padding([rr for r in batch for rr in r[0]])
        target_phrases = padding([rr for r in batch for rr in r[1]])
        
        query_phrase_encodings = self.encode_phrases(query_phrases, None)
        target_phrase_encodings = self.encode_phrases(target_phrases, None)
        
        num_phrases_query = [len(r[0]) for r in batch]
        num_phrases_target = [len(r[1]) for r in batch]
        
        query_select_idx, target_select_idx = [], []
        for i in range(len(num_phrases_query)):
            query_select_idx += [i] * (num_phrases_query[i] * num_phrases_target[i])
            target_spans_i_start_pos = sum([num_phrases_target[j] for j in range(i)])
            target_spans_i_end_pos = target_spans_i_start_pos + num_phrases_target[i]
            target_select_idx += list(range(target_spans_i_start_pos, target_spans_i_end_pos)) * num_phrases_query[i]
                
        selected_query_phrase_encodings = query_phrase_encodings[query_select_idx,:]
        selected_target_phrase_encodings = target_phrase_encodings[target_select_idx,:]
        
        scores = self.predict(selected_query_phrase_encodings, selected_target_phrase_encodings)
        
        all_scores, curr_idx = [], 0
        for i in range(len(num_phrases_query)):
            curr_scores = []
            for j in range(num_phrases_query[i]):
                curr_scores.append(scores[curr_idx:(curr_idx+num_phrases_target[i])])
                curr_idx += num_phrases_target[i]
            all_scores.append(curr_scores)
        
        return all_scores
    
    def forward(self, batch, mode):
        if mode == "paragraph":
            return self.forward_paragraph(batch)
        elif mode == "sent":
            return self.forward_sent(batch)
        else:
            return self.forward_phrase(batch)

    def encode_phrases_for_nmt(self, sent_encoding, spans):
        """sent_encoding: pre-computed bert encoding for one sent."""
        phrase_encodings = []
        self.curr_batch_size = 1
        for c in self.encoder_components[1:]:
            if c == 'lstm':
                sent_encoding, _ = self.lstm(self.dropout(sent_encoding.unsqueeze(0)))
                sent_encoding = sent_encoding.squeeze(0)
            elif c == 'boun':
                for s in spans:
                    phrase_encoding = torch.cat([sent_encoding[s[0], :], sent_encoding[s[1] - 1, :]])
                    phrase_encodings.append(phrase_encoding)
                return torch.stack(phrase_encodings)
            elif c == 'mean':
                for s in spans:
                    phrase = sent_encoding[s[0]:s[1], :]
                    phrase_encoding = phrase.mean(0)
                    phrase_encodings.append(phrase_encoding)
                return torch.stack(phrase_encodings)
            elif c == 'attn':
                pre_attn = torch.tanh(self.dropout(sent_encoding).mm(self.attn_M)).mm(self.attn_v.unsqueeze(1)).squeeze(1)
                for s in spans:
                    phrase = sent_encoding[s[0]:s[1], :]
                    if s[1] - s[0] == 1:
                        phrase_encoding = phrase.squeeze(0)
                    else:
                        phrase_attn = pre_attn[s[0]:s[1]]
                        attn_scores = self.softmax(phrase_attn)
                        phrase_encoding = (attn_scores.unsqueeze(-1) * phrase).sum(0)
                    phrase_encodings.append(phrase_encoding)
                return torch.stack(phrase_encodings)
            elif c == 'baseline':
                for s in spans:
                    phrase = sent_encoding[s[0]:s[1],:]
                    phrase_encoding = phrase.sum(0)
                    phrase_encodings.append(phrase_encoding)
                return torch.stack(phrase_encodings)
            else:
                assert False, "invalid encoder"

        assert False, "imcomplete encoder"



if __name__ == "__main__":
    sample_data = json.load(open("./Data/train_find_1k.json", 'r'))
    batch = sample_data[:2]
    
    encoder_choices = ['bert-mean', 
                       'bert-attn', 
                       'bert-lstm-mean', 
                       'bert-lstm-boun',  # take boundary representations
                       'bert-lstm-attn']
    
    args = {'encoder': 'bert-lstm-attn', 'lstm_hid': 100, 'attn_hid': 100, \
            'fine_tune_bert': 1, 'cuda_device': torch.device("cuda:0"), \
            'output_layer': 'cosine', 'dropout': 0.1, 'lr': 0.0001, \
            'lr_decay': 0.05, 'epochs': 100, 'batch': 32}
    
    model_ = Phrase_Matcher(args)
    scores = model_(batch)
