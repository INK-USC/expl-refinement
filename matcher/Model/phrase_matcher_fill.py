import numpy as np, torch, torch.nn as nn
from transformers import BertTokenizer
from Model.bert import *


class Tokenizer(nn.Module):
    def __init__(self):
        super(Tokenizer, self).__init__()
        # Load pre-trained model tokenizer (vocabulary)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    def forward(self, text):
        # Tokenize input
        tokenized_text = self.tokenizer.tokenize(text)
        return tokenized_text


# Phrase Matching by enumerating all spans
# only supports batch_size == 1
# lengths of phrases are different, so it doesn't make sense processing with batched input
class Phrase_Matcher_Enum_fill(nn.Module):
    def __init__(self, attn_hid, fine_tune, cuda_device, mode, in_dim=768, output='cosine', if_lstm=False):
        assert mode in ['fill', 'find']
        assert output in ['cosine', 'linear']
        super(Phrase_Matcher_Enum_fill, self).__init__()
        self.attn_hid = attn_hid
        self.fine_tune = fine_tune
        self.cuda_device = cuda_device
        self.mode = mode
        self.in_dim = in_dim
        self.output = output
        self.bert = BERT(self.cuda_device, self.fine_tune)
        if not self.fine_tune:
            self.bert.eval()
        self.if_lstm = if_lstm
        if self.if_lstm:
            self.lstm = nn.LSTM(input_size=self.in_dim, hidden_size=attn_hid, num_layers=1, batch_first=True)
            self.M = nn.Parameter(torch.Tensor(self.attn_hid, self.attn_hid))
            nn.init.uniform_(self.M, -np.sqrt(1/self.attn_hid), np.sqrt(1/self.attn_hid))
            self.D = nn.Parameter(torch.Tensor(1, self.attn_hid))
            nn.init.uniform_(self.D, -np.sqrt(1/self.attn_hid), np.sqrt(1/self.attn_hid))
        else:
            self.M = nn.Parameter(torch.Tensor(self.in_dim, self.attn_hid))
            nn.init.uniform_(self.M, -np.sqrt(1/self.in_dim), np.sqrt(1/self.in_dim))
            self.D = nn.Parameter(torch.Tensor(1, self.in_dim))
            nn.init.uniform_(self.D, -np.sqrt(1/self.in_dim), np.sqrt(1/self.in_dim))
        self.attn_v = nn.Parameter(torch.Tensor(self.attn_hid))
        nn.init.uniform_(self.attn_v, -np.sqrt(1/self.attn_hid), np.sqrt(1/self.attn_hid))
        self.softmax = nn.Softmax(dim=0)
        if self.output == 'cosine':
            self.out_layer = nn.CosineSimilarity(dim=1)
        else:
            self.out_layer = nn.Linear(self.in_dim*2, 1)
        
        if self.mode == 'fill':
            self.id_attn_v = nn.Parameter(torch.Tensor(self.attn_hid))
            nn.init.uniform_(self.id_attn_v, -np.sqrt(1/self.attn_hid), np.sqrt(1/self.attn_hid))
            if self.if_lstm:
                self.id_D = nn.Parameter(torch.Tensor(1, self.attn_hid))
                nn.init.uniform_(self.id_D, -np.sqrt(1/self.attn_hid), np.sqrt(1/self.attn_hid))
            else:
                self.id_D = nn.Parameter(torch.Tensor(1, self.in_dim))
                nn.init.uniform_(self.id_D, -np.sqrt(1/self.in_dim), np.sqrt(1/self.in_dim))
        
        self.to(self.cuda_device)
    
    # only attn will be trained/saved/loaded
    def get_trainable(self):
        if self.fine_tune:
            bert_params = self.bert.state_dict()
            for k in bert_params:
                bert_params[k] = bert_params[k].cpu()
            returned = [self.M.detach().cpu(), self.attn_v.detach().cpu(), self.D.detach().cpu(), \
                        self.id_attn_v.detach().cpu(), self.id_D.detach().cpu(), bert_params]
        else:
            returned = [self.M.detach().cpu(), self.attn_v.detach().cpu(), self.D.detach().cpu(), \
                        self.id_attn_v.detach().cpu(), self.id_D.detach().cpu()]
        
        if self.output != 'cosine':
            out_params = self.out_layer.state_dict()
            for k in out_params:
                out_params[k] = out_params[k].cpu()
            returned.append(out_params)
        
        return returned
        # if self.if_lstm:
            # lstm_params = self.lstm.state_dict()
            # for k in lstm_params:
                # lstm_params[k] = lstm_params[k].cpu()
            # return self.M.detach().cpu(), self.attn_v.detach().cpu(), self.D.detach().cpu(), \
                   # self.id_attn_v.detach().cpu(), self.id_D.detach().cpu(), lstm_params
        # else:
            # return self.M.detach().cpu(), self.attn_v.detach().cpu(), self.D.detach().cpu(), \
                   # self.id_attn_v.detach().cpu(), self.id_D.detach().cpu()
    
    def load_trainable(self, state_dict):
        self.M = nn.Parameter(state_dict[0].to(self.cuda_device))
        self.attn_v = nn.Parameter(state_dict[1].to(self.cuda_device))
        self.D = nn.Parameter(state_dict[2].to(self.cuda_device))
        self.id_attn_v = nn.Parameter(state_dict[3].to(self.cuda_device))
        self.id_D = nn.Parameter(state_dict[4].to(self.cuda_device))
        if self.fine_tune:
            self.bert.load_state_dict(state_dict[5])
        if self.output != 'cosine':
            self.out_layer.load_state_dict(state_dict[-1])
        # if self.if_lstm:
            # self.lstm.load_state_dict(state_dict[5])
    
    def train(self):
        self.M.requires_grad = True
        self.attn_v.requires_grad = True
        self.D.requires_grad = True
        self.id_attn_v.requires_grad = True
        self.id_D.requires_grad = True
        if self.fine_tune:
            self.bert.train()
        if self.if_lstm:
            self.lstm.train()
        if self.output != 'cosine':
            self.out_layer.train()
    
    def eval(self):
        self.M.requires_grad = False
        self.attn_v.requires_grad = False
        self.D.requires_grad = False
        self.id_attn_v.requires_grad = False
        self.id_D.requires_grad = False
        self.bert.eval()
        if self.if_lstm:
            self.lstm.eval()
        self.out_layer.eval()
    
    def get_phrase_encodings(self, all_sents_encoding, pre_attn, sent_idx, span, method='attn'):
        if method == 'attn':
            phrase = all_sents_encoding[sent_idx,span[0]:span[1],:]
            if span[1] - span[0] == 1:
                phrase_encoding = (phrase * self.D).squeeze(0)
            else:
                phrase_attn = pre_attn[sent_idx,span[0]:span[1],:]
                attn_scores = self.softmax(phrase_attn)
                phrase_encoding = (attn_scores * phrase).sum(0) * self.D.squeeze(0)
            
            return phrase_encoding
        
        else:
            phrase = all_sents_encoding[sent_idx,span[0]:span[1],:]
            phrase_encoding = phrase.mean(0) * self.D.squeeze(0)
            return phrase_encoding
    
    def forward(self, all_sents, phrase_matching=None, sent_matching=None, method='attn'):
        all_sents_encoding_ = self.bert(all_sents)
        if self.if_lstm:
            all_sents_encoding, _ = self.lstm(all_sents_encoding_)
        else:
            all_sents_encoding = all_sents_encoding_
        batch_size = all_sents_encoding.shape[0]
        # all_sents_encoding_ = all_sents_encoding.bmm(self.M.unsqueeze(0).repeat(batch_size,1,1))
        all_sents_encoding_ = all_sents_encoding
        if phrase_matching:
            pre_attn = all_sents_encoding.bmm(self.M.unsqueeze(0).repeat(batch_size,1,1)).bmm(self.attn_v.unsqueeze(0).unsqueeze(-1).repeat(batch_size,1,1))
            all_scores = []
            for r in phrase_matching:
                target_encodings = []
                query_encoding = self.get_phrase_encodings(all_sents_encoding_, pre_attn, r[0], r[1], 'attn')
                for target_span in r[3]:
                    target_encoding = self.get_phrase_encodings(all_sents_encoding_, pre_attn, r[2], target_span, 'attn')
                    target_encodings.append(target_encoding)
                
                target_encodings = torch.stack(target_encodings)
                if self.output == 'cosine':
                    scores = self.out_layer(query_encoding.unsqueeze(0), target_encodings)
                    all_scores.append(scores)
                else:
                    scores = self.out_layer(torch.cat([query_encoding.unsqueeze(0).repeat(target_encodings.shape[0],1), target_encodings], 1))
                    all_scores.append(scores.squeeze(1))
            
            return torch.sigmoid(torch.cat(all_scores))
        
        else:
            assert False
            pre_attn = all_sents_encoding.bmm(self.M.unsqueeze(0).repeat(batch_size,1,1)).bmm(self.id_attn_v.unsqueeze(0).unsqueeze(-1).repeat(batch_size,1,1))
            sent_encodings = (nn.Softmax(dim=1)(pre_attn) * all_sents_encoding_).sum(1) * self.D.repeat(batch_size,1)
            all_scores = []
            for r in sent_matching:
                score = nn.CosineSimilarity(dim=0)(sent_encodings[r[0]], sent_encodings[r[1]])
                all_scores.append(score)
            
            return torch.sigmoid(torch.stack(all_scores))
    
    
    # def get_phrase_encodings(self, sent_encoding, phrase_spans, method):
        # assert sent_encoding.shape[0] == 1
        
        # phrase_encodings = []
        # if method == 'attn':
            # pre_attn = torch.tanh(sent_encoding.bmm(self.M.unsqueeze(0))).bmm(self.attn_v.unsqueeze(0).unsqueeze(-1))
            # for span in phrase_spans:
                # phrase = sent_encoding[:,span[0]:span[1],:]
                # if span[1] - span[0] == 1:
                    # phrase_encoding = phrase.squeeze(1) * self.D
                # else:
                    # phrase_attn = pre_attn[:,span[0]:span[1],:]
                    # attn_scores = self.softmax(phrase_attn)
                    # phrase_encoding = (attn_scores * phrase).sum(1) * self.D
                
                # phrase_encodings.append(phrase_encoding) # batch_size * hidden_size
        
        # else:
            # for span in phrase_spans:
                # phrase = sent_encoding[:,span[0]:span[1],:]
                # phrase_encoding = phrase.mean(1) * self.D
                # phrase_encodings.append(phrase_encoding)
        
        # return phrase_encodings
    
    # def forward(self, query_sentence, query_span, target_sentence, target_spans, method):
        # assert method in ['attn', 'mean']
        
        # query_sent_encoding = self.bert(query_sentence) # batch_size * sentence_length * hidden_size
        # query_encoding = self.get_phrase_encodings(query_sent_encoding, query_span, method)[0]
        
        # target_sent_encoding = self.bert(target_sentence)
        # target_encodings = self.get_phrase_encodings(target_sent_encoding, target_spans, method)
        # scores = []
        # for target_encoding in target_encodings:
            # if self.output == 'cosine':
                # score = self.out_layer(query_encoding, target_encoding)
            # else:
                # score = self.out_layer(torch.cat([query_encoding, target_encoding], 1).squeeze(0))
            
            # scores.append(score)
        
        # return torch.sigmoid(torch.cat(scores))


# Phrase Matching by sequence tagging
# supports batch_size > 1 except for phrase encoding
class Phrase_Matcher_SeqT(nn.Module):
    def __init__(self, middle_hid, fine_tune, cuda_device, in_dim=768):
        super(Phrase_Matcher_SeqT, self).__init__()
        self.middle_hid = middle_hid
        self.cuda_device = cuda_device
        self.in_dim = in_dim
        self.bert = BERT(self.cuda_device)
        if not self.fine_tune:
            self.bert.eval()
        self.M = nn.Parameter(torch.Tensor(self.in_dim, self.middle_hid))
        nn.init.uniform_(self.M, -np.sqrt(1/self.in_dim), np.sqrt(1/self.in_dim))
        self.attn_v = nn.Parameter(torch.Tensor(self.middle_hid))
        nn.init.uniform_(self.attn_v, -np.sqrt(1/self.middle_hid), np.sqrt(1/self.middle_hid))
        self.D = nn.Parameter(torch.Tensor(1, self.in_dim))
        nn.init.uniform_(self.D, -np.sqrt(1/self.in_dim), np.sqrt(1/self.in_dim))
        self.softmax = nn.Softmax(dim=1)
        self.cos_sim_2 = nn.CosineSimilarity(dim=2)
        self.to(self.cuda_device)
    
    # only attn will be trained/saved/loaded
    def get_trainable(self):
        return self.M.detach().cpu(), self.attn_v.detach().cpu(), self.D.detach().cpu()
    
    def load_trainable(self, state_dict):
        self.M = nn.Parameter(state_dict[0]).to(self.cuda_device)
        self.attn_v = nn.Parameter(state_dict[1]).to(self.cuda_device)
        self.D = nn.Parameter(state_dict[2]).to(self.cuda_device)
    
    def train(self):
        self.M.requires_grad = True
        self.attn_v.requires_grad = True
        self.D.requires_grad = True
        if self.fine_tune:
            self.bert.train()
    
    def eval(self):
        self.M.requires_grad = False
        self.attn_v.requires_grad = False
        self.D.requires_grad = False
        self.bert.eval()
    
    
    def get_phrase_encodings(self, sent_encoding, phrase_spans, method):
        assert sent_encoding.shape[0] == 1
        
        phrase_encodings = []
        if method == 'attn':
            pre_attn = torch.tanh(sent_encoding.bmm(self.M.unsqueeze(0))).bmm(self.attn_v.unsqueeze(0).unsqueeze(-1))
            for span in phrase_spans:
                phrase = sent_encoding[:,span[0]:span[1],:]
                if span[1] - span[0] == 1:
                    phrase_encoding = phrase.squeeze(1) * self.D
                else:
                    phrase_attn = pre_attn[:,span[0]:span[1],:]
                    attn_scores = self.softmax(phrase_attn)
                    phrase_encoding = (attn_scores * phrase).sum(1) * self.D
                
                phrase_encodings.append(phrase_encoding) # batch_size * hidden_size
        
        else:
            for span in phrase_spans:
                phrase = sent_encoding[:,span[0]:span[1],:]
                phrase_encoding = phrase.mean(1) * self.D
                phrase_encodings.append(phrase_encoding)
        
        return phrase_encodings
    
    def forward(self, query_sentence, query_span, target_sentence, method):
        assert method in ['attn', 'mean']
        
        query_sent_encoding = self.bert(query_sentence) # batch_size * sentence_length * hidden_size
        list_query_encodings = []
        for i in range(query_sent_encoding.shape[0]):
            list_query_encodings.append(self.get_phrase_encodings(query_sent_encoding[i,:,:].unsqueeze(0), [query_span[i]], method)[0])
        
        query_encodings = torch.stack(list_query_encodings)
        
        target_sent_encoding = self.bert(target_sentence) * self.D.unsqueeze(0)
        
        scores = self.cos_sim_2(target_sent_encoding, query_encodings.repeat(1,target_sent_encoding.shape[1],1))
        
        return torch.sigmoid(scores)
    
    def seq2spans(self, seq):
        spans, curr_start, in_span = [], None, False
        for i in range(len(seq)):
            if not seq[i] and in_span:
                spans.append((curr_start, i))
                in_span = False
                curr_start = None
            elif seq[i] and not in_span:
                in_span = True
                curr_start = i
        
        return spans
    
    def predict(self, batchs, method):
        all_scores = []
        TP, FP, FN = 0, 0, 0
        for query_sentence, query_span, target_sentence, target_spans, labels in batchs:
            scores = self.forward(query_sentence, query_span, target_sentence, method)
            all_scores.append(scores)
            
        return all_scores
    
    def evaluate(self, batchs, method, all_scores=None):
        TP, FP, FN = 0, 0, 0
        if not all_scores:
            all_scores = self.predict(batchs, method)
        
        best_f = -np.Inf
        for threshold in np.arange(0.3,0.71,0.04):
            for batch_scores, batch_data in zip(all_scores, batchs):
                preds = (batch_scores > 0.5).tolist()
                query_sentence, query_span, target_sentence, target_spans, labels = batch_data
                for pred, true in zip(preds, target_spans):
                    pred_spans = set(self.seq2spans(pred))
                    target_set = set([tuple(r) for r in true])
                    curr_TP = len(target_set.intersection(pred_spans))
                    curr_FP = len(pred_spans) - curr_TP
                    curr_FN = len(target_set) - curr_TP
                    TP += curr_TP
                    FP + curr_FP
                    FN += curr_FN
            
            p = TP / (TP + FP) if TP + FP > 0 else 0
            r = TP / (TP + FN)
            f = 2*p*r / (p + r) if p + r > 0 else 0
            print('%.2f : %.2f %.2f %.2f' % (threshold, p, r, f))
            if f > best_f:
                best_f = f
        
        return best_f


if __name__ == "__main__":
    raw_query_sentence = "[CLS] What is the French name of the Canadian Armed Forces ? [SEP]"
    
    raw_target_sentence = ('[CLS] The Canadian Armed Forces ( CAF ; French : Forces armées canadiennes , FAC ) , or Canadian '
                           'Forces ( CF ) ( French : les Forces canadiennes , FC ), is the unified armed force of Canada'
                           ' , as constituted by the National Defence Act, which states : " The Canadian Forces are '
                           'the armed forces of Her Majesty raised by Canada and consist of one Service called the'
                           ' Canadian Armed Forces . " [SEP]')
    
    tokenizer = Tokenizer()
    query_sentence = tokenizer(raw_query_sentence)
    target_sentence = tokenizer(raw_target_sentence)
    
    query_span = [3,6] # "the French name"
    target_spans = [[i,j] for i in range(len(target_sentence)-1) for j in range(i+1,len(target_sentence))]
    
    phrase_matcher = Phrase_Matcher_Enum(100, torch.device("cuda:0"))
    matched, score = phrase_matcher(query_sentence, query_span, target_sentence, target_spans, 'mean')
    
    print(target_sentence[matched[0]:matched[1]])