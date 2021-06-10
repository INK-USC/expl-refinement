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
class Phrase_Matcher_Enum(nn.Module):
    def __init__(self, attn_hid, fine_tune, cuda_device, in_dim=768, output='cosine'):
        assert output in ['cosine', 'linear']
        super(Phrase_Matcher_Enum, self).__init__()
        self.attn_hid = attn_hid
        self.fine_tune = fine_tune
        self.cuda_device = cuda_device
        self.in_dim = in_dim
        self.output = output
        self.bert = BERT(self.cuda_device, self.fine_tune)
        if not self.fine_tune:
            self.bert.eval()
        self.M = nn.Parameter(torch.Tensor(self.in_dim, self.attn_hid))
        nn.init.uniform_(self.M, -np.sqrt(1/self.in_dim), np.sqrt(1/self.in_dim))
        self.attn_v = nn.Parameter(torch.Tensor(self.attn_hid))
        nn.init.uniform_(self.attn_v, -np.sqrt(1/self.attn_hid), np.sqrt(1/self.attn_hid))
        self.D = nn.Parameter(torch.Tensor(1, self.in_dim))
        nn.init.uniform_(self.D, -np.sqrt(1/self.in_dim), np.sqrt(1/self.in_dim))
        self.softmax = nn.Softmax(dim=1)
        if self.output == 'cosine':
            self.out_layer = nn.CosineSimilarity(dim=1)
        else:
            self.out_layer = nn.Linear(self.in_dim*2, 1)
        self.to(self.cuda_device)
    
    # only attn will be trained/saved/loaded
    def get_trainable(self):
        if self.fine_tune:
            bert_params = self.bert.state_dict()
            for k in bert_params:
                bert_params[k] = bert_params[k].cpu()
            returned = [self.M.detach().cpu(), self.attn_v.detach().cpu(), self.D.detach().cpu(), bert_params]
        else:
            returned = [self.M.detach().cpu(), self.attn_v.detach().cpu(), self.D.detach().cpu()]
        
        if self.output != 'cosine':
            out_params = self.out_layer.state_dict()
            for k in out_params:
                out_params[k] = out_params[k].cpu()
            returned.append(out_params)
        
        return returned
    
    def load_trainable(self, state_dict):
        self.M = nn.Parameter(state_dict[0].to(self.cuda_device))
        self.attn_v = nn.Parameter(state_dict[1].to(self.cuda_device))
        self.D = nn.Parameter(state_dict[2].to(self.cuda_device))
        if self.fine_tune:
            self.bert.load_state_dict(state_dict[3])
        
        if self.output != 'cosine':
            self.out_layer.load_state_dict(state_dict[-1])
    
    def train(self):
        self.M.requires_grad = True
        self.attn_v.requires_grad = True
        self.D.requires_grad = True
        if self.fine_tune:
            self.bert.train()
        if self.output != 'cosine':
            self.out_layer.train()
    
    def eval(self):
        self.M.requires_grad = False
        self.attn_v.requires_grad = False
        self.D.requires_grad = False
        self.bert.eval()
        self.out_layer.eval()
    
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
    
    def forward(self, query_sentence, query_span, target_sentence, target_spans, method):
        assert method in ['attn', 'mean']
        
        query_sent_encoding = self.bert(query_sentence) # batch_size * sentence_length * hidden_size
        query_encoding = self.get_phrase_encodings(query_sent_encoding, query_span, method)[0]
        
        target_sent_encoding = self.bert(target_sentence)
        target_encodings = self.get_phrase_encodings(target_sent_encoding, target_spans, method)
        scores = []
        for target_encoding in target_encodings:
            if self.output == 'cosine':
                score = self.out_layer(query_encoding, target_encoding)
            else:
                score = self.out_layer(torch.cat([query_encoding, target_encoding], 1).squeeze(0))
            
            scores.append(score)
        
        return torch.sigmoid(torch.cat(scores))


class Phrase_Matcher_Enum_lstm(nn.Module):
    def __init__(self, attn_hid, fine_tune, cuda_device, in_dim=768, output='cosine'):
        assert output in ['cosine', 'linear']
        super(Phrase_Matcher_Enum_lstm, self).__init__()
        self.attn_hid = attn_hid
        self.fine_tune = fine_tune
        self.cuda_device = cuda_device
        self.in_dim = in_dim
        self.output = output
        self.bert = BERT(self.cuda_device, self.fine_tune)
        if not self.fine_tune:
            self.bert.eval()
        self.lstm = nn.LSTM(self.in_dim, self.attn_hid, 1, bidirectional=True, batch_first=True)
        self.D = nn.Parameter(torch.Tensor(1, self.attn_hid*2))
        nn.init.uniform_(self.D, -np.sqrt(1/(self.attn_hid*2)), np.sqrt(1/(self.attn_hid*2)))
        self.softmax = nn.Softmax(dim=1)
        if self.output == 'cosine':
            self.out_layer = nn.CosineSimilarity(dim=1)
        else:
            self.out_layer = nn.Linear(self.attn_hid*2, 1)
        self.to(self.cuda_device)
    
    # only attn will be trained/saved/loaded
    def get_trainable(self):
        if self.fine_tune:
            bert_params = self.bert.state_dict()
            for k in bert_params:
                bert_params[k] = bert_params[k].cpu()
            lstm_params = self.lstm.state_dict()
            for k in lstm_params:
                lstm_params[k] = lstm_params[k].cpu()
            returned = [self.D.detach().cpu(), lstm_params, bert_params]
        else:
            returned = [self.D.detach().cpu(), lstm_params]
        
        if self.output != 'cosine':
            out_params = self.out_layer.state_dict()
            for k in out_params:
                out_params[k] = out_params[k].cpu()
            returned.append(out_params)
        
        return returned
    
    def load_trainable(self, state_dict):
        self.D = nn.Parameter(state_dict[0].to(self.cuda_device))
        self.lstm.load_state_dict(state_dict[1])
        if self.fine_tune:
            self.bert.load_state_dict(state_dict[2])
        
        if self.output != 'cosine':
            self.out_layer.load_state_dict(state_dict[-1])
    
    def train(self):
        self.D.requires_grad = True
        self.lstm.train()
        if self.fine_tune:
            self.bert.train()
        if self.output != 'cosine':
            self.out_layer.train()
    
    def eval(self):
        self.D.requires_grad = False
        self.lstm.eval()
        self.bert.eval()
        self.out_layer.eval()
    
    def get_phrase_encodings(self, sent_encoding, phrase_spans, method):
        assert sent_encoding.shape[0] == 1
        
        phrase_encodings = []
        for span in phrase_spans:
            phrase_encoding = torch.cat([sent_encoding[:,(span[1]-1),:self.attn_hid], sent_encoding[:,(span[0]),self.attn_hid:]], 1)
            phrase_encoding_ = phrase_encoding * self.D
            phrase_encodings.append(phrase_encoding_)
        
        return phrase_encodings
    
    def forward(self, query_sentence, query_span, target_sentence, target_spans, method):
        assert method in ['attn', 'mean']
        
        query_sent_encoding, _ = self.lstm(self.bert(query_sentence)) # batch_size * sentence_length * hidden_size
        query_encoding = self.get_phrase_encodings(query_sent_encoding, query_span, method)[0]
        
        target_sent_encoding, _ = self.lstm(self.bert(target_sentence))
        target_encodings = self.get_phrase_encodings(target_sent_encoding, target_spans, method)
        scores = []
        for target_encoding in target_encodings:
            if self.output == 'cosine':
                score = self.out_layer(query_encoding, target_encoding)
            else:
                score = self.out_layer(torch.cat([query_encoding, target_encoding], 1).squeeze(0))
            
            scores.append(score)
        
        return torch.sigmoid(torch.cat(scores))



class Phrase_Matcher_Enum_2(nn.Module):
    def __init__(self, attn_hid, word2idx, word_emb, if_lstm, cuda_device, in_dim=100, output='cosine'):
        assert output in ['cosine', 'linear']
        super(Phrase_Matcher_Enum_2, self).__init__()
        self.attn_hid = attn_hid
        self.if_lstm = if_lstm
        self.cuda_device = cuda_device
        self.in_dim = in_dim
        self.output = output
        self.word2idx = word2idx
        self.word_emb = nn.Embedding.from_pretrained(nn.Parameter(torch.FloatTensor(word_emb)), freeze=False)
        if self.if_lstm:
            self.lstm = nn.LSTM(self.word_emb.weight.shape[1], int(self.in_dim/2), 1, bidirectional=True, batch_first=True)
        self.M = nn.Parameter(torch.Tensor(self.in_dim, self.attn_hid))
        nn.init.uniform_(self.M, -np.sqrt(1/self.in_dim), np.sqrt(1/self.in_dim))
        self.attn_v = nn.Parameter(torch.Tensor(self.attn_hid))
        nn.init.uniform_(self.attn_v, -np.sqrt(1/self.attn_hid), np.sqrt(1/self.attn_hid))
        self.D = nn.Parameter(torch.Tensor(1, self.in_dim))
        nn.init.uniform_(self.D, -np.sqrt(1/self.in_dim), np.sqrt(1/self.in_dim))
        self.softmax = nn.Softmax(dim=1)
        if self.output == 'cosine':
            self.out_layer = nn.CosineSimilarity(dim=1)
        else:
            self.out_layer = nn.Linear(self.in_dim*2, 1)
        self.to(self.cuda_device)
    
    # only attn will be trained/saved/loaded
    def get_trainable(self):
        if self.if_lstm:
            lstm_params = self.lstm.state_dict()
            for k in lstm_params:
                lstm_params[k] = lstm_params[k].cpu()
            return [self.M.detach().cpu(), self.attn_v.detach().cpu(), self.D.detach().cpu(), lstm_params]
        else:
            word_params = self.word_emb.state_dict()
            for k in word_params:
                word_params[k] = word_params[k].cpu()
            return [self.M.detach().cpu(), self.attn_v.detach().cpu(), self.D.detach().cpu(), word_params]
    
    def load_trainable(self, state_dict):
        self.M = nn.Parameter(state_dict[0].to(self.cuda_device))
        self.attn_v = nn.Parameter(state_dict[1].to(self.cuda_device))
        self.D = nn.Parameter(state_dict[2].to(self.cuda_device))
        if self.if_lstm:
            self.lstm.load_state_dict(state_dict[-1])
        else:
            self.word_emb.load_state_dict(state_dict[-1])
    
    def train(self):
        self.M.requires_grad = True
        self.attn_v.requires_grad = True
        self.D.requires_grad = True
        if self.if_lstm:
            self.lstm.train()
        else:
            self.word_emb.train()
    
    def eval(self):
        self.M.requires_grad = False
        self.attn_v.requires_grad = False
        self.D.requires_grad = False
        if self.if_lstm:
            self.lstm.eval()
        self.word_emb.eval()
    
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
    
    def forward(self, query_sentence, query_span, target_sentence, target_spans, method):
        assert method in ['attn', 'mean']
        query_sentence_ = torch.LongTensor([[self.word2idx[r] if r in self.word2idx else self.word2idx['<unk>'] for r in rr] for rr in query_sentence]).to(self.cuda_device)
        target_sentence_ = torch.LongTensor([[self.word2idx[r] if r in self.word2idx else self.word2idx['<unk>'] for r in rr] for rr in target_sentence]).to(self.cuda_device)
        if self.if_lstm:
            query_sent_encoding, _ = self.lstm(self.word_emb(query_sentence_)) # batch_size * sentence_length * hidden_size
            target_sent_encoding, _ = self.lstm(self.word_emb(target_sentence_))
        else:
            query_sent_encoding = self.word_emb(query_sentence_)
            target_sent_encoding = self.word_emb(target_sentence_)
        
        query_encoding = self.get_phrase_encodings(query_sent_encoding, query_span, method)[0]
        target_encodings = self.get_phrase_encodings(target_sent_encoding, target_spans, method)
        scores = []
        for target_encoding in target_encodings:
            if self.output == 'cosine':
                score = self.out_layer(query_encoding, target_encoding)
            else:
                score = self.out_layer(torch.cat([query_encoding, target_encoding], 1).squeeze(0))
            
            scores.append(score)
        
        return torch.sigmoid(torch.cat(scores))


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