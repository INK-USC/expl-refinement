from Model.Phrase_Matcher import *
import torch, time, numpy as np

class Phrase_Matcher_Predictor:
    def __init__(self, batch_size):
        self.batch_size = batch_size
    
    def predict(self, model_, data, train_mode):
        all_scores = []
        model_.eval()
        for i in range(int(np.ceil(len(data)/self.batch_size))):
            batch = data[(i*self.batch_size):((i+1)*self.batch_size)]
            scores = model_(batch, train_mode)
            all_scores += [[r.tolist() for r in rr] for rr in scores]
        
        return all_scores
    
    def load_model(self, gpu_id, module='find'):
        if module == 'find':
            # model_path = "/home/xiaohuang/qa-nle/checkpoint/train_find_91.model" # best without fine-tuning BERT
            # model_path = "/home/xiaohuang/qa-nle/checkpoint/train_find_86.model" # best of all
            model_path = "/home/huihan/newest/Expl-Reg/matcher/checkpoint/clean_find.model" # untrained baseline
            # model_path = "/home/xiaohuang/qa-nle/checkpoint/train_find_baseline_train2.model" # best trained baseline

            saved = torch.load(model_path)
            self.args = saved['args']
            self.args['gpu'] = gpu_id
            self.args['cuda_device'] = torch.device("cuda:"+str(gpu_id))
            self.model_ = Phrase_Matcher(self.args)
            self.model_.load_state_dict(saved['state_dict'])
        elif module == 'fill':
            # model_path = "/home/xiaohuang/qa-nle/checkpoint/train_fill_62.model" # best without fine-tuning BERT
            model_path = "/home/xiaohuang/qa-nle/checkpoint/train_fill_69.model" # previous best of all
            # model_path = "/home/xiaohuang/qa-nle/checkpoint/train_fill_baseline.model" # untrained baseline
            # model_path = "/home/xiaohuang/qa-nle/checkpoint/train_fill_baseline_train2.model" # best trained baseline, best of all
            saved = torch.load(model_path)
            self.args = saved['args']
            self.args['gpu'] = gpu_id
            self.args['cuda_device'] = torch.device("cuda:"+str(gpu_id))
            self.model_ = Phrase_Matcher(self.args)
            self.model_.load_state_dict(saved['state_dict'])
    
    def run(self, data):
        assert self.model_ , "must call load_model first"
        all_scores = self.predict(self.model_, data, self.args['train_mode'])
        # returned scores are flattened once, need reforming
        all_scores_, i = [], 0
        for d in data:
            num_query_spans = len(d[1])
            all_scores_.append(all_scores[i:(i+num_query_spans)])
            i += num_query_spans
        
        return all_scores_

