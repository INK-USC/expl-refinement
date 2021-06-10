import numpy as np

class Phrase_Matcher_Evaluator:
    def __init__(self, k_range = [1,3,5,10]):
        self.k_range = k_range
    
    def calc_f(self, p, r):
        return 2 * p * r / (p + r) if (p + r) else 0
    
    def evaluate(self, all_scores, all_labels, train_mode):
        all_scores = [r for rr in all_scores for r in rr]
        all_labels = [r for rr in all_labels for r in rr]
        if train_mode == 'sent':
            all_labels = [r for rr in all_labels for r in rr]
        # micro scores
        TP, FP, FN = 0, 0, 0
        TP_k, FP_k, FN_k = [{k:0 for k in self.k_range} for j in range(3)]
        # macro scores
        p, r, f = [], [], []
        p_k, r_k, f_k = [{k:[] for k in self.k_range} for j in range(3)]
        for scores, labels in zip(all_scores, all_labels):
            if not 1 in labels:
                continue
            # ranking scores
            labels_sorted = [r[1] for r in sorted(zip(scores, labels), key=lambda x:x[0], reverse=True)]
            for k in self.k_range:
                labels_k = labels_sorted[:k]
                curr_TP_k = labels_k.count(1)
                curr_FP_k = k - curr_TP_k
                curr_FN_k = labels.count(1) - curr_TP_k
                TP_k[k] += curr_TP_k
                FP_k[k] += curr_FP_k
                FN_k[k] += curr_FN_k
                curr_p_k = curr_TP_k / k
                curr_r_k = curr_TP_k / labels.count(1) if labels.count(1) else 0
                curr_f_k = self.calc_f(curr_p_k, curr_r_k)
                p_k[k].append(curr_p_k)
                r_k[k].append(curr_r_k)
                f_k[k].append(curr_f_k)
                
            
            # classification scores
            preds = [1 if s > 0.5 else 0 for s in scores]
            curr_TP = len([i for i in range(len(preds)) if preds[i]==1 and labels[i]==1])
            curr_FP = preds.count(1) - curr_TP
            curr_FN = labels.count(1) - curr_TP
            TP += curr_TP
            FP += curr_FP
            FN += curr_FN
            curr_p = curr_TP / preds.count(1) if preds.count(1) else 0
            curr_r = curr_TP / labels.count(1) if labels.count(1) else 0
            curr_f = self.calc_f(curr_p, curr_r)
            p.append(curr_p)
            r.append(curr_r)
            f.append(curr_f)
        
        micro_p = TP/(TP+FP) if (TP+FP) else 0
        micro_r = TP/(TP+FN)
        micro_f = self.calc_f(micro_p, micro_r)
        micro_p_k = {k:(TP_k[k]/(TP_k[k]+FP_k[k]) if TP_k[k]+FP_k[k] else 0) for k in self.k_range}
        micro_r_k = {k:TP_k[k]/(TP_k[k]+FN_k[k]) for k in self.k_range}
        micro_f_k = {k:self.calc_f(micro_p_k[k], micro_r_k[k]) for k in self.k_range}
        micro_scores = {'p': micro_p, 'r': micro_r, 'f': micro_f, 'p_k': micro_p_k, 'r_k': micro_r_k, 'f_k': micro_f_k}
        
        macro_p = np.mean(p)
        macro_r = np.mean(r)
        macro_f = np.mean(f)
        macro_p_k = {k:np.mean(p_k[k]) for k in self.k_range}
        macro_r_k = {k:np.mean(r_k[k]) for k in self.k_range}
        macro_f_k = {k:np.mean(f_k[k]) for k in self.k_range}
        macro_scores = {'p': macro_p, 'r': macro_r, 'f': macro_f, 'p_k': macro_p_k, 'r_k': macro_r_k, 'f_k': macro_f_k}
        
        self.print_result(macro_scores)
        return {'micro': micro_scores, 'macro': macro_scores}
    
    def print_result(self, scores):
        for k in self.k_range:
            print(" - top %d P: %.2f" % (k, scores['p_k'][k]))
            print(" - top %d R: %.2f" % (k, scores['r_k'][k]))
            print(" - top %d F: %.2f" % (k, scores['f_k'][k]))
        
        print(" - P: %.2f" % scores['p'])
        print(" - R: %.2f" % scores['r'])
        print(" - F: %.2f" % scores['f'])
    
