import torch, torch.nn as nn, time, numpy as np

class Phrase_Matcher_Trainer:
    def __init__(self, model_, args, criterion, optimizer, predictor, evaluator):
        self.model_ = model_
        self.args = args
        self.criterion = criterion
        self.optimizer = optimizer
        self.predictor = predictor
        self.evaluator = evaluator
    
    def train_epoch(self, data):
        # save all_scores and all_labels for fast evaluation
        all_scores, total_loss, total_inst = [], 0, 0
        self.model_.train()
        for i in range(int(np.ceil(len(data)/self.args['batch']))):
            self.model_.zero_grad()
            batch = data[(i*self.args['batch']):((i+1)*self.args['batch'])]
            if self.args['train_mode'] == 'phrase':
                labels = [r[2] for r in batch]
            elif self.args['train_mode'] == 'sent':
                labels = [r[4] for r in batch][0]
            else:
                labels = [r[4] for r in batch]                
            
            labels_flat = [r for rr in labels for rrr in rr for r in rrr]
            scores = self.model_(batch, self.args["train_mode"])
            # adding softmax
            # if self.args['train_mode'] == 'sent':
                # scores = [[nn.Softmax(dim=0)(r) for r in rr] for rr in scores]
            
            loss = self.criterion(torch.cat([r for rr in scores for r in rr]), torch.Tensor(labels_flat).to(self.args['cuda_device']))
            loss.backward()
            nn.utils.clip_grad_norm_(self.model_.parameters(), 5)
            self.optimizer.step()
            all_scores += [[r.tolist() for r in rr] for rr in scores]
            total_loss += loss.tolist() * len(labels_flat)
            total_inst += len(labels_flat)
        
        print("avg loss:", total_loss/total_inst)
        return all_scores
    
    def adjust_learning_rate(self, optimizer, lr):
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    
    def train(self, train_data, dev_data, test_data):
        start = time.time()
        best_score, best_epoch, best_state_dict = -np.Inf, 0, None
        patience = 0
        for i in range(1, self.args['epochs']+1):
            patience += 1
            print()
            print("Epoch:", i)
            print()
            train_scores = self.train_epoch(train_data)
            self.adjust_learning_rate(self.optimizer, self.args['lr'] / (1 + (i) * self.args['lr_decay']))
            if self.args['train_mode'] == 'phrase':
                train_labels = [r[2] for r in train_data]
            else:
                train_labels = [r[4] for r in train_data]
            print("Eval on train (during training):")
            self.evaluator.evaluate(train_scores, train_labels, self.args["train_mode"])
            
            print("Eval on dev (after training):")
            n_iters = 1 if self.args["model"] == 'find' else 2
            for ii in range(n_iters):
                if self.args["model"] == 'fill':
                    dev_data_ = dev_data[ii]
                else:
                    dev_data_ = dev_data
                dev_scores = self.predictor.predict(self.model_, dev_data_, self.args["train_mode"])
                if self.args['train_mode'] == 'phrase':
                    dev_labels = [r[2] for r in dev_data_]
                else:
                    dev_labels = [r[4] for r in dev_data_]
                scores = self.evaluator.evaluate(dev_scores, dev_labels, self.args["train_mode"])
                score = scores['macro']['p_k'][1] # precision @ 1
            
            if score > best_score:
                best_score = score
                best_epoch = i
                best_state_dict = self.model_.dump_state_dict()
                patience = 0
                self.save_model(i, best_score)
            
            print("Best score: %.2f  epoch: %d  patience: %d" % (best_score, best_epoch, patience))
            print("Total time (minutes): %.2f" % ((time.time() - start)/60))
            
            if patience > self.args['patience']:
                break
        
        print('\n')
        print("Eval on test:")
        print()
        self.model_.load_state_dict(best_state_dict)
        n_iters = 1 if self.args["model"] == 'find' else 2
        for ii in range(n_iters):
            if self.args["model"] == 'fill':
                test_data_ = test_data[ii]
            else:
                test_data_ = test_data
            test_scores = self.predictor.predict(self.model_, test_data_, self.args["train_mode"])
            if self.args['train_mode'] == 'phrase':
                test_labels = [r[2] for r in test_data_]
            else:
                test_labels = [r[4] for r in test_data_]
            
            self.evaluator.evaluate(test_scores, test_labels, self.args["train_mode"])
        
        print()
    
    def save_model(self, epoch, score):
        torch.save({"epoch": epoch, "args": self.args, "state_dict": self.model_.dump_state_dict(), \
                    "score": score}, self.args["save_chk"])
