import torch
import numpy as np
import pickle
import os
from tqdm import tqdm
from torch.nn import functional as F
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler,TensorDataset
import logging
from torch.nn import CrossEntropyLoss, MSELoss

logger = logging.getLogger(__name__)


class DistillLossWrapper:
    def __init__(self, model, processor, configs, class_weight):
        self.pred_file = configs.distill_pred_cache
        #self.distill_data_dir = configs.distill_data_dir

        self.processor = processor
        self.configs = configs
        self.model = model
        self.class_weight = class_weight
        self.data_loader = None
        self.batch_iterator = None

    def get_labeled_dataset_cached(self, matched_data: TensorDataset, exclude_unmatched_data):
        device = 'cuda'
        if os.path.isfile(self.pred_file):
            with open(self.pred_file,'rb') as f:
                pred_data = pickle.load(f)
            all_input_ids, all_input_mask, all_segment_ids, all_preds = \
                pred_data['all_input_ids'].to(device), pred_data['all_input_mask'].to(device), \
                pred_data['all_segment_ids'].to(device), pred_data['all_preds'].to(device)
        else:
            all_input_ids, all_input_mask, all_segment_ids, all_preds = [], [], [], []

            unlabeled_data_loader = self.processor.get_dataloader('train', batch_size=self.configs.train_batch_size)
            with torch.no_grad():
                for batch in tqdm(unlabeled_data_loader, desc="Evaluating"):

                    input_ids, input_mask, segment_ids, label_ids = batch.text, batch.input_mask, batch.segment_id,\
                                                                     None

                    input_ids = input_ids.to(device)
                    input_mask = input_mask.to(device)
                    segment_ids = segment_ids.to(device)
                    # label_ids = label_ids.to(device)

                    logits = self.model(input_ids, segment_ids, input_mask, labels=None)
                    #pred_labels = np.argmax(logits.cpu().numpy(), axis=1)
                    #print(logits)
                    #pred_probs = F.softmax(logits, -1)
                    #print(pred_labels)
                    #exit(1)
                    all_input_ids.append(input_ids)
                    all_input_mask.append(input_mask)
                    all_segment_ids.append(segment_ids)
                    all_preds.append(logits)
            all_input_ids, all_input_mask, all_segment_ids, all_preds = \
                [torch.cat(x, 0) for x in [all_input_ids, all_input_mask, all_segment_ids, all_preds]]
            with open(self.pred_file, 'wb') as wf:
                pred_data = {
                    'all_input_ids': all_input_ids.cpu(),
                    'all_input_mask': all_input_mask.cpu(),
                    'all_segment_ids': all_segment_ids.cpu(),
                    'all_preds': all_preds.cpu()
                }
                pickle.dump(pred_data, wf)

        matched_data_hashes = set()
        train_input_ids_np = [x[0].cpu().numpy() for x in matched_data]
        #train_input_ids_np = matched_data[0].cpu().numpy()
        for i in range(len(matched_data)):
            hash_v = hash(tuple(list(train_input_ids_np[i])))
            matched_data_hashes.add(hash_v)
        
        if exclude_unmatched_data:
            selector = [True] * all_input_ids.size(0)
            all_input_ids_np = all_input_ids.cpu().numpy()
            for i in range(all_input_ids.size(0)):
                hash_v = hash(tuple(list(all_input_ids_np[i])))
                if hash_v not in matched_data_hashes:
                    selector[i] = False
            selector = torch.ByteTensor(selector)
            all_input_ids, all_input_mask, all_segment_ids, all_preds = \
                all_input_ids[selector], all_input_mask[selector], all_segment_ids[selector], all_preds[selector]
            
        labeled_dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_preds)
        return labeled_dataset

    def initialize_dataloader(self, matched_data: TensorDataset, exclude_unmatched_data=True):
        labeled_dataset = self.get_labeled_dataset_cached(matched_data, exclude_unmatched_data)
        train_sampler = RandomSampler(labeled_dataset)
        train_dataloader = DataLoader(labeled_dataset, sampler=train_sampler, batch_size=self.configs.train_batch_size)
        self.data_loader = train_dataloader
        self.reset_iterator()

    def reset_iterator(self):
        self.batch_iterator = iter(self.data_loader)

    def do_distill_step(self):
        """
        draw a batch from the batch iterator and perform a distillation step
        if StopIteration is raised, reset the iterator automatically
        """
        device = 'cuda'
        try:
            batch = next(self.batch_iterator)
        except StopIteration:
            self.reset_iterator()
            batch = next(self.batch_iterator)

        input_ids, input_mask, segment_ids, ref_pred_logits = batch
        ref_pred_labels = torch.LongTensor(np.argmax(ref_pred_logits.cpu().numpy(), axis=1))

        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        ref_pred_labels = ref_pred_labels.to(device)
        

        logits = self.model(input_ids, segment_ids, input_mask, labels=None)
        loss_fct = CrossEntropyLoss(self.class_weight)
        loss = self.configs.distill_strength * loss_fct(logits.view(-1, 2), ref_pred_labels.view(-1))
        #log_probs = F.log_softmax(logits, -1)
        #loss = self.configs.distill_strength * F.kl_div(log_probs, ref_pred_probs)
        return loss

    def __len__(self):
        return len(self.data_loader) if self.data_loader is not None else 0

