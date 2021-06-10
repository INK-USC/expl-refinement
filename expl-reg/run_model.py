# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Running BERT finetuning & evaluation on hate speech classification datasets.

Integrated with SOC explanation regularization
"""

from __future__ import absolute_import, division, print_function

import argparse
import csv
import logging
import os
import random
import sys
import json

import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from torch import nn, optim
from torch.nn import functional as F
from tqdm import tqdm, trange

from torch.nn import CrossEntropyLoss, MSELoss
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef, f1_score
from sklearn.metrics import precision_score, recall_score, roc_auc_score, roc_curve, confusion_matrix

from bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE, WEIGHTS_NAME, CONFIG_NAME
from bert.modeling import BertForSequenceClassification, BertConfig
from bert.tokenization import BertTokenizer
from bert.optimization import BertAdam, WarmupLinearSchedule

from bert.modeling_roberta import RobertaForSequenceClassification, RobertaConfig
from bert.tokenization_roberta import RobertaTokenizer

from hiex.lm import AttnBiLSTM

from loader import GabProcessor, AmazonProcessor, YelpProcessor, SstProcessor, WSProcessor, HatevalProcessor, NytProcessor, convert_examples_to_features, convert_examples_with_advices_to_features
from utils.config import configs, combine_args

# for hierarchical explanation algorithms
from hiex import SamplingAndOcclusionExplain, SOCWithInteraction, GradientExplainer
from distill import DistillLossWrapper
from baseline import L2RegWrapper

logger = logging.getLogger(__name__)

def simple_accuracy(preds, labels):
    return (preds == labels).mean()

def compute_fprd(preds, labels, pred_probs, subgroups):
    def get_fpr(p, l):
        C = confusion_matrix(y_true=l, y_pred=p)
        fp = C[0][1]
        tn = C[0][0]
        return fp/(fp+tn)
    
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds)

    fpr = get_fpr(preds, labels)
    fprd = 0

    for group in subgroups:
        if len(group) == 0:
            continue
        fprd += abs(fpr - get_fpr(preds[group], labels[group]))

    return {
        "acc": acc,
        "f1": f1,
        "fprd": fprd
    }

def acc_and_f1(preds, labels, pred_probs):
    # C = confusion_matrix(y_true=labels, y_pred=preds)
    # fp = C[0][1]
    # tn = C[0][0]
    # fpr = fp/(fp+tn)
    # fn = C[1][0]
    # tp = C[1][1]
    # fnr = fn/(fn+tp)
    
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds)
    p, r = precision_score(y_true=labels, y_pred=preds), recall_score(y_true=labels, y_pred=preds)

    try:
        roc = roc_auc_score(y_true=labels, y_score=pred_probs[:,1])
    except ValueError:
        roc = 0.
    
    return {
        "acc": acc,
        "f1": f1,
        "precision": p,
        "recall": r,
        "auc_roc": roc
    }


def pearson_and_spearman(preds, labels):
    pearson_corr = pearsonr(preds, labels)[0]
    spearman_corr = spearmanr(preds, labels)[0]
    return {
        "pearson": pearson_corr,
        "spearmanr": spearman_corr,
        "corr": (pearson_corr + spearman_corr) / 2,
    }


def compute_metrics(task_name, preds, labels, pred_probs, subgroups):
    assert len(preds) == len(labels)

    if task_name == 'iptt':
        return compute_fprd(preds, labels, pred_probs, subgroups)
    
    return acc_and_f1(preds, labels, pred_probs)

def compute_eer_threshold(preds, labels):
    fpr, tpr, threshold = roc_curve(labels, preds[:, 1] - preds[:, 0], pos_label=1)
    fnr = 1 - tpr
    eer_threshold = threshold[np.nanargmin(np.absolute((fnr - fpr)))]
    return eer_threshold

def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--bert_model", default=None, type=str, required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                             "bert-base-multilingual-cased, bert-base-chinese."
                             "RoBERTa pre-trained model: roberta-base, roberta-large, roberta-large-mnli.")
    parser.add_argument("--base_model", choices=['bert', 'roberta', 'lstm'], default='bert')
    parser.add_argument("--task_name",
                        default=None,
                        type=str,
                        required=True,
                        help="The name of the task to train.")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--negative_weight", default=1., type=float)

    # matched regularization files
    parser.add_argument("--train_advice", default=None)
    parser.add_argument("--dev_advice", default=None)
    parser.add_argument("--no_label", action='store_true')
    parser.add_argument("--mix_label", action='store_true') # w/ advice: only reg, w/o advice: only supervision
    parser.add_argument("--mix_label_ver2", action = 'store_true')  # conf = 0: only reg

    # if true, use test data instead of val data
    parser.add_argument("--test", action='store_true')

    # Explanation specific arguments below

    # whether run explanation algorithms
    parser.add_argument("--explain", action='store_true', help='if true, explain test set predictions')
    parser.add_argument("--debug", action='store_true')

    # which algorithm to run
    parser.add_argument("--algo", choices=['soc', 'gradshap'], default='soc')
    parser.add_argument("--gradshap_softmax", action='store_true')

    # the output filename without postfix
    parser.add_argument("--output_filename", default='temp.tmp')

    parser.add_argument("--use_eer", action='store_true')
    parser.add_argument("--less_detail", action='store_true')

    # see utils/config.py
    parser.add_argument("--use_padding_variant", action='store_true')
    parser.add_argument("--mask_outside_nb", action='store_true')
    parser.add_argument("--nb_range", type=int)
    parser.add_argument("--sample_n", type=int)
    parser.add_argument("--reg_interaction", action='store_true')
    parser.add_argument("--only_interaction", action='store_true')
    parser.add_argument("--confidence", action='store_true')

    # whether use explanation regularization
    parser.add_argument("--reg_explanations", action='store_true')
    parser.add_argument("--reg_strength", type=float)
    parser.add_argument("--reg_mse", action='store_true')
    parser.add_argument("--reg_interaction_strength", type=float)

    # configurations about distillation
    parser.add_argument("--do_distill", action='store_true')
    parser.add_argument("--distill_pred_cache")
    parser.add_argument("--distill_exclude", action='store_true', help="if exclude unmatched data")
    #parser.add_argument("--distill_data_dir")
    parser.add_argument("--distill_strength", type=float)
    parser.add_argument("--distill_freq", type=int, help="the number of batch drawn to compute"
                                                              "distillation loss at each training iter")
    #configuration about l2 reg
    parser.add_argument("--do_l2reg", action='store_true')
    parser.add_argument("--l2reg_strength", type=float)

    # whether discard other neutral words during regularization. default: False
    parser.add_argument("--discard_other_nw", action='store_false', dest='keep_other_nw')

    # whether remove neutral words when loading datasets
    parser.add_argument("--remove_nw", action='store_true')

    # if true, generate hierarchical explanations instead of word level outputs.
    # Only useful when the --explain flag is also added.
    parser.add_argument("--hiex", action='store_true')
    parser.add_argument("--hiex_tree_height", default=5, type=int)

    # whether add the sentence itself to the sample set in SOC
    parser.add_argument("--hiex_add_itself", action='store_true')

    # the directory where the lm is stored
    parser.add_argument("--lm_dir", default='runs/lm')

    # if configured, only generate explanations for instances with given line numbers
    parser.add_argument("--hiex_idxs", default=None)
    # if true, use absolute values of explanations for hierarchical clustering
    parser.add_argument("--hiex_abs", action='store_true')

    # if either of the two is true, only generate explanations for positive / negative instances
    parser.add_argument("--only_positive", action='store_true')
    parser.add_argument("--only_negative", action='store_true')

    # stop after generating x explanation
    parser.add_argument("--stop", default=100000000, type=int)

    # early stopping with decreasing learning rate. 0: direct exit when validation F1 decreases
    parser.add_argument("--early_stop", default=5, type=int)
    parser.add_argument("--early_stop_iter", default=200, type=int)
    parser.add_argument("--early_stop_restore", action="store_true")

    # other external arguments originally here in pytorch_transformers

    parser.add_argument("--cache_dir",
                        default="",
                        type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    parser.add_argument('--server_ip', type=str, default='', help="Can be used for distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="Can be used for distant debugging.")
    args = parser.parse_args()

    combine_args(configs, args)
    args = configs

    #if args.distill_strength == 0:
    #    args.do_distill = False

    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()
    
    model_classes = {
        'bert': (BertConfig, BertForSequenceClassification, BertTokenizer),
        'roberta': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
        'lstm': (None, AttnBiLSTM, BertTokenizer),
    }

    config_class, model_class, tokenizer_class = model_classes[args.base_model]

    processors = {
        'gab': GabProcessor,
        'yelp': YelpProcessor,
        'sst': SstProcessor,
        'ws': WSProcessor,
        'nyt': NytProcessor,
        'amazon': AmazonProcessor,
        'hateval': HatevalProcessor,
        'iptt': GabProcessor
    }

    output_modes = {
        'gab': 'classification',
        'yelp': 'classification',
        'sst': 'classification',
        'ws': 'classification',
        'nyt': 'classification',
        'amazon': 'classification',
        'hateval': 'classification',
        'iptt': 'classification'
    }

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)

    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    #if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train:
    #    raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # save configs
    if args.do_train:
        f = open(os.path.join(args.output_dir, 'args.json'), 'w')
        json.dump(args.__dict__, f, indent=4)
        f.close()

    task_name = args.task_name.lower()

    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    if args.base_model == 'lstm':
        tokenizer = tokenizer_class.from_pretrained('bert-base-uncased', do_lower_case=args.do_lower_case)
    else:
        tokenizer = tokenizer_class.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)
    
    processor = processors[task_name](configs, tokenizer=tokenizer)
    output_mode = output_modes[task_name]

    label_list = processor.get_labels()
    num_labels = len(label_list)

    train_examples = None
    num_train_optimization_steps = None
    if args.do_train:
        if hasattr(args, 'train_advice'):
            train_examples = processor.get_train_examples_with_advices(args.data_dir, args.train_advice)
        else:
            train_examples = processor.get_train_examples(args.data_dir)
        
        num_train_optimization_steps = int(
            len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs
        if args.local_rank != -1:
            num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()

    
    tokenizer_vocab = tokenizer.encoder if args.base_model in ['roberta'] else tokenizer.vocab
    
    # Prepare model
    if args.base_model == 'lstm':
        model = model_class(args, tokenizer_vocab, device=device)
        if not args.do_train:
            model.load_state_dict(torch.load(os.path.join(args.output_dir, WEIGHTS_NAME)))
        elif args.bert_model != 'bert-base-uncased':
            model.load_state_dict(torch.load(os.path.join(args.bert_model, WEIGHTS_NAME)))
    else:
        config = config_class.from_pretrained(args.bert_model, num_labels=num_labels)
        cache_dir = args.cache_dir if args.cache_dir else os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE),
                                                                    'distributed_{}'.format(args.local_rank))
        if args.do_train:
            model = model_class.from_pretrained(args.bert_model, config=config, cache_dir=cache_dir)
        else:
            model = model_class.from_pretrained(args.output_dir, config=config, cache_dir=cache_dir)
    model.to(device)

    if args.fp16:
        model.half()

    if args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        model = DDP(model)
    # elif n_gpu > 1:
    #     model = torch.nn.DataParallel(model)

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    if args.fp16:
        try:
            from apex.optimizers import FP16_Optimizer
            from apex.optimizers import FusedAdam
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        optimizer = FusedAdam(optimizer_grouped_parameters,
                              lr=args.learning_rate,
                              bias_correction=False,
                              max_grad_norm=1.0)
        if args.loss_scale == 0:
            optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
        else:
            optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.loss_scale)
        warmup_linear = WarmupLinearSchedule(warmup=args.warmup_proportion,
                                             t_total=num_train_optimization_steps)

    else:
        if args.do_train:
            if args.base_model == 'lstm':
                optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
            else:
                optimizer = BertAdam(optimizer_grouped_parameters,
                                 lr=args.learning_rate,
                                 warmup=args.warmup_proportion,
                                 t_total=num_train_optimization_steps)

    global_step = 0
    nb_tr_steps = 0
    tr_loss, tr_reg_loss = 0, 0
    tr_reg_cnt = 0
    epoch = -1
    val_best_f1 = -1
    val_best_loss = 1e10
    early_stop_countdown = args.early_stop

    use_acc_early_stop = False
    val_best_acc = -1

    if args.base_model in ['roberta']:
        args.lm_dir = 'runs/lm_roberta'
    
    if args.reg_explanations:        
        if args.algo == 'soc':
            train_lm_dataloder = processor.get_dataloader('train', configs.train_batch_size)
            dev_lm_dataloader = processor.get_dataloader('dev', configs.train_batch_size)
            explainer = SOCWithInteraction(model, configs, tokenizer, device=device, vocab=tokenizer_vocab,
                                            train_dataloader=train_lm_dataloder,
                                            dev_dataloader=dev_lm_dataloader,
                                            lm_dir=args.lm_dir,
                                            output_path=os.path.join(configs.output_dir,
                                                                        configs.output_filename),
                                            )
        elif args.algo == 'gradshap':
            #val_dataloader = processor.get_dataloader('dev', configs.train_batch_size)
            explainer = GradientExplainer(model, configs, tokenizer)
    else:
        explainer = None

    if args.do_train:
        epoch = 0

        if args.reg_explanations:
            train_features = convert_examples_with_advices_to_features(
                train_examples, label_list, args.max_seq_length, tokenizer, output_mode, configs,
                cls_token_at_end=False,            # xlnet has a cls token at the end
                cls_token=tokenizer.cls_token,
                cls_token_segment_id=0,
                sep_token=tokenizer.sep_token,
                sep_token_extra=bool(args.base_model in ['roberta']), # roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
                pad_on_left=False,                 # pad on the left for xlnet
                pad_token=tokenizer.encoder[tokenizer.pad_token] if args.base_model in ['roberta'] else tokenizer.vocab[tokenizer.pad_token],
                pad_token_segment_id=0,)
        else:
            train_features = convert_examples_to_features(train_examples, label_list, args.max_seq_length, tokenizer, output_mode, configs,
                cls_token_at_end=False,            # xlnet has a cls token at the end
                cls_token=tokenizer.cls_token,
                cls_token_segment_id=0,
                sep_token=tokenizer.sep_token,
                sep_token_extra=bool(args.base_model in ['roberta']), # roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
                pad_on_left=False,                 # pad on the left for xlnet
                pad_token=tokenizer.encoder[tokenizer.pad_token] if args.base_model in ['roberta'] else tokenizer.vocab[tokenizer.pad_token],
                pad_token_segment_id=0,
            )
        
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_optimization_steps)
        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)

        if output_mode == "classification":
            all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
        elif output_mode == "regression":
            all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.float)

        if args.reg_explanations:
            all_importances = torch.tensor([f.importances for f in train_features], dtype=torch.long)
            all_interactions = torch.tensor([f.interactions for f in train_features], dtype=torch.long)

            if args.confidence or args.mix_label_ver2:
                all_confidences = torch.tensor([f.confidence for f in train_features], dtype=torch.float)
                train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids,
                                        all_importances, all_interactions, all_confidences)
            else:
                train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids,
                                        all_importances, all_interactions)
        else:
            train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
        
        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

        class_weight = torch.FloatTensor([args.negative_weight, 1]).to(device)

        if args.algo == 'gradshap' and explainer is not None:
            bg_data = TensorDataset(all_input_ids)
            explainer.init_bg_dataset(bg_data)

        ## distillation wrapper
        if args.do_distill:
            distill_wrapper = DistillLossWrapper(model, processor, args, class_weight)
            distill_wrapper.initialize_dataloader(train_data, args.distill_exclude)
        else:
            distill_wrapper = None

        # l2 reg wrapper
        if args.do_l2reg:
            l2reg_wrapper = L2RegWrapper(model)

        model.train()
        for _ in trange(int(args.num_train_epochs), desc="Epoch"):
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                batch = tuple(t.to(device) for t in batch)

                confidences = None
                if args.reg_explanations:
                    if args.confidence or args.mix_label_ver2:
                        input_ids, input_mask, segment_ids, label_ids, importances, interactions, confidences = batch
                    else:
                        input_ids, input_mask, segment_ids, label_ids, importances, interactions = batch
                else:
                    input_ids, input_mask, segment_ids, label_ids = batch
                
                # define a new function to compute loss values for both output_modes
                inputs = {'input_ids': input_ids,
                          'attention_mask': input_mask,
                          'token_type_ids': segment_ids if args.base_model == 'bert' else None,  # XLM don't use segment_ids
                          'labels': None}
                logits = model(**inputs)

                if args.mix_label:
                    use_supervision = torch.zeros(args.train_batch_size).to(device)
                    for b in range(args.train_batch_size):
                        if importances[b].nonzero().nelement() == 0: # no advice
                            use_supervision[b] = 1
                
                mix_label_ver2_error = False
                if args.mix_label_ver2:
                    if confidences.shape[0] != args.train_batch_size:
                        print(confidences.shape)
                        mix_label_ver2_error = True
                    else:
                        use_supervision = torch.ones(args.train_batch_size).to(device)
                        for b in range(args.train_batch_size):
                            if confidences[b] == 0:
                                use_supervision[b] = 0

                if output_mode == "classification":
                    if args.mix_label or (args.mix_label_ver2 and not mix_label_ver2_error):
                        loss_fct = CrossEntropyLoss(class_weight, reduction='none')
                        loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1))
                        loss = (loss * use_supervision).sum() / use_supervision.nonzero().nelement()
                    else:
                        loss_fct = CrossEntropyLoss(class_weight)
                        loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1))
                elif output_mode == "regression":
                    loss_fct = MSELoss()
                    loss = loss_fct(logits.view(-1), label_ids.view(-1))

                if n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                
                if not args.no_label:
                    tr_loss += loss.item()
                
                if args.fp16:
                    optimizer.backward(loss)
                else:
                    if not args.no_label:
                        if args.do_l2reg:
                            l2_loss = l2reg_wrapper(model)
                            loss += args.l2reg_strength * l2_loss
                        loss.backward()
                    else:
                        del loss, loss_fct, logits

                # regularize explanations
                # NOTE: backward performed inside this function to prevent OOM

                if args.reg_explanations:
                    if args.confidence:
                        reg_loss, reg_cnt = explainer.compute_expl_loss_with_advice(input_ids, input_mask, segment_ids, label_ids,
                                                                            importances, interactions, confidences,
                                                                            do_backprop=True)
                    else:
                        reg_loss, reg_cnt = explainer.compute_expl_loss_with_advice(input_ids, input_mask, segment_ids, label_ids,
                                                                            importances, interactions,
                                                                            do_backprop=True)
                    tr_reg_loss += reg_loss # float
                    tr_reg_cnt += reg_cnt

                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1

                # do distillation
                if args.do_distill:
                    for i in range(args.distill_freq):
                        distill_loss = distill_wrapper.do_distill_step()
                        distill_loss.backward()

                if (step + 1) % args.gradient_accumulation_steps == 0:
                    if args.fp16:
                        # modify learning rate with special warm up BERT uses
                        # if args.fp16 is False, BertAdam is used that handles this automatically
                        lr_this_step = args.learning_rate * warmup_linear.get_lr(
                            global_step / num_train_optimization_steps,
                            args.warmup_proportion)
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr_this_step             
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1
                
                if global_step % args.early_stop_iter == 0:
                    val_result = validate(args, model, processor, tokenizer, output_mode, label_list, device,
                                          num_labels,
                                          task_name, tr_loss, global_step, epoch, explainer)
                    val_acc, val_f1 = val_result['acc'], val_result['f1']

                    if val_best_f1 == -1 and val_f1 == 0 and use_acc_early_stop is False: # meaningless to compare f1
                        logger.warning('F1 is 0, using accuracy for early stop instead')
                        use_acc_early_stop = True
                    
                    if use_acc_early_stop:
                        if val_acc > val_best_acc:
                            val_best_acc = val_acc
                            if args.local_rank == -1 or torch.distributed.get_rank() == 0:
                                save_model(args, model, tokenizer, num_labels)
                            
                            if args.early_stop_restore:
                                early_stop_countdown = args.early_stop
                                logger.info("Early stop countdown restored to %d" % early_stop_countdown)
                        else:
                            # halve the learning rate
                            for param_group in optimizer.param_groups:
                                param_group['lr'] *= 0.5
                            early_stop_countdown -= 1
                            logger.info("Reducing learning rate... Early stop countdown %d" % early_stop_countdown)
                    else:
                        if val_f1 > val_best_f1:
                            val_best_f1 = val_f1
                            if args.local_rank == -1 or torch.distributed.get_rank() == 0:
                                save_model(args, model, tokenizer, num_labels)
                            
                            if args.early_stop_restore:
                                early_stop_countdown = args.early_stop
                                logger.info("Early stop countdown restored to %d" % early_stop_countdown)
                        else:
                            # halve the learning rate
                            for param_group in optimizer.param_groups:
                                param_group['lr'] *= 0.5
                            early_stop_countdown -= 1
                            logger.info("Reducing learning rate... Early stop countdown %d" % early_stop_countdown)
                    if early_stop_countdown < 0:
                        break
            if early_stop_countdown < 0:
                break
            epoch += 1

    if args.do_eval and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        if not args.explain:
            validate(args, model, processor, tokenizer, output_mode, label_list, device, num_labels,
                     task_name, tr_loss, global_step=0, epoch=-1, explainer=explainer)
        else:
            explain(args, model, processor, tokenizer, output_mode, label_list, device)


def validate(args, model, processor, tokenizer, output_mode, label_list, device, num_labels,
             task_name, tr_loss, global_step, epoch, explainer=None):
    subgroups = None
    if hasattr(args, 'dev_advice'):
        eval_examples = processor.get_dev_examples_with_advices(args.data_dir, args.dev_advice)
        eval_features = convert_examples_with_advices_to_features(
                eval_examples, label_list, args.max_seq_length, tokenizer, output_mode, configs,
                cls_token_at_end=False,            # xlnet has a cls token at the end
                cls_token=tokenizer.cls_token,
                cls_token_segment_id=0,
                sep_token=tokenizer.sep_token,
                sep_token_extra=bool(args.base_model in ['roberta']), # roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
                pad_on_left=False,                 # pad on the left for xlnet
                pad_token=tokenizer.encoder[tokenizer.pad_token] if args.base_model in ['roberta'] else tokenizer.vocab[tokenizer.pad_token],
                pad_token_segment_id=0,)
    else:
        if task_name == 'iptt':
            eval_examples, subgroups = processor.get_test_examples(args.data_dir, neutral_words_file=args.neutral_words_file)
        else:
            eval_examples = processor.get_test_examples(args.data_dir)
        eval_features = convert_examples_to_features(eval_examples, label_list, args.max_seq_length, tokenizer, output_mode, configs,
            cls_token_at_end=False,            # xlnet has a cls token at the end
            cls_token=tokenizer.cls_token,
            cls_token_segment_id=0,
            sep_token=tokenizer.sep_token,
            sep_token_extra=bool(args.base_model in ['roberta']), # roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
            pad_on_left=False,                 # pad on the left for xlnet
            pad_token=tokenizer.encoder[tokenizer.pad_token] if args.base_model in ['roberta'] else tokenizer.vocab[tokenizer.pad_token],
            pad_token_segment_id=0,
        )
    
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_examples))
    logger.info("  Batch size = %d", args.eval_batch_size)
    all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)

    if output_mode == "classification":
        all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
    elif output_mode == "regression":
        all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.float)
    
    if args.reg_explanations and not args.test:
        all_importances = torch.tensor([f.importances for f in eval_features], dtype=torch.long)
        all_interactions = torch.tensor([f.interactions for f in eval_features], dtype=torch.long)

        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids,
                                        all_importances, all_interactions)
    else:
        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    # Run prediction for full data
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

    model.train(False)
    eval_loss, eval_loss_reg = 0, 0
    eval_reg_cnt = 0
    nb_eval_steps = 0
    preds = []

    # for detailed prediction results
    input_seqs = []

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        #if args.reg_explanations and not args.test:
        if args.reg_explanations:
            input_ids, input_mask, segment_ids, label_ids,\
                importances, interactions                 = batch
        else:
            input_ids, input_mask, segment_ids, label_ids = batch

        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        label_ids = label_ids.to(device)

        #if args.reg_explanations and not args.test:
        if args.reg_explanations:
            importances = importances.to(device)
            interactions = interactions.to(device)

        with torch.no_grad():
            inputs = {'input_ids': input_ids,
                      'attention_mask': input_mask,
                      'token_type_ids': segment_ids if args.base_model == 'bert' else None,  # XLM don't use segment_ids
                      'labels': None}
            logits = model(**inputs)

        # create eval loss and other metric required by the task
        if output_mode == "classification":
            loss_fct = CrossEntropyLoss()
            tmp_eval_loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1))
        elif output_mode == "regression":
            loss_fct = MSELoss()
            tmp_eval_loss = loss_fct(logits.view(-1), label_ids.view(-1))

        eval_loss += tmp_eval_loss.mean().item()

        if args.reg_explanations and args.algo == 'soc':
            with torch.no_grad():
                reg_loss, reg_cnt = explainer.compute_expl_loss_with_advice(input_ids, input_mask, segment_ids, label_ids,
                                                              importances, interactions, do_backprop=False)
            #eval_loss += reg_loss.item()
            eval_loss_reg += reg_loss
            eval_reg_cnt += reg_cnt

        nb_eval_steps += 1
        if len(preds) == 0:
            preds.append(logits.detach().cpu().numpy())
        else:
            preds[0] = np.append(preds[0], logits.detach().cpu().numpy(), axis=0)

        for b in range(input_ids.size(0)):
            i = 0
            while i < input_ids.size(1) and input_ids[b,i].item() != 0:
                i += 1
            token_list = tokenizer.convert_ids_to_tokens(input_ids[b,:i].cpu().numpy().tolist())
            input_seqs.append(' '.join(token_list))


    eval_loss = eval_loss / nb_eval_steps
    eval_loss_reg = eval_loss_reg / (eval_reg_cnt + 1e-10)
    preds = preds[0]
    
    if args.use_eer:
        eer_threshold = compute_eer_threshold(preds, all_label_ids.numpy())

    if output_mode == "classification":
        pred_labels = np.argmax(preds, axis=1)
        if args.use_eer:
            pred_labels = (preds[:, 1] - preds[:, 0]) > eer_threshold

    elif output_mode == "regression":
        pred_labels = np.squeeze(preds)
    
    pred_prob = F.softmax(torch.from_numpy(preds).float(), -1).numpy()
    result = compute_metrics(task_name, pred_labels, all_label_ids.numpy(), pred_prob, subgroups)
    loss = tr_loss / (global_step + 1e-10) if args.do_train else None

    result['eval_loss'] = eval_loss
    result['eval_loss_reg'] = eval_loss_reg
    result['global_step'] = global_step
    result['loss'] = loss

    split = 'dev' if not args.test else 'test'

    output_eval_file = os.path.join(args.output_dir, "eval_results_%d_%s_%s.txt"
                                    % (global_step, split, args.task_name))
    with open(output_eval_file, "w") as writer:
        logger.info("***** Eval results *****")
        logger.info("Epoch %d" % epoch)
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))

    output_detail_file = os.path.join(args.output_dir, "eval_details_%d_%s_%s.txt"
                                    % (global_step, split, args.task_name))
    with open(output_detail_file,'w') as writer:
        for i, seq in enumerate(input_seqs):
            pred = preds[i]
            gt = all_label_ids[i]
            if args.less_detail:
                pred_label = 0
                if pred[1] > pred[0]:
                    pred_label = 1
                writer.write('{}\t{}\n'.format(pred_label, seq))
            else:
                writer.write('{}\t{}\t{}\n'.format(gt, pred, seq))

    model.train(True)
    return result


def explain(args, model, processor, tokenizer, output_mode, label_list, device):
    """
    Added into run_model.py to support explanations
    :param args: configs, or args
    :param model: The model to be explained
    :param processor: For explanations on Gab/WS etc. Dataset, take an instance of Processor as input.
                    See Processor for details about the processor
    :param tokenizer: The default BERT tokenizer
    :param output_mode: "classification" for Gab
    :param label_list: "[0,1]" for Gab
    :param device: An instance of torch.device
    :return:
    """
    assert args.eval_batch_size == 1
    processor.set_tokenizer(tokenizer)

    if args.algo == 'soc':
        try:
            train_lm_dataloder = processor.get_dataloader('train', configs.train_batch_size)
            dev_lm_dataloader = processor.get_dataloader('dev', configs.train_batch_size)
        except FileNotFoundError:
            train_lm_dataloder = None
            dev_lm_dataloader = None

        explainer = SamplingAndOcclusionExplain(model, configs, tokenizer, device=device, vocab=tokenizer.vocab,
                                                train_dataloader=train_lm_dataloder,
                                                dev_dataloader=dev_lm_dataloader,
                                                lm_dir=args.lm_dir,
                                                output_path=os.path.join(configs.output_dir, configs.output_filename),
                                               )
    else:
        raise ValueError

    label_filter = None
    if args.only_positive and args.only_negative:
        label_filter = None
    elif args.only_positive: label_filter = 1
    elif args.only_negative: label_filter = 0

    if not args.test:
        eval_examples = processor.get_dev_examples(args.data_dir, label=label_filter)
    else:
        eval_examples = processor.get_test_examples(args.data_dir, label=label_filter)
    
    if args.reg_explanations:
        eval_features = convert_examples_with_advices_to_features(
                eval_examples, label_list, args.max_seq_length, tokenizer, output_mode, configs)
    else:    
        eval_features = convert_examples_to_features(
            eval_examples, label_list, args.max_seq_length, tokenizer, output_mode, configs)
    
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_examples))
    logger.info("  Batch size = %d", args.eval_batch_size)
    all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)

    if output_mode == "classification":
        all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
    elif output_mode == "regression":
        all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.float)

    eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    # Run prediction for full data
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

    if args.hiex_idxs:
        with open(args.hiex_idxs) as f:
            hiex_idxs = json.load(f)['idxs']
            print('Loaded line numbers for explanation')
    else:
        hiex_idxs = []

    model.train(False)
    for i, (input_ids, input_mask, segment_ids, label_ids) in tqdm(enumerate(eval_dataloader), desc="Evaluating"):
        if i == args.stop: break
        if hiex_idxs and i not in hiex_idxs: continue
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        label_ids = label_ids.to(device)

        if not args.hiex:
            explainer.word_level_explanation_bert(input_ids, input_mask, segment_ids, label_ids)
        else:
            explainer.hierarchical_explanation_bert(input_ids, input_mask, segment_ids, label_ids)
    if hasattr(explainer, 'dump'):
        explainer.dump()

def save_model(args, model, tokenizer, num_labels):
    # Save a trained model, configuration and tokenizer
    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self

    # If we save using the predefined names, we can load using `from_pretrained`
    output_model_file = os.path.join(args.output_dir, WEIGHTS_NAME)
    output_config_file = os.path.join(args.output_dir, CONFIG_NAME)

    torch.save(model_to_save.state_dict(), output_model_file)

    if args.base_model == 'lstm':
        return
    
    model_to_save.config.to_json_file(output_config_file)
    tokenizer.save_vocabulary(args.output_dir)

if __name__ == "__main__":
    main()