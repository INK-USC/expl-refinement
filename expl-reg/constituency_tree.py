"""
Generate ground truth constituency parsing explanations
"""

import argparse
import csv
import json
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import torch

import spacy

from allennlp.predictors.predictor import Predictor
from bert.modeling import BertForSequenceClassification
from bert.tokenization import BertTokenizer
from hiex.soc_algo import _SamplingAndOcclusionAlgo
from loader import GabProcessor
from tqdm import tqdm
from utils.config import configs, combine_args

logger = logging.getLogger(__name__)
nlp = spacy.load("en_trf_bertbaseuncased_lg")

import matplotlib
matplotlib.use('Agg')

class ConstituencyExplainer:
    def __init__(self, model, configs, tokenizer, output_path, fig_dir, lm_dir=None):
        self.model = model
        self.configs = configs
        self.tokenizer = tokenizer
        self.output_path = output_path
        self.fig_dir = fig_dir
        self.lm_dir = lm_dir

        self.lm_model = self.detect_and_load_lm_model()
        self.algo = _SamplingAndOcclusionAlgo(model, tokenizer, self.lm_model, output_path, configs)

        self.constituency_parser = Predictor.from_path(
            "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo-constituency-parser-2018.03.14.tar.gz")

        self.tabs = []

    def detect_and_load_lm_model(self):
        if not self.lm_dir:
            self.lm_dir = 'runs/lm/'
        if not os.path.isdir(self.lm_dir):
            os.mkdir(self.lm_dir)

        file_name = None
        for x in os.listdir(self.lm_dir):
            if x.startswith('best'):
                file_name = x
                break
        if not file_name:
            self.train_lm()
            for x in os.listdir(self.lm_dir):
                if x.startswith('best'):
                    file_name = x
                    break
        lm_model = torch.load(open(os.path.join(self.lm_dir,file_name), 'rb'))
        return lm_model
    
    def train_lm(self):
        logger.info('Missing pretrained LM. Now training')
        raise NotImplementedError
    
    def traverse_tree(self, tree, tokens, st, ed, input_ids, input_mask, segment_ids, label, mapping, depth=0):
        def find_idx(word):
            for i in range(st, ed):
                for j in range(i, ed):
                    if ' '.join(tokens[i:j+1]) == word:
                        return i, j
            return -1, -1
        
        st1, ed1 = find_idx(tree['word'])
        bert_st, bert_ed = mapping[st1][0], mapping[ed1][1]

        score = self.algo.do_attribution(input_ids, input_mask, segment_ids, (bert_st, bert_ed), label)

        while len(self.tabs) <= depth:
            self.tabs.append([])
        self.tabs[depth].append((bert_st, bert_ed, score))

        if 'children' in tree:
            for child in tree['children']:
                self.traverse_tree(child, tokens, st1, ed1+1, input_ids, input_mask, segment_ids, label, mapping, depth+1)
    
    def get_tokenize_mapping(self, sentence):
        doc = nlp(sentence)
        toks = []
        spacy_alignment = doc._.trf_alignment

        """
        0   1  2 3   4  5 
        aaa bbb. ccc ddd.

        [[1,2], [3,3], [4,4], [7]]

        0  1  2 3  4 5  6  7   8  9 10 11
        [] aa a bbb. [] [] ccc dd d .  []

        0  1  2 3  4 5   6  7 8 9
        [] aa a bbb. ccc dd d . []
        """
        sentence_seps = []
        for i, tok in enumerate(doc._.trf_word_pieces_):
            if tok != '[CLS]' and tok != '[SEP]':
                toks.append(tok)
            elif tok == '[SEP]':
                sentence_seps.append(i)
        
        mapping = []
        j = 0
        for i, ele in enumerate(spacy_alignment):
            if j == len(sentence_seps):
                break
            if len(ele) == 0:
                continue

            while j < len(sentence_seps) and sentence_seps[j] < ele[0]:
                j += 1
            
            if j:
                for k in range(len(ele)):
                    spacy_alignment[i][k] -= 2*j
            
            mapping.append([spacy_alignment[i][0], spacy_alignment[i][-1]])
        
        # print([tok for tok in doc])
        # print(toks)
        # print(mapping)
            
        return toks, mapping

    def get_features(self, sentence):
        max_seq_length = self.configs.max_seq_length
        tokens_a, mapping = self.get_tokenize_mapping(sentence)

        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[:(max_seq_length - 2)]
        
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]

        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        segment_ids = [0] * len(input_ids)
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        input_mask = torch.tensor(input_mask, dtype=torch.long)
        segment_ids = torch.tensor(segment_ids, dtype=torch.long)
        
        return input_ids, input_mask, segment_ids, mapping, tokens

    def plot_tabs(self, label, tokens, sentence_id):
        width = self.tabs[0][0][1]
        height = len(self.tabs)

        score_array = np.zeros((height, width))
        
        for i in range(height):
            for span in self.tabs[i]:
                st, ed, score = span
                score_array[i, st-1:ed] = score
        
        # print(score_array)
        fig, ax = plt.subplots(figsize=(width, height))
        vmin, vmax = -5.0, 5.0
        im = ax.imshow(score_array, cmap='coolwarm', aspect=0.5, vmin=vmin, vmax=vmax)
        
        # place text on plot
        for i in range(height):
            for j in range(width):
                if score_array[i, j] != 0:
                    fontsize = 12
                    if len(tokens[j+1]) >= 8:
                        fontsize = 8
                    if len(tokens[j+1]) >= 12:
                        fontsize = 6
                    ax.text(j, i, tokens[j+1], ha='center', va='center',
                           fontsize=fontsize)
        
        plt.title(str(label), fontsize=14)
        dir_name = self.fig_dir
        if not os.path.isdir(dir_name): os.mkdir(dir_name)
        plt.savefig(dir_name + '/fig_{}.png'.format(sentence_id), bbox_inches='tight')
        plt.close()

    def explain(self, sentence, label, sentence_id):
        self.tabs = []
        
        try:
            input_ids, input_mask, segment_ids, mapping, tokens = self.get_features(sentence)

            # constituency parse
            constituency = self.constituency_parser.predict(sentence=sentence)
            original_tokens = constituency['tokens']
            # print(original_tokens)
            # print(len(original_tokens))
            self.traverse_tree(constituency['hierplane_tree']['root'], original_tokens, 0, len(original_tokens),
                                                    input_ids, input_mask, segment_ids, label, mapping)
            
            # print(self.tabs)
            self.plot_tabs(label, tokens, sentence_id)
        except Exception as e:
            logger.info('Met error for sentence id {}'.format(sentence_id))
            print(e)
            
    def explain_data_json(self, file_path):
        with open(file_path) as f:
            for i, line in enumerate(tqdm(f.readlines())):
                data = json.loads(line)
                # sentence = data["Text"]
                # if len(sentence) == 0:
                #     continue
                # label = data["hd"] or data["cv"]
                # sentence_id = data["text_id"]

                sentence = data['text']
                if len(sentence) == 0:
                    continue

                sentence_id = i
                label = data['label']

                self.explain(sentence, label, sentence_id)
    
    def explain_data(self, file_path):
        f = open(file_path)
        reader = csv.reader(f, delimiter='\t')
        next(reader)  # skip header
        for i, row in enumerate(tqdm(reader)):
            sentence = row[0]
            label = int(row[1])
            sentence_id = i
            self.explain(sentence, label, sentence_id)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--base_model", choices=['bert'], default='bert')
    parser.add_argument("--bert_model", default=None, type=str, required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                             "bert-base-multilingual-cased, bert-base-chinese.")
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
    parser.add_argument("--neutral_words_file", default='data/identity.csv')
    parser.add_argument("--hateful_words_file", default = 'data/hateword.txt')
    parser.add_argument("--sentiment_words_file", default = 'data/sentiment.csv')

    # if true, use test data instead of val data
    parser.add_argument("--test", action='store_true')

    # Explanation specific arguments below

    # whether run explanation algorithms
    parser.add_argument("--explain", action='store_true', help='if true, explain test set predictions')
    parser.add_argument("--debug", action='store_true')

    # which algorithm to run
    parser.add_argument("--algo", choices=['soc'])

    # the output filename without postfix
    parser.add_argument("--output_filename", default='temp.tmp')

    parser.add_argument("--input_file", default="train_small.jsonl")
    parser.add_argument("--fig_dir", default="case_study/figs/")

    # see utils/config.py
    parser.add_argument("--use_padding_variant", action='store_true')
    parser.add_argument("--mask_outside_nb", action='store_true')
    parser.add_argument("--nb_range", type=int)
    parser.add_argument("--sample_n", type=int)
    parser.add_argument("--reg_interaction", action='store_true')

    # whether use explanation regularization
    parser.add_argument("--reg_explanations", action='store_true')
    parser.add_argument("--reg_strength", type=float)
    parser.add_argument("--reg_mse", action='store_true')

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
    args = parser.parse_args()
    combine_args(configs, args)
    args = configs

    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)
    processor = GabProcessor(configs, tokenizer=tokenizer)

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

    label_list = processor.get_labels()
    num_labels = len(label_list)

    model = BertForSequenceClassification.from_pretrained(args.bert_model, num_labels=num_labels)
    model.to(device)
    model.train(False)

    logging.getLogger('allennlp.common.params').disabled = True
    logging.getLogger('allennlp.common.from_params').disabled = True
    logging.getLogger('allennlp.nn.initializers').disabled = True 
    # logging.getLogger('allennlp.modules.token_embedders.embedding').setLevel(logging.INFO) 
    # logging.getLogger('urllib3.connectionpool').disabled = True 
    
    explainer = ConstituencyExplainer(model, configs, tokenizer, output_path=os.path.join(configs.output_dir,
                                                                                         configs.output_filename),
                                                                fig_dir=configs.fig_dir)
    
    if args.input_file[-5:] == 'jsonl':
        explainer.explain_data_json(os.path.join(args.data_dir, args.input_file))
    else:
        explainer.explain_data(os.path.join(args.data_dir, args.input_file))