from .soc_algo import _SamplingAndOcclusionAlgo
from .lm import BiGRULanguageModel
from .train_lm import do_train_lm
import os, logging, torch, pickle
import json
import itertools

logger = logging.getLogger(__name__)

class SamplingAndOcclusionExplain:
    def __init__(self, model, configs, tokenizer, output_path, device, lm_dir=None, train_dataloader=None,
                 dev_dataloader=None, vocab=None):
        self.configs = configs
        self.model = model
        self.lm_dir = lm_dir
        self.train_dataloader = train_dataloader
        self.dev_dataloader = dev_dataloader
        self.vocab = vocab
        self.output_path = output_path
        self.device = device
        self.hiex = configs.hiex
        self.tokenizer = tokenizer

        self.lm_model = self.detect_and_load_lm_model()

        self.algo = _SamplingAndOcclusionAlgo(model, tokenizer, self.lm_model, output_path, configs)

        self.use_padding_variant = configs.use_padding_variant
        try:
            self.output_file = open(self.output_path, 'w' if not configs.hiex else 'wb')
        except FileNotFoundError:
            self.output_file = None
        self.output_buffer = []

        # for explanation regularization
        self.neutral_words_file = configs.neutral_words_file
        self.neutral_words_ids = None
        self.neutral_words = None
        #self.debug = debug

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
        pad_tok = self.tokenizer.encoder[self.tokenizer.pad_token] if self.configs.base_model in ['roberta'] else self.tokenizer.vocab[self.tokenizer.pad_token]
        model = BiGRULanguageModel(self.configs, vocab=self.vocab, pad_token=pad_tok, device=self.device).to(self.device)
        do_train_lm(model, lm_dir=self.lm_dir, lm_epochs=20,
                    train_iter=self.train_dataloader, dev_iter=self.dev_dataloader)

    def word_level_explanation_bert(self, input_ids, input_mask, segment_ids, label=None):
        # requires batch size is 1
        # get sequence length
        i = 0
        while i < input_ids.size(1) and input_ids[0,i] != 0: # pad
            i += 1
        inp_length = i
        # do not explain [CLS] and [SEP]
        spans, scores = [], []
        for i in range(1, inp_length-1, 1):
            span = (i, i)
            spans.append(span)
            if not self.use_padding_variant:
                score = self.algo.do_attribution(input_ids, input_mask, segment_ids, span, label)
            else:
                score = self.algo.do_attribution_pad_variant(input_ids, input_mask, segment_ids, span, label)
            scores.append(score)
        inp = input_ids.view(-1).cpu().numpy()
        s = self.algo.repr_result_region(inp, spans, scores)
        self.output_file.write(s + '\n')

    def hierarchical_explanation_bert(self, input_ids, input_mask, segment_ids, label=None):
        tab_info = self.algo.do_hierarchical_explanation(input_ids, input_mask, segment_ids, label)
        self.output_buffer.append(tab_info)
        # currently store a pkl after explaining each instance
        self.output_file = open(self.output_path, 'w' if not self.hiex else 'wb')
        pickle.dump(self.output_buffer, self.output_file)
        self.output_file.close()

    def _initialize_neutral_words(self):
        f = open(self.neutral_words_file)
        neutral_words = []
        neutral_words_ids = set()
        for line in f.readlines():
            word = line.strip().split('\t')[0]
            canonical = self.tokenizer.tokenize(word)
            if len(canonical) > 1:
                canonical.sort(key=lambda x: -len(x))
                print(canonical)
            word = canonical[0]
            neutral_words.append(word)
            neutral_words_ids.add(self.tokenizer.vocab[word])
        self.neutral_words = neutral_words
        self.neutral_words_ids = neutral_words_ids
        assert neutral_words

    def compute_explanation_loss(self, input_ids_batch, input_mask_batch, segment_ids_batch, label_ids_batch,
                                 do_backprop=False):
        if self.neutral_words is None:
            self._initialize_neutral_words()
        batch_size = input_ids_batch.size(0)
        neutral_word_scores, cnt = [], 0
        for b in range(batch_size):
            input_ids, input_mask, segment_ids, label_ids = input_ids_batch[b], \
                                                            input_mask_batch[b], \
                                                            segment_ids_batch[b], \
                                                            label_ids_batch[b]
            nw_positions = []
            for i in range(len(input_ids)):
                word_id = input_ids[i].item()
                if word_id in self.neutral_words_ids:
                    nw_positions.append(i)
            # only generate explanations for neutral words
            for i in range(len(input_ids)):
                word_id = input_ids[i].item()
                if word_id in self.neutral_words_ids:
                    x_region = (i, i)
                    #score = self.algo.occlude_input_with_masks_and_run(input_ids, input_mask, segment_ids,
                    #                                                   [x_region], nb_region, label_ids,
                    #                                                    return_variable=True)
                    if not self.configs.use_padding_variant:
                        score = self.algo.do_attribution(input_ids, input_mask, segment_ids, x_region, label_ids,
                                                         return_variable=True, additional_mask=nw_positions)
                    else:
                        score = self.algo.do_attribution_pad_variant(input_ids, input_mask, segment_ids,
                                                                     x_region, label_ids, return_variable=True,
                                                                     additional_mask=nw_positions)
                    score = self.configs.reg_strength * (score ** 2)

                    if do_backprop:
                        score.backward()

                    neutral_word_scores.append(score.item())

        if neutral_word_scores:
            return sum(neutral_word_scores), len(neutral_word_scores)
        else:
            return 0., 0

class SOCWithInteraction(SamplingAndOcclusionExplain):
    def __init__(self, model, configs, tokenizer, output_path, device, lm_dir=None, train_dataloader=None,
                 dev_dataloader=None, vocab=None):
        self.configs = configs
        self.model = model
        self.lm_dir = lm_dir
        self.train_dataloader = train_dataloader
        self.dev_dataloader = dev_dataloader
        self.vocab = vocab
        self.output_path = output_path
        self.device = device
        self.hiex = configs.hiex
        self.tokenizer = tokenizer

        self.lm_model = self.detect_and_load_lm_model()

        self.algo = _SamplingAndOcclusionAlgo(model, tokenizer, self.lm_model, output_path, configs)

        self.use_padding_variant = configs.use_padding_variant
        try:
            self.output_file = open(self.output_path, 'w' if not configs.hiex else 'wb')
        except FileNotFoundError:
            self.output_file = None
        self.output_buffer = []

        self.inter_strength = self.configs.reg_strength
        if hasattr(configs, 'reg_interaction_strength'):
            self.inter_strength = self.configs.reg_interaction_strength
        
    def compute_expl_loss_with_advice(self, input_ids_batch, input_mask_batch, segment_ids_batch, label_ids_batch,
                                 importances_batch, interactions_batch, confidences_batch=None,
                                 do_backprop=False):
        
        importance_scores = []
        interaction_scores = []
        
        batch_size = input_ids_batch.size(0)
        for b in range(batch_size):
            input_ids, input_mask, segment_ids, label_ids,\
                importances, interactions                 = input_ids_batch[b], \
                                                            input_mask_batch[b], \
                                                            segment_ids_batch[b], \
                                                            label_ids_batch[b], \
                                                            importances_batch[b], \
                                                            interactions_batch[b]
            
            if self.configs.confidence:
                confidences = confidences_batch[b]
            else:
                confidences = 1
            
            length = len(input_ids)

            """
            Regularize attribution scores
            """
            if not self.configs.only_interaction:
                for label in range(importances.shape[0]):
                    start = -1

                    for i in range(length):
                        if importances[label, i] == 0:
                            start = -1
                            continue

                        if start == -1:
                            start = i
                        
                        if i == length - 1 or importances[label, i+1] != importances[label, start]:
                            x_region = (start, i)

                            # compute attribution score using input occlusion
                            scores = self.algo.do_attribution(input_ids, input_mask, segment_ids, x_region, label_ids,
                                                            return_variable=True, have_target=True)
                            
                            if importances[label, start] > 0:
                                target_score = 1
                            else:
                                assert importances[label, start] < 0, 'Importance extraction error'
                                target_score = 0
                            
                            score = self.configs.reg_strength * confidences * ((scores[0, label] - target_score) ** 2)
                            if do_backprop:
                                score.backward()
                            
                            importance_scores.append(score.item())
                        
                            start = i + 1
            
            """
            Regularize interaction scores
            """
            if self.configs.reg_interaction or self.configs.only_interaction:
                for label in range(interactions.shape[0]):
                    i = 0
                    while i < length and interactions[label, i] != 0:
                        direction, st1, ed1, st2, ed2 = interactions[label, i:i+5]

                        pair = (min(st1, st2), max(ed1, ed2))
                        region_i = (st1, ed1)
                        region_j = (st2, ed2)

                        mask_for_i = [k for k in range(st2, ed2+1)]
                        mask_for_j = [k for k in range(st1, ed1+1)]
                        i += 5
                    
                        if not self.configs.use_padding_variant:
                            score_i_j = self.algo.do_attribution(input_ids, input_mask, segment_ids, pair, label_ids,
                                                                return_variable=True, occlude_pair=True, have_target=True)

                            score_i = self.algo.do_attribution(input_ids, input_mask, segment_ids, region_i, label_ids,
                                                                return_variable=True, additional_mask=mask_for_i, have_target=True)
                            
                            score_j = self.algo.do_attribution(input_ids, input_mask, segment_ids, region_j, label_ids,
                                                                return_variable=True, additional_mask=mask_for_j, have_target=True)
                        else:
                            raise NotImplementedError
                        
                        scores = score_i_j - score_i - score_j
                        
                        score = 0
                        target_score = ((direction + 1)/2).float()
                        score += self.inter_strength * confidences * ((scores[0, label] - target_score) ** 2)

                        if do_backprop:
                            score.backward()

                        interaction_scores.append(score.item())

        if importance_scores and interaction_scores:
            return sum(importance_scores) + sum(interaction_scores), \
                    len(importance_scores) + len(interaction_scores)
        else:
            return 0., 0
