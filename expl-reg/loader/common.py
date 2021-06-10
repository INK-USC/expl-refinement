import json, os, csv, sys, logging, tqdm
import itertools
import torch
from torch.utils.data.dataloader import default_collate

logger = logging.getLogger(__name__)

class DotDict:
    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self,k,v)

def dotdict_collate(batch):
    return DotDict(**default_collate(batch))


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid # unused
        self.text_a = text_a
        self.text_b = text_b
        self.label = label

class InputExampleWithAdvice(object):
    """A single training/test example for simple sequence classification with regularization advice provided."""

    def __init__(self, guid, text_a, text_b=None, label=None, advice=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
            advice: Regularization advice list provided by human.
        """
        self.guid = guid # unused
        self.text_a = text_a
        self.text_b = text_b
        self.label = label
        self.advice = advice
        self.confidence = None

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id

class InputFeaturesWithAdvice(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id, importances, interactions, confidence):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.importances = importances
        self.interactions = interactions
        self.confidence = confidence

class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
            return lines

def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def convert_examples_to_features(examples, label_list, max_seq_length,
                                 tokenizer, output_mode, configs,
                                 cls_token_at_end=False,
                                 cls_token='[CLS]',
                                 cls_token_segment_id=1,
                                 sep_token='[SEP]',
                                 sep_token_extra=False,
                                 pad_on_left=False,
                                 pad_token=0,
                                 pad_token_segment_id=0,
                                 sequence_a_segment_id=0, 
                                 sequence_b_segment_id=1,
                                 mask_padding_with_zero=True):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = tokens_a + [sep_token]
        # if sep_token_extra:
        #     # roberta uses an extra separator b/w pairs of sentences
        #     tokens += [sep_token]
        segment_ids = [sequence_a_segment_id] * len(tokens)

        if tokens_b:
            tokens += tokens_b + [sep_token]
            segment_ids += [sequence_b_segment_id] * (len(tokens_b) + 1)

        if cls_token_at_end:
            tokens = tokens + [cls_token]
            segment_ids = segment_ids + [cls_token_segment_id]
        else:
            tokens = [cls_token] + tokens
            segment_ids = [cls_token_segment_id] + segment_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        segment_ids = [0] * len(input_ids)
        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        if output_mode == "classification":
            label_id = label_map[example.label]
        elif output_mode == "regression":
            label_id = float(example.label)
        else:
            raise KeyError(output_mode)

        # if ex_index < 5:
        #     logger.info("*** Example ***")
        #     logger.info("guid: %s" % (example.guid))
        #     logger.info("tokens: %s" % " ".join(
        #         [str(x) for x in tokens]))
        #     logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        #     logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        #     logger.info(
        #         "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        #     logger.info("label: %s (id = %d)" % (example.label, label_id))

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_id=label_id))
    return features

def write_intermediates(intermediates, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        for ele in intermediates:
            sentence, advice, i0, i1 = ele
            f.write(sentence + '\t' + str(advice) + '\n')
            f.write(str(i0) + '\n')
            f.write(str(i1) + '\n')


def convert_examples_with_advices_to_features(examples, label_list, max_seq_length,
                                 tokenizer, output_mode, configs, debug=False,
                                 cls_token_at_end=False,
                                 cls_token='[CLS]',
                                 cls_token_segment_id=1,
                                 sep_token='[SEP]',
                                 sep_token_extra=False,
                                 pad_on_left=False,
                                 pad_token=0,
                                 pad_token_segment_id=0,
                                 sequence_a_segment_id=0, 
                                 sequence_b_segment_id=1,
                                 mask_padding_with_zero=True):    
    """
    Loads a data file into a list of `InputBatch`s.
    Each example is provided with human advice.
    """
    def get_span_ids(span):
        st, ed = span
        res = []
        for i in range(st, ed+1):
            if i >= max_seq_length-1:
                break
            res.append(i + 1)
        
        return res

    def get_importance(spans, sign, importance, debug=False):
        count = 1
        for span in spans:
            hf_bert_ids = get_span_ids(span)

            hf_bert_tokens = []
            for idx in hf_bert_ids:
                importance[idx] = sign * count
                hf_bert_tokens.append(tokens[idx])
            
            if debug:
                print(' '.join(hf_bert_tokens))
            
            count += 1
        
        if debug and len(spans) > 1:
            logger.info('Span count %d' % (len(spans)))
            print(example.text_a)
            print(importance)

    def get_interaction(pair_inc, pair_dec, interaction):
        count = 0
        for span_pair in pair_inc:
            if count+5 >= max_seq_length:
                return
            span_a, span_b = span_pair
            st1, ed1 = span_a
            st2, ed2 = span_b

            if st1 + 2 >= max_seq_length or st2 + 2 >= max_seq_length:
                continue

            interaction[count: count+5] = 1, st1+1, min(ed1+1, max_seq_length-1), st2+1, min(ed2+1, max_seq_length-1)
            count += 5
        
        for span_pair in pair_dec:
            if count+5 >= max_seq_length:
                return
            span_a, span_b = span_pair
            st1, ed1 = span_a
            st2, ed2 = span_b
            interaction[count: count+5] = -1, st1+1, ed1+1, st2+1, ed2+1
            count += 5
    
    def get_advice_features(single_inc, single_dec, pair_inc, pair_dec):
        importance = [0] * max_seq_length
        interaction = [0] * max_seq_length  # format: [+-1, st1, ed1, st2, ed2, +-1, st1, ed1, st2, ed2, 0, ..., 0]

        get_importance(single_inc, 1, importance)
        get_importance(single_dec, -1, importance)
        
        get_interaction(pair_inc, pair_dec, interaction)

        return importance, interaction

    label_map = {label: i for i, label in enumerate(label_list)}

    features = []

    intermediates = []

    for (ex_index, example) in enumerate(examples):
        # if ex_index % 10000 == 0:
        #     logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        tokens_a = tokenizer.tokenize(example.text_a)

        single_inc_0, single_dec_0, pair_inc_0, pair_dec_0, \
            single_inc_1, single_dec_1, pair_inc_1, pair_dec_1 = example.advice
        
        tokens_b = None
        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[:(max_seq_length - 2)]

        tokens = tokens_a + [sep_token]
        # if sep_token_extra:
        #     # roberta uses an extra separator b/w pairs of sentences
        #     tokens += [sep_token]
        segment_ids = [sequence_a_segment_id] * len(tokens)

        if tokens_b:
            tokens += tokens_b + [sep_token]
            segment_ids += [sequence_b_segment_id] * (len(tokens_b) + 1)

        if cls_token_at_end:
            tokens = tokens + [cls_token]
            segment_ids = segment_ids + [cls_token_segment_id]
        else:
            tokens = [cls_token] + tokens
            segment_ids = [cls_token_segment_id] + segment_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        importances_0, interactions_0 = get_advice_features(single_inc_0, single_dec_0, pair_inc_0, pair_dec_0)
        importances_1, interactions_1 = get_advice_features(single_inc_1, single_dec_1, pair_inc_1, pair_dec_1)

        segment_ids = [0] * len(input_ids)
        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        if output_mode == "classification":
            label_id = label_map[example.label]
        elif output_mode == "regression":
            label_id = float(example.label)
        else:
            raise KeyError(output_mode)

        # if ex_index < 5:
        #     logger.info("*** Example ***")
        #     logger.info("guid: %s" % (example.guid))
        #     logger.info("tokens: %s" % " ".join(
        #         [str(x) for x in tokens]))
        #     logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        #     logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        #     logger.info(
        #         "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        #     logger.info("label: %s (id = %d)" % (example.label, label_id))

        features.append(
            InputFeaturesWithAdvice(input_ids=input_ids,
                                    input_mask=input_mask,
                                    segment_ids=segment_ids,
                                    label_id=label_id,
                                    importances=[importances_0, importances_1],
                                    interactions=[interactions_0, interactions_1],
                                    confidence=example.confidence))
        
        if debug:
            intermediates.append((example.text_a, example.advice, importances_0, importances_1))
    
    if debug:
        write_intermediates(intermediates, 'loader/decoupled')
    
    return features
