from .common import *
from torch.utils.data import DataLoader, Dataset

import ast
import torch
import random
import string
import csv

class SstProcessor(DataProcessor):
    """
    Data processor using DataProcessor class provided by BERT
    """
    def __init__(self, configs, tokenizer=None):
        super().__init__()
        self.data_dir = configs.data_dir
        #self.label_groups = configs.label_groups
        self.tokenizer = tokenizer
        self.max_seq_length = configs.max_seq_length
        self.configs = configs

    def _create_examples(self, data_dir, split, label=None):
        """
        Create a list of InputExample, where .text_a is raw text and .label is specified
        as configs.label_groups
        :param data_dir:
        :param split:
        :param label:
        :return:
        """

        f = open(os.path.join(data_dir, '%s.tsv' % split))
        reader = csv.reader(f, delimiter='\t')
        next(reader)  # skip header
        examples = []
        for i, row in enumerate(reader):
            example = InputExample(text_a=row[0], guid='%s-%s' % (split, i))
            setattr(example, 'label', int(row[1]))
            if label is None or example.label == label:
                examples.append(example)

        return examples

    def _create_examples_with_advices(self, data_dir, split, advice_file, label=None):
        """
        Create a list of InputExampleWithAdvice, where .text_a is raw text and .label is specified
        as configs.label_groups; .advice is read from text file
        :param data_dir:
        :param split:
        :advice_file
        :param label:
        :return:
        """

        examples = []
        printable = set(string.printable)

        f = open(advice_file, encoding='utf-8')
        advice_lines = f.readlines()
        f.close()

        if len(advice_lines) == 0:
            return examples

        split_count = len(advice_lines[0].split('\t'))

        has_confidence = False

        if split_count >= 3:
            if split_count == 4:
                has_confidence = True
            has_label = True
            print('loading advice file %s, labels provided' % (advice_file))
        elif split_count == 2:
            has_label = False
            print('loading advice file %s, labels not provided' % (advice_file))
        else:
            raise NotImplementedError

        for i, line in enumerate(advice_lines):
            if has_confidence:
                sentence, advice, instance_label, confidence = line.split('\t') 
            elif has_label:
                sentence, advice, instance_label = line.split('\t')
            else:
                sentence, advice = line.split('\t')
                instance_label = 0

            example = InputExampleWithAdvice(text_a=sentence, guid='%s-%s' % (split, i))

            setattr(example, 'label', int(instance_label))

            advice = ''.join(filter(lambda x: x in printable, advice))
            advice_literal = ast.literal_eval(advice)
            setattr(example, 'advice', advice_literal)

            if has_confidence:
                confidence = float(confidence)
                setattr(example, 'confidence', confidence)

            if label is None or example.label == label:
                examples.append(example)

        return examples

    def get_train_examples(self, data_dir, label=None):
        return self._create_examples(data_dir, 'train', label)

    def get_dev_examples(self, data_dir, label=None):
        return self._create_examples(data_dir, 'dev', label)

    def get_test_examples(self, data_dir, label=None):
        return self._create_examples(data_dir, 'test', label)

    def get_train_examples_with_advices(self, data_dir, advice_file, label=None):
        return self._create_examples_with_advices(data_dir, 'train', advice_file, label)

    def get_dev_examples_with_advices(self, data_dir, advice_file, label=None):
        return self._create_examples_with_advices(data_dir, 'dev', advice_file, label)

    def get_example_from_tensor_dict(self, tensor_dict):
        raise NotImplementedError

    def get_labels(self):
        return [0,1]

    def get_features(self, split):
        """
        Return a list of dict, where each dict contains features to be fed into the BERT model
        for each instance. ['text'] is a LongTensor of length configs.max_seq_length, either truncated
        or padded with 0 to match this length.
        :param split: 'train' or 'dev'
        :return:
        """

        examples = self._create_examples(self.data_dir, split)
        features = []
        for example in examples:
            tokens = self.tokenizer.tokenize(example.text_a)
            if len(tokens) > self.max_seq_length - 2:
                tokens = tokens[:(self.max_seq_length - 2)]
            tokens = [ self.tokenizer.cls_token ] + tokens + [ self.tokenizer.sep_token ]
            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

            length = len(input_ids)
            padding = [0] * (self.max_seq_length - length)
            input_ids += padding
            input_ids = torch.LongTensor(input_ids)

            input_mask = torch.LongTensor([1] * length + [0] * len(padding))
            segment_id = torch.zeros_like(input_mask)
            
            features.append({'text': input_ids, 'length': length,
                             'input_mask': input_mask, 'segment_id': segment_id
                             })
        return features

    def get_dataloader(self, split, batch_size=1):
        """
        return a torch.utils.DataLoader instance, mainly used for training the language model.
        :param split:
        :param batch_size:
        :return:
        """
        features = self.get_features(split)
        dataset = SstDataset(features)
        dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=dotdict_collate)
        return dataloader

    def set_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer


class SstDataset(Dataset):
    """
    torch.utils.Dataset instance for building torch.utils.DataLoader, for training the language model.
    """
    def __init__(self, features):
        super().__init__()
        self.features = features

    def __getitem__(self, item):
        return self.features[item]

    def __len__(self):
        return len(self.features)