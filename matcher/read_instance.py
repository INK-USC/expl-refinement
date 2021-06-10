import json
import time
import logging
import sys
import matplotlib as plt
import pickle
from tqdm import tqdm
import networkx as nx
from allennlp.predictors.predictor import Predictor
from allennlp.data.tokenizers.sentence_splitter import SpacySentenceSplitter
from allennlp.data.tokenizers import WordTokenizer
from allennlp.data.dataset_readers.reading_comprehension.util import char_span_to_token_span

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name
HEADER = ['< Table >', '< Tr >', '< Ul >', '< Ol >', '< Dl >', '< Li >', '< Dd >', '< Dt >', '< Th >', '< Td >']

HATEWORD_FILE = 'data/hateword.txt'
SENTIMENT_FILE = 'data/sentiment.csv'
IDENTITY_FILE = 'data/identity.csv'
NEGATION_FILE = 'data/negation.txt'

LEXICON_TYPE = ['sentiment', 'identity', 'negation', 'hateful']

class Instance():
    def __init__(self,
                 context_tokens,
                 context_offsets=None,
                 context_info=None,
                 original_context=None,
                 label=None):
        self.context_tokens = context_tokens
        self.context_offsets = context_offsets
        self.context_info = context_info
        self.original_context = original_context
        self.label = label

class InstancePreprocess:
    def __init__(self, pre=True) -> None:
        print('Instance preprocessor is initialized.')
        self._tokenizer = WordTokenizer()
        self._sentence_splitter = SpacySentenceSplitter()
        if pre:
            self._constituency_parser = Predictor.from_path(
                "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo-constituency-parser-2018.03.14.tar.gz")
            self._ner_tagger = Predictor.from_path(
                "https://allennlp.s3.amazonaws.com/models/fine-grained-ner-model-elmo-2018.12.21.tar.gz")
            self._dependency_parser = Predictor.from_path(
                "https://s3-us-west-2.amazonaws.com/allennlp/models/biaffine-dependency-parser-ptb-2018.08.23.tar.gz")
            self.hatewords_lexicon = self.get_hatewords()
            self.sentiment_lexicon = self.get_sentiment()
            self.identity_lexicon = self.get_identities()
            self.negation_lexicon = self.get_negations()

    def read_preprocess(self, file_path):
        all_instances = []
        with open(file_path, 'r') as f:
            idx = 0
            start_time = time.time()
            for i, line in enumerate(f.readlines()):
                idx += 1
                data = json.loads(line)
                sentence = data["text"]
                label = data["label"]
                if idx %100 == 0:
                    sys.stdout.write("Done {}, Time: {} sec\n".format(idx,time.time() - start_time))
                    sys.stdout.flush()
                try:
                    instance = self.read_one(sentence, label)
                    all_instances.append(instance)
                except AssertionError:
                    print(sentence)

        return all_instances

    def read_one(self, sentence, label):
        """read one instance and transform to our form"""

        max_seq_length=128

        tokens = self._tokenizer.tokenize(sentence)
        #tokens = tokens[:(max_seq_length - 2)]
        #sentence = ' '.join([token.text for token in tokens])
        constituency = self._constituency_parser.predict(sentence=sentence)
        ners = self._ner_tagger.predict(sentence=sentence)
        dependency = self._dependency_parser.predict(sentence=sentence)
        passage_info = self.get_info(tokens, constituency, ners, dependency, 0)

        passage_tokens = self._tokenizer.tokenize(sentence)
        passage_offsets = [(token.idx, token.idx + len(token.text)) for token in passage_tokens]

        return Instance(context_tokens=passage_tokens,
                        context_offsets=passage_offsets,
                        context_info=passage_info,
                        original_context=sentence,
                        label=label
                        )

    def get_info(self, _tokens, constituency, ner, dependency, offset):
        tokens = [token.text for token in _tokens]
        lemmas = [token.lemma_ for token in _tokens]

        lexicon_output = self.get_lexicon(_tokens)
        constituency_output = self.traverse_tree(constituency['hierplane_tree']['root'], tokens, 0, len(tokens))
        dependency_output = self.get_dependency_output(dependency)

        '''
        print("tokens:",tokens)
        print("lemmas:", lemmas)
        print("lexicons:", lexicon_output)
        print("ners:", ner['tags'])
        print("chunk_type:", constituency_output)
        exit(1)
        '''

        return {'tokens': tokens,
                'lemmas': lemmas,
                'constituency': constituency_output,
                'ner_seq': ner['tags'],
                'ner': self.ner_spans(ner['words'], ner['tags']),
                'lexicon': lexicon_output,
                'dependency': dependency_output,
                'location': 'context',
                'sentence_idx': 0,
                'offset': offset}

    def get_dependency_output(self, dependency):
        words = dependency['words']
        # pos = dependency['pos']
        # predicted_dependencies = dependency['predicted_dependencies']
        predicted_heads = dependency['predicted_heads']

        # build graph
        edges = []
        for i, word in enumerate(words):
            head = predicted_heads[i]
            if head == 0:
                continue

            head -= 1  # allennlp heads count from 1

            edges.append(('{0}-{1}'.format(word, i),
                          '{0}-{1}'.format(words[head], head)))
        graph = nx.Graph(edges)
        '''
        for i in list(graph.nodes):
            for j in list(graph.nodes):
                dist = nx.shortest_path_length(graph, source=i, target=j)
                print("node1:{}, node2:{}, dist:{}".format(i, j, dist))
        '''
        return {'tokens': words,
                'graph': graph,
                'pos': dependency['pos'],
                'predicted_dependencies': dependency['predicted_dependencies'],
                'predicted_heads': dependency['predicted_heads']
                }

    def ner_spans(self, tokens, predicted_tags):
        predicted_spans = []
        i = 0
        while i < len(predicted_tags):
            tag = predicted_tags[i]
            # if its a U, add it to the list
            if tag[0] == 'U':
                current_tags = {'span': ' '.join(tokens[i: i + 1]),
                                'begin_idx': i,
                                'end_idx': i,
                                'tag': tag.split('-')[1]
                                }
                predicted_spans.append(current_tags)
            # if its a B, keep going until you hit an L.
            elif tag[0] == 'B':
                begin_idx = i
                while tag[0] != 'L':
                    i += 1
                    tag = predicted_tags[i]
                end_idx = i
                current_tags = {'span': ' '.join(tokens[begin_idx: end_idx + 1]),
                                'begin_idx': begin_idx,
                                'end_idx': end_idx,
                                'tag': tag.split('-')[1]
                                }
                predicted_spans.append(current_tags)
            i += 1
        return predicted_spans

    def find_idx(self, word, tokens, st, ed):
        for i in range(st, ed):
            for j in range(i, ed):
                if ' '.join(tokens[i:j+1]) == word:
                    return i, j+1
        return -1, -1

    def traverse_tree(self, tree, tokens, st, ed):
        st1, ed1 = self.find_idx(tree['word'], tokens, st, ed)
        ret = [{'span': tree['word'],
                'tag': tree['nodeType'],
                'begin_idx': st1,
                'end_idx': ed1 - 1}]
        if 'children' in tree:
            for child in tree['children']:
                ret += self.traverse_tree(child, tokens, st1, ed1)
        return ret

    def get_hatewords(self, filename=HATEWORD_FILE):
        lexicon = []
        with open(filename, 'r') as f:
            line = f.readline()
            while line:
                lexicon.append(line[:-1].lower())
                line = f.readline()
        return lexicon

    def get_sentiment(self, filename=SENTIMENT_FILE):
        lexicon = {}
        flag = 1
        with open(filename, 'r') as f:
            line = f.readline()
            while line:
                if flag:
                    flag = 0
                    continue
                words = line.strip('\n').split(',')
                lexicon[words[2]] = (words[3], words[5])
                line = f.readline()
        return lexicon

    def get_identities(self, filename=IDENTITY_FILE):
        identities = []
        with open(filename, 'r') as f:
            lines = f.readlines()
            for line in lines:
                identity = line.split('\t')[0]
                identities.append(identity)
        return identities

    def get_negations(self, filename=NEGATION_FILE):
        negations = []
        with open(filename, 'r') as f:
            lines = f.readlines()
            for line in lines:
                negations.append(line.strip())
        return negations

    def get_lexicon(self, _tokens):
        tags = []
        tokens = [token.lemma_.lower() for token in _tokens]
        #print("tokens:", tokens)
        #print("negation_lexicon:", self.negation_lexicon)
        for idx, token in enumerate(tokens):
            #print("token:", token)
            if token in self.identity_lexicon:
                tags.append('IDEN')
            elif token in self.negation_lexicon:
                tags.append('NOT')
            elif token in self.hatewords_lexicon:
                tags.append('HATE')
            elif token in self.sentiment_lexicon:
                _, y = self.sentiment_lexicon[token]
                if y == 'positive':
                    tags.append('POS')
                elif y == 'neutral':
                    tags.append('NEU')
                elif y == 'negative':
                    tags.append('NEG')
            else:
                tags.append('EMPTY')
        predicted_spans = []
        i = 0
        while i < len(tags):
            tag = tags[i]
            # if its a U, add it to the list
            if tag != 'EMPTY':
                current_tags = {'span': ' '.join(tokens[i:i+1]),
                                'begin_idx': i,
                                'end_idx': i,
                                'tag': tag
                                }
                predicted_spans.append(current_tags)
            i += 1
        # print("lexicon_spans:", predicted_spans)
        return predicted_spans