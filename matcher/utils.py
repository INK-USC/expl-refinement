import numpy as np
import string
import re
import networkx as nx
from allennlp.data.tokenizers import WordTokenizer
from spacy.lang.en import English
from spacy.tokenizer import Tokenizer

tokenizer = WordTokenizer()

COMMA_INDEX = {',': 0, '-LRB-': 1, '-RRB-': 2, '.': 3, '-': 4}
SPECIAL_CHARS = {' ': '_', '(': '[LEFT_BRACKET]', ')': '[RIGHT_BRACKET]', '.': '[DOT]', ',': '[COMMA]', '!': '[EX]'}
REVERSE_SPECIAL_CHARS = {v.lower(): k for k, v in SPECIAL_CHARS.items()}
REVERSE_SPECIAL_CHARS.update({v: k for k, v in SPECIAL_CHARS.items()})

class Occurrence:
    def __init__(self, span, begin_idx, end_idx,
                 location, sentence_idx, offset, confidence = 1.0):
        self.span = span
        self.begin_idx = begin_idx
        self.end_idx = end_idx
        self.location = location # 'C' (context) or 'Q' (question)
        self.sentence_idx = sentence_idx
        self.offset = offset
        self.confidence = confidence
        self.length = end_idx - begin_idx + 1

    def __str__(self):
        return "==Occurrence== {}, idx: [{}-{}], sent_id: {}-{}".format(self.span,
                                                                     self.begin_idx,
                                                                     self.end_idx,
                                                                     self.location,
                                                                     self.sentence_idx)

def remove_repeated(lst, keys):
    to_return = []
    unique_keys = set()
    for item in lst:
        key = ','.join([str(getattr(item, key)) for key in keys])
        if key not in unique_keys:
            unique_keys.add(key)
            to_return.append(item)
    return to_return

def get_tags(span, lst):
    ret = []
    #print("span:", span)
    lemma_span = ' '.join(span['lemmas'] if isinstance(span, dict) else span)
    #print("lemma_spans:", lemma_span)
    token_span = ' '.join(span['tokens'] if isinstance(span, dict) else span)
    #print("token_spans:", token_span)
    for item in lst:
        t = ' '.join(item['span']) if isinstance(item['span'], list) else item['span']
        #print("t:", t)
        if softened_string_eq(t, lemma_span) or softened_string_eq(t, token_span):
        # if normalize_answer(t) == normalize_answer(span) and len(normalize_answer(t)) > 0:
            ret.append(item['tag'])
    return ret

def get_span(info, key, span_type):
    to_return = []
    tag = key['tag']
    valid_type = [tag]
    if span_type == 'ner' and tag in NER_DICT:
        valid_type += NER_DICT[tag]
    elif span_type == 'lexicon' and tag in LEXICON_DICT:
        valid_type += LEXICON_DICT[tag]

    all_spans = info[span_type]
    for one_span in all_spans:
        if one_span['tag'] in valid_type:
            to_return.append(Occurrence(
                span = one_span['span'],
                begin_idx = one_span['begin_idx'],
                end_idx = one_span['end_idx'],
                location = info['location'],
                sentence_idx = info['sentence_idx'],
                offset = info['offset']
            ))
    return to_return

def get_st_ed(phrase, info):
    """
    :param phrase: a list of tokens for phrase
    :param tokens: a list of tokens for sentence
    :return: the st and ed indices
    """
    if isinstance(phrase, dict):
        phrase = phrase['lemmas']
    length_p = len(phrase)
    p_raw = ' '.join(phrase)
    sentence = info['lemmas']
    for idx in range(len(sentence) - length_p + 1):
        p_new = ' '.join(sentence[idx: idx + length_p])
        if softened_string_eq(p_raw, p_new):
            # use this line if using phrase_matcher
            # +1 because we add [CLS] token later
            # return (idx + 1, idx + length_p + 1)
            # use this line if using find_dummy
            #  deleted "+1" for now for Find_Dummy()
            return (idx, idx + length_p)
    return None

def get_st_ed_all(phrase, info):
    """
    :param phrase: a list of tokens for phrase
    :param tokens: a list of tokens for sentence
    :return: the st and ed indices
    """
    to_return = []
    if isinstance(phrase, dict):
        phrase = phrase['lemmas']
    length_p = len(phrase)
    p_raw = ' '.join(phrase)
    sentence = info['lemmas']
    for idx in range(len(sentence) - length_p + 1):
        p_new = ' '.join(sentence[idx: idx + length_p])
        if softened_string_eq(p_raw, p_new):
            # use this line if using phrase_matcher
            # +1 because we add [CLS] token later
            # return (idx + 1, idx + length_p + 1)
            # use this line if using find_dummy
            #  deleted "+1" for now for Find_Dummy()
            to_return.append((idx, idx + length_p))
    return to_return

def find_same_dependency(info, pattern_dep):
    """given a pattern_dependency, find span in the sentence (info) that fits the pattern"""

    def valid_head(lst1, lst2):
        """determine if heads are the same. -1 means pointing to outside"""
        for h1, h2 in zip(lst1, lst2):
            if h1 != h2 and h1 != -1 and h2 != -1:
                return False
        return True

    to_return = []
    new_dep = info['dependency'] # sentence to be matched with the pattern
    sent_len = len(info['tokens'])
    patt_len = pattern_dep['len']

    for st in range(sent_len - patt_len + 1):
        span = ' '.join(info['tokens'][st:st + patt_len])
        candidate_pos = new_dep['pos'][st:st + patt_len]
        candidate_dependencies = new_dep['predicted_dependencies'][st:st + patt_len]
        candidate_heads = np.array(new_dep['predicted_heads'][st:st + patt_len]) - st - 1

        if candidate_pos == pattern_dep['pos'] \
                and candidate_dependencies == pattern_dep['dependencies'] \
                and valid_head(pattern_dep['heads'], candidate_heads):
            to_return.append(Occurrence(
                span = span,
                begin_idx = st,
                end_idx = st + patt_len - 1,
                location = info['location'],
                sentence_idx = info['sentence_idx'],
                offset = info['offset']
            ))
    return to_return

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
        return re.sub(regex, ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def softened_string_eq(s1, s2):
    if s1 == s2:
        return True
    norm_s1 = normalize_answer(s1)
    norm_s2 = normalize_answer(s2)
    if norm_s1 == norm_s2 and len(norm_s1) > 0:
        return True
    return False

def tuple_fy(t):
    return (t,) if not isinstance(t, tuple) else t

def get_lemma(st):
    """input a string, get a list of lemmatized tokens in string."""
    if isinstance(st, list):
        st = ' '.join(st)
    output = tokenizer.tokenize(st)
    return {'tokens':[token.text.lower() for token in output],
            'lemmas': [token.lemma_ for token in output]}

def get_tokens(st):
    if isinstance(st, list):
        st = ' '.join(st)
    output = tokenizer.tokenize(st.lower())
    return [token.text for token in output]

def get_tokens_notlow(st):
    if isinstance(st, list):
        st = ' '.join(st)
    output = tokenizer.tokenize(st)
    return [token.text for token in output]

def update_inputs(inputs, args, lemma=True):
    """update x to its actual span"""
    ret = []
    for item in args:
        if item in ['X', 'Y', 'Z']:
            if isinstance(inputs[item], dict):
                item = get_lemma(inputs[item]['span'])
            elif isinstance(inputs[item], Occurrence):
                item = get_lemma(inputs[item].span)
            else:
                item = get_lemma(inputs[item])
        elif item == 'all':
            if isinstance(inputs[item], str):
                item = get_lemma(inputs[item])
            else:
                item = inputs['all']['lemmas']
        elif item == 'Question':
            item = inputs['instance'].question_info
        elif item == 'Sentence' or item == 'Context':
            if 'Context' in inputs and inputs['Context']:
                item = inputs['Context']
            else:
                item = inputs['instance'].context_info
        elif item == 'Answer':
            if isinstance(inputs['Answer'], dict):
                item = get_lemma(inputs['Answer']['span']) if lemma \
                    else get_tokens(inputs['Answer']['span'])
            else:
                item = get_lemma(inputs['Answer'].span) if lemma \
                    else get_tokens(inputs['Answer'].span)
        elif item in REVERSE_SPECIAL_CHARS:
            item = [REVERSE_SPECIAL_CHARS[item]]
            item = {'tokens': item, 'lemmas': item}
        else:
            item = get_lemma(item.replace('_', ' '))
        ret.append(item)
    if len(ret) == 1:
        ret = ret[0]
    return ret

def get_dependency_distance(oc1, oc2, inputs):
    dependency_output = inputs['instance'].context_info['dependency']
    graph = dependency_output['graph']
    tokens = inputs['instance'].context_info['tokens']
    min_dist = 100

    for x_pos in range(oc1.begin_idx, oc1.end_idx+1):
        for y_pos in range(oc2.begin_idx, oc2.end_idx+1):
            x_str = '{0}-{1}'.format(tokens[x_pos], x_pos)
            y_str = '{0}-{1}'.format(tokens[y_pos], y_pos)
            try:
                dist = nx.shortest_path_length(graph, source=x_str, target=y_str)
                min_dist = min(min_dist, dist)
            except nx.exception.NodeNotFound:
                print("-"*20)
                print("x_str:", x_str)
                print("y_str:", y_str)
                print("nodes:")
                for n in list(graph.nodes):
                    print(n)
                print("-" * 20)

    return min_dist

nlp = English()
tokenizer0 = Tokenizer(nlp.vocab)

def fill_whitespace_in_quote(sentence):
    """input: a string containing multiple sentences;
    output: fill all whitespaces in a quotation mark into underscore"""

    def convert_special_chars(s, flag):
        return SPECIAL_CHARS[s] if s in SPECIAL_CHARS and flag else s

    flag = False  # whether space should be turned into underscore, currently
    output_sentence = ''
    for i in range(len(sentence)):
        if sentence[i] == "\"":
            flag = not flag  # flip the flag if a quote mark appears
        output_sentence += convert_special_chars(sentence[i], flag)
    return output_sentence


def preprocess_sent(sentence):
    """input: a string containing multiple sentences;
    output: a list of tokenized sentences"""
    sentence = fill_whitespace_in_quote(sentence)
    output = tokenizer0(sentence)
    # tokens = [token.text for token in tokenizer.tokenize(sentence)]
    tokens = list(map(lambda x: x.text, output))
    ret_sentences = []
    st = 0

    # fix for ','
    new_tokens = []
    for i, token in enumerate(tokens):
        if token.endswith(','):
            new_tokens += [token.rstrip(','), ',']
        else:
            new_tokens += [token]
    tokens = new_tokens

    for i, token in enumerate(tokens):
        if token.endswith('.'):
            ret_sentences.append(tokens[st: i] + [token.strip('.')])
            st = i + 1
    return ret_sentences


CHUNK_DICT = {
    'N': ['N', 'NP', 'NN', 'NNS', 'NNP', 'NNPS'],
    'V': ['VP', 'VB', 'VBD', 'VBG', 'VBN',  'VBP', 'VBZ'],
    'P': ['PP'],
    'ADJ': ['JJ', 'JJR', 'JJS', 'ADJP'],
    'ADV': ['RB', 'RBR', 'RBS', 'ADVP'],
    'NUM': ['CD', 'QP'],
    'PRP': ['PRP', 'PRP$'],
}

NER_DICT = {
    'PERSON': ['PERSON'],
    'NORP': ['NORP'],
    'ORGANIZATION': ['ORG'],
    'GPE': ['GPE'],
    'LOCATION': ['GPE', 'FACILITY', 'ORG', 'LOCATION'],
    'DATE': ['DATE'],
    'TIME': ['DATE', 'TIME'],
    'NUMBER': ['PERCENT', 'QUANTITY', 'ORDINAL', 'CARDINAL', 'MONEY'],
    'PERCENT': ['PERCENT'],
    'MONEY': ['MONEY'],
    'ORDINAL': ['ORDINAL'],
}

LEXICON_DICT = {
    'POS':['POS'],
    'NEG':['NEG'],
    'NEU':['NEU'],
    'HATE':['HATE'],
    'NOT':['NOT'],
    'IDEN':['IDEN']
}

TAGS_OF_INTEREST = ['NP', 'VP', 'PP',
                    'NN', 'NNS', 'NNP', 'NNPS', 'PRP', 'PRP$',
                    'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']

VAR_NAMES = ['X', 'Y', 'Z']