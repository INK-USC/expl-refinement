import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import logging

from dictionary import STRING2PREDICATE, WORD2NUMBER, OPS_FEATURE, RAW_LEXICON
from utils import SPECIAL_CHARS, REVERSE_SPECIAL_CHARS
from rule import Rule
from nltk.ccg import chart, lexicon
from read_instance import InstancePreprocess
from utils import preprocess_sent


MAX_PHRASE_LEN = 4
BEAM_WIDTH = 100

logger = logging.getLogger(__name__)


def string_to_predicate(s):
    """input: one string (can contain multiple tokens with ;
    output: a list of predicates."""
    if s != ',' and s not in REVERSE_SPECIAL_CHARS:
        s = s.lower().strip(',')
    if s.startswith("$"):
        return [s]
    elif s.startswith("\"") and s.endswith("\""):
        return ["'" + s[1:-1] + "'"]
    elif s in STRING2PREDICATE:
        return STRING2PREDICATE[s]
    elif s.isdigit():
        return ["'" + s + "'"]
    elif s in WORD2NUMBER:
        return ["'" + WORD2NUMBER[s] + "'"]
    else:
        return []


def tokenize(sentence):
    """input: a list of tokens;
    output: a list of possible tokenization of the sentence;
    each token can be mapped to multiple predicates"""
    # log[j] is a list containing temporary results using 0..(j-1) tokens
    log = {i: [] for i in range(len(sentence) + 1)}
    log[0] = [[]]
    for i, token in enumerate(sentence):
        for _range in range(1, MAX_PHRASE_LEN + 1):
            if i + _range > len(sentence):
                break
            phrase = ' '.join(sentence[i:i + _range])
            predicates = string_to_predicate(phrase)
            for temp_result in log[i]:
                for predicate in predicates:
                    log[i + _range].append(temp_result + [predicate])
            if token.startswith("\""):  # avoid --"A" and "B"-- treated as one predicate
                break
    return log[len(sentence)]


def get_word_name(layer, st, idx):
    return "$Layer{}_St{}_{}".format(str(layer), str(st), str(idx))


def get_entry(word_name, category, semantics):
    return "\n\t\t{0} => {1} {{{2}}}".format(word_name, str(category), str(semantics))

def quote_word_lexicon(sentence):

    def is_quote_word(token):
        return (token.startswith("\'") and token.endswith("\'")) \
            or (token.startswith("\"") and token.endswith("\""))

    ret = ""
    for token in sentence:
        if is_quote_word(token):
            ret += get_entry(token, 'NP', token)
            ret += get_entry(token, 'N', token)
            ret += get_entry(token, 'NP', "'@In'({},'all')".format(token))
            if token[1:-1].isdigit():
                ret += get_entry(token, 'NP/NP', "\\x.'@Num'({},x)".format(token))
                ret += get_entry(token, 'N/N', "\\x.'@Num'({},x)".format(token))
                ret += get_entry(token, 'PP/PP/NP/NP', "\\x y F.'@WordCount'('@Num'({},x),y,F)".format(token))
                ret += get_entry(token, 'PP/PP/N/N', "\\x y F.'@WordCount'('@Num'({},x),y,F)".format(token))

    return ret


class Parser(nn.Module):
    def __init__(self):
        super(Parser, self).__init__()
        self.raw_lexicon = RAW_LEXICON
        self.beam_width = BEAM_WIDTH
        self.feature_size = len(OPS_FEATURE)
        self.theta = nn.Parameter(torch.randn(self.feature_size, dtype=torch.float64))
        # self.theta.data.uniform_(0.0, 0.2)
        # torch.nn.init.xavier_uniform(self.theta)

        print('Parser of dimension {} is initialized.'.format(self.feature_size))
        # print('Initial weights:', self.theta)

    def forward(self, instances):
        """
        :param instances: [[]], for each instance, there is a list of successful parses
        :return: [[]], the score for each parse in each sentence.
        """
        ret = []
        for instance in instances:
            inputs = [item[2] for item in instance]
            t = torch.tensor(inputs, dtype=torch.float64)
            logits = torch.matmul(self.theta, t.t())
            probs = F.softmax(logits, dim=0)
            ret.append(probs)
        return ret

    def loss(self, preds, xys):
        ret = 0.0
        for pred, xy in zip(preds, xys):
            labels = [item[1] for item in xy]
            t = torch.tensor(labels, dtype=torch.float64)
            ret -= torch.matmul(torch.log(pred), t.t())
        return ret


    def forward_single(self, str_rep):
        rule = Rule(str_rep)
        t = torch.tensor(rule.features, dtype=torch.float64)
        ret = torch.matmul(self.theta, t.t())
        return ret.item()


    def parse(self, sentence, beam=True):
        """
        :param sentence: a list of tokens in one sentence.
                e.g. ['"may_be"', '$Is', '$Between', '$ArgX', '$And', '$ArgY']
        :return: a list of successful parses.
        """
        beam_lexicon = copy.deepcopy(self.raw_lexicon) + quote_word_lexicon(sentence)

        # the first index of forms is layer
        # the second index of forms is starting index
        all_forms = [[[token] for token in sentence]]

        # parsed results to be returned
        ret = []

        # Width of tokens to be parsed. Start with width 1 and stack to len(sentence)
        for layer in range(1, len(sentence)):
            layer_form = []

            # update the lexicon from previous layers
            lex = lexicon.fromstring(beam_lexicon, True)
            parser = chart.CCGChartParser(lex, chart.DefaultRuleSet)

            # parse the span (st, st+layer)
            for st in range(0, len(sentence) - layer):
                form = []
                memory = []  # keep a memory and remove redundant parses
                word_index = 0
                ed = st + layer
                # try to combine (st, split), (split+1, ed) into (st, ed)
                for split in range(st, ed):

                    # get candidates for (st, split) and (split+1, ed)
                    words_L = all_forms[split-st][st]
                    words_R = all_forms[ed-split-1][split+1]

                    for word_L in words_L:
                        for word_R in words_R:
                            # try to combine word_L and word_R
                            try:
                                for parse in parser.parse([word_L, word_R]):
                                    token, _ = parse.label()
                                    category, semantics = token.categ(), token.semantics()
                                    memory_key = str(category) + '_' + str(semantics)
                                    if memory_key not in memory:
                                        memory.append(memory_key)
                                        word_index += 1
                                        form.append((parse, category, semantics, word_index))
                            except (AssertionError, SyntaxError) as e:
                                logger.info('Error when parsing {} and {}'.format(word_L, word_R))
                                logger.info('Error information: {}'.format(e.args))

                # beam here. todo: implement feature selection and beam; use if beam
                to_add = []
                for item in form:
                    parse, category, semantics, word_index = item
                    word_name = get_word_name(layer, st, word_index)
                    to_add.append(word_name)
                    beam_lexicon += get_entry(word_name, category, semantics)

                    # if this is the last layer (covering the whole sentence)
                    # add this to output
                    if layer == len(sentence) - 1:
                        ret.append(str(semantics))
                layer_form.append(to_add)

            all_forms.append(layer_form)

        # filter incomplete parses
        ret = list(filter(lambda x: x.startswith("'@"), ret))
        ret = sorted(ret, key=lambda x: self.forward_single(x), reverse=True)
        return list(ret)


if __name__ == "__main__":
    sent = 'X is country. Y is negative. X is less than 3 dependencies from Y.'

    sentences = preprocess_sent(sent)
    # print(sentences)
    tokenized_sentences = [tokenize(sentence) for sentence in sentences]
    # print(tokenized_sentences)

    parser = Parser()

    print('=' * 20 + ' start parsing ' + '=' * 20 + '\n')

    rule_list_sentence = []
    for i, sentence in enumerate(tokenized_sentences):
        print('=== sentence {}: {}'.format(i, sentences[i]))
        for potential_sentence in sentence:
            print('sentence predicates: {}'.format(potential_sentence))
            all_possible_parses = parser.parse(potential_sentence)
            if len(all_possible_parses) > 0:
                rule_list_sentence += all_possible_parses
                print('parses: {}\n'.format(all_possible_parses))

    rule = Rule(rule_list_sentence[0])
    sentence = " Well, In the end. Sweden has proven to be a failure. Especially anything to do with feminsium."
    label = 1
    instance_reader = InstancePreprocess(pre=True)
    instance = instance_reader.read_one(sentence, label)
    inputs = {'Label': label,
              'X': "Sweden",
              'Y': "failure",
              'Z': "",
              'instance': instance,
              'pretrained_modules': None,
              'soft': 0
              }
    result = rule.execute(inputs)
    print("result:",result)


    # funcs = [Rule(item) for item in ret]
    # test_sent = "bag may be hello in ArgY".split(' ')
    # rule1 = Rule(ret[3])
    # print(rule1.tree)
    # ret = rule1.func({'s': test_sent, 'ArgX': 'hello'})
    # print(ret)
