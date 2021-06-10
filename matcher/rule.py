import re
import copy
import logging
import torch
import numpy as np
from modules import OPS_SOFT
from allennlp.data.tokenizers import WordTokenizer
from dictionary import LEN_OP_FEATURE, OPS_FEATURE
from utils import normalize_answer
from itertools import product
from utils import CHUNK_DICT, NER_DICT, LEXICON_DICT, VAR_NAMES
from modules import fill, _and
from utils import preprocess_sent

# sentence_splitter = SpacySentenceSplitter()
tokenizer = WordTokenizer()

logger = logging.getLogger(__name__)
OPS = OPS_SOFT

class Rule:
    """Same as LF in NMET."""

    def __init__(self, inputs, str_rep=""):
        # This part is using NMN
        # String Representation
        self.raw_str = str_rep
        self.features = [0] * LEN_OP_FEATURE
        self.tree = ('.root',)
        self.inputs = inputs
        # self.parsed = self.parse(str_rep)
        # Executable Function

        self.tree = self.str2tree(str_rep)
        self.features = self.get_features()
        self.func = self.recurse(self.tree, inputs)
        #except:
        #    print("Rule not executable")
        #    logging.info("Rule not executable: {}".format(self.tree))
        #    self.func = Rule.return_false

    @classmethod
    def return_false(cls, *argv):
        return False

    @classmethod
    def return_true(cls, *argv):
        return True

    def __str__(self):
        return self.raw_str

    @classmethod
    def str2tree(cls, str_rep):
        # print(str_rep)
        sem = list(filter(lambda i: i, re.split(',|(\()', str_rep)))
        # print("sem:", sem)
        for idx, w in enumerate(sem):
            if w == '(' and idx > 0:
                sem[idx], sem[idx-1] = sem[idx-1], sem[idx]
        sem_ = ''
        for i, s in enumerate(sem):
            if sem[i-1] != '(':
                sem_ += ','
            sem_ += sem[i]
        try:
            em = ('.root', eval(sem_[1:]))
        except SyntaxError as e:
            #logging.info('Error when transforming'.format(str_rep))
            #logging.info('Error information: {}'.format(e.args))
            em = ('.root',)
        # print("em:", em)
        return em

    @classmethod
    def recurse(cls, sem, inputs):
        #print("recurse:",sem)
        if isinstance(sem, tuple):
            if sem[0] not in OPS:
                logging.info('Error: {} not implemented'.format(sem[0]))
                raise NotImplementedError
            op = OPS[sem[0]]
            args = [cls.recurse(arg, inputs) for arg in sem[1:]]
            if len(args) > 1 and args[1] == 'ExactWord':
                args[1] = inputs[args[0]]
            return op(*args) if args else op
        else:
            return sem

    def execute(self, args):
        # give args and output score
        try:
            ret = self.func(args)
        except (TypeError, AttributeError, ValueError, AssertionError) as e:
            logging.info('Error in executing the rule: {}'.format(self.tree))
            logging.info('Error information: {}'.format(e.args))
            ret = 0.0

        return ret

    def get_features(self):
        """returns a feature vector for the current parse."""
        feature_list = []
        self.collect_features(self.tree, feature_list)
        #print("feature_list:", feature_list)

        feature_vec = [0] * LEN_OP_FEATURE
        for feature in feature_list:
            key = feature[0]+feature[1]
            if key in OPS_FEATURE:
                feature_vec[OPS_FEATURE[key]] += 1
        #print("feature_vec:", feature_vec)
        return feature_vec

    def collect_features(self, semantics, feature_list):
        if isinstance(semantics, tuple):
            for child in semantics[1:]:
                if isinstance(child, tuple) and child[0] != semantics[0]:
                    feature_list.append((semantics[0], child[0]))
                self.collect_features(child, feature_list)

    def clean_rules(self):
        self.func = None

    def reload_rules(self):
        self.func = self.recurse(self.tree, self.inputs)



class Variable:
    """One variable in the rule.
    e.g. X(NP)=packet switching; Y(VBZ)=characterize; Z(NP,LOCATION)=USC
    """

    def __init__(self, name, chunk=None, ner=None, value=None, lexicon=None):
        self.name = name
        self.chunk = chunk
        self.ner = ner
        self.value = value
        self.dependency = []
        self.candidates = []
        self.lexicon = lexicon
        self.in_context = False
        self.loc_in_context = []

    @classmethod
    def get_variable(cls, instance, value, name):

        def remove_repeated(lst):
            return list(filter(lambda x: x, set(lst)))

        def get_tag(lst, key):
            ret = []
            for item in lst:
                t = ' '.join(item['span']) if isinstance(item['span'], list) else item['span']
                if normalize_answer(t) == normalize_answer(key) and len(normalize_answer(t)) > 0:
                    ret.append(item)
            return ret

        def locate(value, tokens, lemmas, offset):
            ret = []
            for st in range(len(tokens)):
                for ed in range(st+1, len(tokens) + 1):
                    if ' '.join(value).lower() == ' '.join(lemmas[st:ed]).lower():
                        ret.append({'st': st + offset, 'ed': ed + offset,
                                    'st_in_sentence': st, 'ed_in_sentence': ed,
                                    'original': ' '.join(tokens[st:ed])})
            return ret

        def dependency_info(dep_dict, st, ed):
            old_heads = dep_dict['predicted_heads'][st:ed]
            new_heads = [item-1-st if st<=item-1<ed else -1 for item in old_heads]
            return {
                'pos': dep_dict['pos'][st:ed],
                'dependencies' : dep_dict['predicted_dependencies'][st:ed],
                'heads': new_heads,
                'offset': st,
                'len': ed - st
            }

        variable = Variable(name)

        chunk_tag = []
        ner_tag = []
        lexicon_tag = []
        dependency = []
        value_tokenized = tokenizer.tokenize(value)
        value_tokens = [token.text for token in value_tokenized]
        value_lemmas = [token.lemma_ for token in value_tokenized]

        sentence = instance.context_info
        locations_in_sentence = locate(value_lemmas,
                                       sentence['tokens'],
                                       sentence['lemmas'],
                                       offset=sentence['offset'])
        if len(locations_in_sentence) > 0:
            variable.in_context = True
            for item in locations_in_sentence:
                chunk_tag += get_tag(sentence['constituency'], item['original'])
                ner_tag += get_tag(sentence['ner'], item['original'])
                lexicon_tag += get_tag(sentence['lexicon'], item['original'])
                if 'dependency' in sentence:
                    dependency.append(dependency_info(sentence['dependency'], item['st_in_sentence'], item['ed_in_sentence']))

        variable.chunk = chunk_tag
        variable.ner = ner_tag
        variable.lexicon = lexicon_tag
        variable.value = value
        variable.tokens = value_tokens
        variable.lemmas = value_lemmas
        variable.loc_in_context = locations_in_sentence
        variable.dependency = dependency
        return variable


class AnsFunc:
    def __init__(self):
        # Variables used in the Func
        # e.g. X(NP), Y(VBZ), ANS(PP)
        self.variables = {}

        self.all_rules = []

        # The QA instance that this AnsFunc is extracted from
        self.reference_instance = None
        self.label = None
        self.advice = []
        self.version = ''
        self.score = 0

    def all_rules_str(self):
        return ','.join([item.raw_str for item in self.all_rules])

    def instantiate(self, instance):
        self.reference_instance = instance
        self.label = instance.label

    def clean_vars(self):
        for var in self.variables.values():
            var.value = ""
            var.candidates = []
            var.loc_in_question = -1
            var.loc_in_context = -1

    def set_version(self, version):
        self.version = version

    def set_quality(self, score):
        self.score = score

    def set_advice(self, part_b):
        Attr0Inc = []
        Attr0Dec = []
        Inter0Inc = []
        Inter0Dec = []
        Attr1Inc = []
        Attr1Dec = []
        Inter1Inc = []
        Inter1Dec = []

        # print(sent)
        sentences = preprocess_sent(part_b)
        # print(sentences)
        for sentence in sentences:
            if sentence[0] == "Attribution0":
                if sentence[-1] == "increased":
                    Attr0Inc.append(sentence[2])
                elif sentence[-1] == "decreased":
                    Attr0Dec.append(sentence[2])
            elif sentence[0] == "Attribution1":
                if sentence[-1] == "increased":
                    Attr1Inc.append(sentence[2])
                elif sentence[-1] == "decreased":
                    Attr1Dec.append(sentence[2])
            elif sentence[0] == "Interaction0":
                if sentence[-1] == "increased":
                    Inter0Inc.append((sentence[2], sentence[4]))
                elif sentence[-1] == "decreased":
                    Inter0Dec.append((sentence[2], sentence[4]))
            elif sentence[0] == "Interaction1":
                if sentence[-1] == "increased":
                    Inter1Inc.append((sentence[2], sentence[4]))
                elif sentence[-1] == "decreased":
                    Inter1Dec.append((sentence[2], sentence[4]))
        self.advice = [Attr0Inc,
                       Attr0Dec,
                       Inter0Inc,
                       Inter0Dec,
                       Attr1Inc,
                       Attr1Dec,
                       Inter1Inc,
                       Inter1Dec]

    def answer(self, instance, pretrained_modules=None, thres=1.0, soft=False):
        """
        :param instance: a SQuADExampleExtended instance
        :return: answer, confidence
        """
        def get_key(state):
            key = ''
            for var_name in VAR_NAMES:
                item = state[var_name] if var_name in state else None
                if item:
                    key += ','.join([var_name, str(item.begin_idx), str(item.end_idx),
                                     str(item.location), str(item.sentence_idx)]) + '_'
            key += ',context_idx_' + str(state['context_idx'])
            return key

        def filter_states(list_of_states, soft):
            unique_keys = []
            ret_list = []
            for state in list_of_states:
                key = get_key(state)
                if key not in unique_keys:
                    unique_keys.append(key)
                    ret_list.append(state)
            if soft:
                ret_list = sorted(ret_list, key=lambda x: x['confidence'], reverse=True)
                ret_list = ret_list[:15]
            return ret_list

        version = self.version

        if '-' in version:
            versions = version.split('-')
        else:
            versions = None
        
        # Fill candidates for each variable
        variables = self.variables.values()
        for var_id, var in enumerate(variables):
            # var.candidates = self.fill(var, instance)
            if versions is not None:
                var.candidates = fill(self.reference_instance.context_info, var, instance.context_info, pretrained_modules, versions[var_id], soft)
            else:
                var.candidates = fill(self.reference_instance.context_info, var, instance.context_info, pretrained_modules, version, soft)

        ## Initialize states
        initial_state = {k: None for k in self.variables}
        # initial_state['Context'] = None
        initial_state['instance'] = instance
        initial_state['soft'] = soft
        initial_state['version'] = version
        initial_state['pretrained_modules'] = pretrained_modules
        thres = thres if soft else 1.0
        n_vars = len(self.variables)
        var_list = [key for key in self.variables.keys()]
        all_answer = []
        prev_states = []
        sentence = instance.context_info
        new_state = copy.copy(initial_state)
        new_state['Context'] = sentence
        new_state['reference'] = self.reference_instance.context_info
        new_state['context_idx'] = 0
        new_state['all'] = {'tokens': sentence['tokens'] + instance.context_info['tokens'],
                            'lemmas': sentence['lemmas'] + instance.context_info['lemmas']}
        prev_states.append(new_state)

        for j in var_list:  # Fill XYZ Ans
            new_states = []
            for state in prev_states:
                # for j in state.keys():
                #     if (j in VAR_NAMES) and (state[j] is None) and not (i < n_vars - 1 and j == 'Answer'):
                candidates = self.variables[j].candidates
                for candidate in candidates:
                    a_new_state = copy.copy(state)
                    a_new_state[j] = candidate
                    confidence_for_this_new_state = self.eva_state(a_new_state, soft)
                    if isinstance(confidence_for_this_new_state, torch.Tensor):
                        confidence_for_this_new_state_scalar = confidence_for_this_new_state.item()
                    else:
                        confidence_for_this_new_state_scalar = float(confidence_for_this_new_state)
                    assert np.less_equal(confidence_for_this_new_state_scalar, 1.0)
                    if np.greater_equal(confidence_for_this_new_state_scalar, thres):
                        a_new_state['confidence'] = confidence_for_this_new_state_scalar
                        new_states.append(a_new_state)
            new_states = filter_states(new_states, version)
            prev_states = new_states

        prev_states = filter_states(prev_states, version)

        return prev_states

    def eva_state(self, inputs, soft=False):
        """
        inputs may be partially filled; only some of the rules can be evaluated.
        select these rules and evaluate
        :return: Boolean
        """
        def vars_needed(str_rep, var_names):
            ret = []
            for var_name in var_names:
                if '\'' + var_name + '\'' in str_rep:
                    ret.append(var_name)
            return set(ret)

        def overlap(st1, ed1, st2, ed2):
            return st1 <= st2 <= ed1 or st1 <= ed2 <= ed1 or \
                   st2 <= st1 <= ed2 or st2 <= ed1 <= ed2

        filled_vars = [item[0] for item in filter(lambda x: x[1], inputs.items())]
        # inputs_copy = copy.copy(inputs)

        # Check the variables are not the same
        for i in filled_vars:
            for j in filled_vars:
                if i != j and i in VAR_NAMES and j in VAR_NAMES:
                    if inputs[i].span in inputs[j].span or inputs[j].span in inputs[i].span:
                        return False
        #                 and inputs[i]['location'] == inputs[j]['location']:
        #             st1, ed1 = inputs[i]['begin_idx'], inputs[i]['end_idx']
        #             st2, ed2 = inputs[j]['begin_idx'], inputs[j]['end_idx']
        #             if overlap(st1, ed1, st2, ed2):
        #                 return False

        probs_list = []
        for rule in self.all_rules:
            vars_needed_for_this_rule = vars_needed(rule.raw_str, VAR_NAMES)
            # print("vars_needed_for_this_rule:", vars_needed_for_this_rule)
            if vars_needed_for_this_rule.issubset(set(filled_vars)):
                probs_list.append(rule.execute(inputs))
                # if not np.equal(1, rule.execute(inputs)):
                #     return False
        for var in filled_vars:
            if var in VAR_NAMES:
                probs_list.append(inputs[var].confidence)
        # print("probs_list:", probs_list)
        return _and(probs_list, soft)

    def eva(self, rules, inputs):
        for rule in rules:
            if not rule.execute(inputs):
                return False
        return True

    def add_variable(self, v: Variable):
        """
        :param v: a variable to be added to the dict
        """
        name = v.name
        self.variables[name] = v

    def add_rule(self, inputs,  exp):
        self.all_rules.append(Rule(inputs, exp))

    def clean_rules(self):
        for rule in self.all_rules:
            rule.clean_rules()

    def reload_rules(self):
        for rule in self.all_rules:
            rule.reload_rules()

    def get_candidate_list(self, variables):
        var_list = variables.keys()
        candidate_list = []
        for name in var_list:
            var = self.variables[name]
            if len(var.candidates) == 0:
                return var_list, []
            if not candidate_list:
                candidate_list = var.candidates
            else:
                candidate_list = list(product(candidate_list, var.candidates))
                if isinstance(candidate_list[0][0], tuple):
                    candidate_list_new = []
                    for a, b in candidate_list:
                        candidate_list_new.append(a + (b,))
                    candidate_list = candidate_list_new
        return var_list, candidate_list

    def delete_redundant_rules(self):
        var_list = list(self.variables.keys())
        var_list.remove('Answer')
        n_all_rules = len(self.all_rules)
        # var_list = [item for item in ['X', 'Y', 'Z'] if item in self.variables.keys()]
        for idx0, var in enumerate(var_list[::-1]):
            kept_idx = n_all_rules - (idx0 + 1)
            temp_rule = self.all_rules[kept_idx]
            for rule in self.all_rules[:-len(var_list)]:
                if var in rule.raw_str and 'Question' not in rule.raw_str:
                    self.all_rules.pop(kept_idx)
                    break

if __name__ == "__main__":
    # str_rep = "'@And'('@Is'('is','@between'('@And'('citySPACEin','ArgX'))),'@And'('@Is'('citySPACEin','@Direct'('@Left0'('ArgY'))),'@Is'('citySPACEin','@between'('@And'('ArgY','ArgX')))))"
    # str_rep = "'@Is'('True','@Greater'('ArgX'))"
    # str_rep = "'@And'('@Is'('is','@Between'('@And'('city','bag'))),'@Is'('not','@AtMost'('@Left'('bag'),'@Num'('6','tokens'))))"
    # str_rep = "'@In'('ff', 'aabbccff')"
    # str_rep = "'@And'('@Is'('ArgY','@Chunk'('V')),'@Is'('may_be','@Between'('@And'('ArgY','ArgX'))))"
    str_rep = "'@Is'('X','@In0'('Context'))"
    # str_rep = "'@In1'('@Is'('X','Context'),'Context')"
    r1 = Rule(str_rep)
    sent = "city is not beep no ee ff bag".split(' ')

    ret = r1.func(sent)
    print(ret)
