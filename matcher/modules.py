import numpy as np
import copy
import torch
from utils import get_span, get_dependency_distance, get_tags, get_st_ed, get_st_ed_all, \
    find_same_dependency, remove_repeated, \
    softened_string_eq, tuple_fy, update_inputs, Occurrence, \
    CHUNK_DICT, NER_DICT, LEXICON_DICT, TAGS_OF_INTEREST, SPECIAL_CHARS
# from Phrase_Matcher_Predictor import Phrase_Mathcer_Predictor
#
# q2c_predictor = Phrase_Mathcer_Predictor('find_q2c', 0)
# q2q_predictor = Phrase_Mathcer_Predictor('fill_q2q', 1)

MAXINT = int(2e10)
WIDTH = 3

OPS_SOFT = {
    ".root": lambda xs: lambda c: root(xs, c),
    "@Is": lambda w, func: lambda inputs: _is(func, w, inputs),
    "@Is0": lambda x, y: lambda inputs: _is0(y, x, inputs),
    "@Num": lambda x, y: int(x),
    "@And": lambda x, y: merge(x, y),

    "@In": lambda x, y: lambda inputs: _in(x, y, inputs),
    "@In0": lambda s: lambda key: lambda inputs: _in(key, s, inputs),
    "@In1": lambda func, y: lambda inputs: _in1(func, y, inputs),

    "@Between": lambda keys: lambda w: lambda inputs: between({}, keys, w, inputs),
    "@Sandwich": lambda keys: lambda w: lambda inputs: between({'tight': True}, keys, w, inputs),

    "@Left": lambda key: lambda args: lambda w: lambda inputs: left(args, key, w, inputs),
    "@Right": lambda key: lambda args: lambda w: lambda inputs: right(args, key, w, inputs),
    "@Left0": lambda key1, key2: lambda inputs: left({}, key1, key2, inputs),
    "@Right0": lambda key1, key2: lambda inputs: right({}, key1, key2, inputs),
    "@Left1": lambda key: lambda w: lambda inputs: left({}, key, w, inputs),
    "@Right1": lambda key: lambda w: lambda inputs: right({}, key, w, inputs),

    "@DepDist": lambda key: lambda args: lambda w: lambda inputs: dep_dist(args, key, w, inputs),
    "@DepDist0": lambda key1, key2: lambda inputs: dep_dist({}, key1, key2, inputs),
    "@DepDist1": lambda key: lambda w: lambda inputs: dep_dist({}, key, w, inputs),

    "@Direct": lambda func: lambda w: lambda inputs: func({'dist': 0})(w)(inputs),
    "@LessThan": lambda func, dist: lambda w: lambda inputs: func({'dist': dist})(w)(inputs),

    "@StartsWith": lambda key1, key2: lambda inputs: starts_with(key1, key2, inputs),
    "@EndsWith": lambda key1, key2: lambda inputs: ends_with(key1, key2, inputs),

    "@NER": lambda ner_type: lambda w: lambda inputs: ner(ner_type, w, inputs),
    "@Chunk": lambda chunk_type: lambda w: lambda inputs: chunk(chunk_type, w, inputs),
    "@Lexicon": lambda lexicon_type: lambda w: lambda inputs: lexicon(lexicon_type, w, inputs)
}

def root(funcs, inputs):
    #if 'soft' not in inputs:
    #    inputs['soft'] = False
    soft = inputs['soft']
    funcs = tuple_fy(funcs)
    all_probs = []
    for func in funcs:
        ret = func(inputs)
        if callable(ret):
            all_probs.append(0.0)
        else:
            all_probs.append(ret)
    return _and(all_probs, soft)

def fill(ref_s, ref_p, s, pretrained_modules, version, soft):
    """Given a reference sentence (ref_s) and a reference phrase (ref_p),
    list out all candidate phrases in a new sentence (s).
    s can be a question or a sentence in the context.
    (Is this true? variables should be in Q?)
    -- Similar to variable.fill() in previous version
        -- Temporarily assume ref_s is a 'info' type, ref_p is a variable type.
    """
    candidates = []
    # soft = False
    ### strict
    if version == 'strict' or version == 'hard':
        '''
        tokens = ref_p.tokens
        lemmas = ref_p.lemmas
        token_spans = get_st_ed_all(tokens, s)
        if len(token_spans) > 0:
            for token_span in token_spans:
                (st, ed) = token_span
                candidates.append(Occurrence(
                    span=' '.join(tokens),
                    begin_idx=st,
                    end_idx=ed - 1,
                    location=s['location'],
                    sentence_idx=s['sentence_idx'],
                    offset=s['offset']
                ))
        lemma_spans = get_st_ed_all(lemmas, s)
        if len(lemma_spans) > 0:
            for lemma_span in lemma_spans:
                (st, ed) = lemma_span
                candidates.append(Occurrence(
                    span=' '.join(tokens),
                    begin_idx=st,
                    end_idx=ed - 1,
                    location=s['location'],
                    sentence_idx=s['sentence_idx'],
                    offset=s['offset']
                ))
        '''
        results = find(ref_s, {"tokens": ref_p.tokens, "lemmas": ref_p.lemmas}, s, pretrained_modules, soft)
        candidates += results

    ### hard
    if version == 'hard':
        for chunk_type in ref_p.chunk:
            candidates += get_span(s, chunk_type, 'constituency')
        for ner_type in ref_p.ner:
            candidates += get_span(s, ner_type, 'ner')
        for lexicon_type in ref_p.lexicon:
            candidates += get_span(s, lexicon_type, 'lexicon')
        for dependency_occurrence in ref_p.dependency:
            candidates += find_same_dependency(s, dependency_occurrence)
    '''
    # run the fill module to get an extended list
    if version == 'soft':
        query_span = get_st_ed(ref_p.lemmas, ref_s) if ref_s else None
        if query_span:
            q2q_predictor = pretrained_modules['fill_q2q']
            target_spans = [(constituency['begin_idx'], constituency['end_idx'] + 1)
                            for constituency in s['constituency']
                            if constituency['tag'] in TAGS_OF_INTEREST]
            span_scores = q2q_predictor.predict_fill_custom(ref_s, s, query_span, target_spans, topn=3)
            for (st, ed), score in span_scores[:WIDTH]:
                # the st,ed indices are with [CLS] token, so we minus 1
                if 1 <= st <= ed <= len(s['tokens']) + 1:
                    # exclude spans containing [CLS] and [SEP]
                    occurrence = Occurrence(
                        span = ' '.join(s['tokens'][st:ed]),
                        begin_idx = st,
                        end_idx = ed - 1,
                        location=s['location'],
                        sentence_idx=s['sentence_idx'],
                        offset=s['offset'],
                        confidence=score
                    )
                    occurrence.__str__()
                    candidates.append(occurrence)
    '''
    return remove_repeated(candidates, ['span','begin_idx','end_idx'])


def find(ref_q, ref_p, s, pretrained_modules, soft):
    """ref_p is a list of tokens (lemmas), ref_q / s are info_type.
    should be using lemmatized ref_p.
    ref_p must be from ref_q."""

    length_p = len(ref_p['lemmas'])
    p_raw_lemma = ' '.join(ref_p['lemmas'])
    p_raw_lower = ' '.join(ref_p['tokens'])

    to_return = []
    sentence_lemma = s['lemmas']
    sentence_lower = [token.lower() for token in s['tokens']]
    for idx in range(len(sentence_lemma) - length_p + 1):
        p_new_lemma = ' '.join(sentence_lemma[idx: idx + length_p])
        p_new_lower = ' '.join(sentence_lower[idx: idx + length_p])
        if softened_string_eq(p_raw_lemma, p_new_lemma) or softened_string_eq(p_raw_lower, p_new_lower):
            to_return.append(Occurrence(
                span = p_new_lower,
                begin_idx = idx,
                end_idx = idx + length_p - 1,
                location = s['location'],
                sentence_idx = s['sentence_idx'],
                offset = s['offset']
            ))

    # run the find module to get an extended list
    query_span = get_st_ed(ref_p, ref_q)

    #remove len(to_return) == 0
    if soft and query_span:
        find_predictor = pretrained_modules['find']
        # use this line if using phrase_matcher
        # st0 = s['offset'] + s['sentence_idx'] + 1
        # use this line if using dummy_find
        st0 = 0
        # st0 = s['offset'] # + s['sentence_idx'] + 1 # using dummy_find / don't need this.
        # ed0 = st0 + len(s['tokens'])
        target_spans = [(constituency['begin_idx'] + st0, constituency['end_idx'] + st0 + 1)
                        for constituency in s['constituency']
                        if constituency['tag'] in TAGS_OF_INTEREST]
        span_scores = find_predictor.predict_find_custom(s['sentence_idx'], ref_q, query_span, target_spans, topn=3)
        # find raw string ref_p in s.
        # else:
        #     sent_score, span_scores = q2c_predictor.predict([""], query_span, target_sentence)
        for (st, ed), score in span_scores:
            # the st,ed indices are with [CLS] token, so we minus 1
            # if 1 <= st <= ed <= len(s['tokens']) + 1:
            # exclude spans containing [CLS] and [SEP]
            to_return.append(Occurrence(
                span = ' '.join(s['tokens'][st-st0:ed-st0]),
                begin_idx = st-st0,
                end_idx = ed-st0-1,
                location=s['location'],
                sentence_idx=s['sentence_idx'],
                offset=s['offset'],
                confidence=score
            ))

    return remove_repeated(to_return, ['span','begin_idx','end_idx'])

def find_in_tokens(ref_q, ref_p, tokens, args, soft):
    """a weaker version of find. the input is not info_type but a list of tokens."""
    length_p = len(ref_p['lemmas'])
    p_raw_lemma = ' '.join(ref_p['lemmas'])
    p_raw_lower = ' '.join(ref_p['tokens'])
    to_return = []
    sentence_lemma = tokens['lemmas']
    sentence_lower = [token.lower() for token in tokens['tokens']]
    for idx in range(len(sentence_lemma) - length_p + 1):
        p_new_lemma = ' '.join(sentence_lemma[idx: idx + length_p])
        p_new_lower = ' '.join(sentence_lower[idx: idx + length_p])
        if softened_string_eq(p_raw_lemma, p_new_lemma) or softened_string_eq(p_raw_lower, p_new_lower):
            to_return.append(Occurrence(
                span = p_new_lemma,
                begin_idx = idx,
                end_idx = idx + length_p - 1,
                location = 'unknown',
                sentence_idx = -1,
                offset = -1
            ))
    # if soft:
    # run the find module to get an extended list
    return to_return

# ================== compare ================
def geq(d0, d1, soft): # d0 >= d1
    # soft = False
    if d0 >= d1:
        return 1.0
    elif soft:
        # print(d0, d1)
        # print(max(1.0 - float((d1 - d0) ** 2) / (4.0 * (abs(d0) + 1) ** 2), 0))
        return max(1.0 - float((d1 - d0) ** 2) / (2.0 * (abs(d0) + 1) ** 2), 0)
    else:
        return 0.0

def leq(d1, d0, soft): # d1 <= d0
    # soft = False
    if d1 <= d0:
        return 1.0
    elif soft:
        # print(d0, d1)
        return max(1.0 - float((d1 - d0) ** 2) / (2.0 * (abs(d0) + 1) ** 2), 0)
    else:
        return 0.0

def eq(d0, d1, soft):
    """a == b -> a <= b and a >= b."""
    return _and([geq(d0, d1, soft), leq(d0, d1, soft)], soft)

def distance(o1: Occurrence, o2: Occurrence, soft=False):
    """returns the abs distance between o1 and o2.
    not sure why we need a soft flag. is this even needed at all?"""
    # when o1 and o2 are not even in the same sentence, we return MAXINT
    if o1.location != o2.location or o1.sentence_idx != o2.sentence_idx:
        return MAXINT

    st1, ed1 = o1.begin_idx, o1.end_idx
    st2, ed2 = o2.begin_idx, o2.end_idx
    if st1 > ed2:
        dist = st1 - ed2 - 1
    elif st2 > ed1:
        dist = st2 - ed1 - 1
    else:
        dist = MAXINT
    return dist

# =============== logics ===============
def to_tensor(lst):
    ret = []
    for item in lst:
        if not isinstance(item, torch.Tensor):
            ret.append(torch.tensor(item, dtype=torch.float))
        else:
            ret.append(item.cpu())
    return torch.stack(ret)

def _and(lst, soft):
    """lst is a list of probabilities"""
    lst = to_tensor(lst)
    if soft:
        if len(lst) > 0:
            return torch.max(torch.sum(lst) - (len(lst) - 1), torch.tensor(0.0))
        else:
            return 1.0
        # return sum(lst) - (len(lst) - 1)
    else:
        return float(all(np.equal(1, item) for item in lst))

def _or(lst, soft):
    """not sure if _or is used ..."""
    lst = to_tensor(lst)
    if soft:

        if len(lst) > 0:
            return torch.max(lst)
        else:
            return 1.0
    else:
        return float(any(np.equal(1, item) for item in lst))


def _not(p, soft):
    if soft:
        return 1 - p
    else:
        return float(not np.equal(1, p))

# ============= basics ==================
def merge(x, y):
    if type(x) != tuple:
        x = (x,)
    if type(y) != tuple:
        y = (y,)
    return x + y

# =============== compositional ===============
# Wrappers that re-uses the modules above.
# Links the semantic parsing result and actual LFs.

def _in(x,y,inputs):
    return (x in y)

def _in2(x,y,inputs):
    #print("inputs:", inputs)
    _x = inputs[x]
    _y = inputs['instance'].context_info['tokens']
    #print("x:", _x)
    #print("y:", _y)
    return (_x in _y)

def _in1(funcs, y, inputs):
    """funcs(inputs) is true in the context of y."""
    soft = inputs['soft']
    funcs = tuple_fy(funcs)
    ret = []
    inputs_new = inputs

    for func in funcs:
        print("func:", func)
        one_ret = func(inputs_new)
        if not isinstance(one_ret, bool) and not isinstance(one_ret, float) and not isinstance(one_ret, torch.Tensor):
            raise TypeError('Not executable')
        ret.append(one_ret)
    return _and(ret, soft)

def _is(funcs, ws, inputs):
    soft = inputs['soft']
    funcs = tuple_fy(funcs)
    ws = tuple_fy(ws)
    ret = []
    for func in funcs:
        for w in ws:
            one_ret = func(w)(inputs)
            #print("results:", one_ret)
            if not isinstance(one_ret, bool) and not isinstance(one_ret, float) and not isinstance(one_ret, torch.Tensor):
                raise TypeError('Not executable')
            ret.append(one_ret)
    return _and(ret, soft)

def _is0(y, x, inputs):
    soft = inputs['soft']
    y = tuple_fy(y)
    x = tuple_fy(x)
    ret = []
    for y0 in y:
        for x0 in x:
            #print("x0y0:",x0, y0)
            if isinstance(inputs[x0], Occurrence):
                if SPECIAL_CHARS.__contains__(inputs[x0].span):
                    xx = SPECIAL_CHARS[inputs[x0].span]
                else:
                    xx = inputs[x0].span
            else:
                if SPECIAL_CHARS.__contains__(inputs[x0]):
                    xx = SPECIAL_CHARS[inputs[x0]]
                else:
                    xx = inputs[x0]
            #print("xx:", xx)
            if xx == y0:
                one_ret = 1.0
            else:
                one_ret = 0.0
            if not isinstance(one_ret, bool) and not isinstance(one_ret, float) and not isinstance(one_ret, torch.Tensor):
                raise TypeError('Not executable')
            ret.append(one_ret)
    return _and(ret, soft)

# ============ ner & chunk =============
def ner(ner_type, w, inputs):
    """check if w is 'ner_type' (e.g. a person) in the input sentence.
    Turn off lemmatize, otherwise the chunk cannot be found."""
    soft = inputs['soft']
    c, w = update_inputs(inputs, ['Context', w], lemma=False)
    tags = get_tags(w, c['ner'])
    return float(len(set(tags).intersection(set(NER_DICT[ner_type]))) > 0)

def chunk(chunk_type, w, inputs):
    """check if w is 'chunk_type' (e.g. a noun phrase) in the input sentence.
    Turn off lemmatize, otherwise the chunk cannot be found."""
    soft = inputs['soft']
    c, w = update_inputs(inputs, ['Context', w], lemma=False)
    tags = get_tags(w, c['constituency'])
    # print("tags:", tags)
    return float(len(set(tags).intersection(set(CHUNK_DICT[chunk_type]))) > 0)

def lexicon(lexicon_type, w, inputs):
    """check if w is 'lexicon_type' (e.g. a noun phrase) in the input sentence.
    Turn off lemmatize, otherwise the lexicon cannot be found."""
    soft = inputs['soft']
    c, w = update_inputs(inputs, ['Context', w], lemma=False)
    tags = get_tags(w, c['lexicon'])
    return float(len(set(tags).intersection(set(LEXICON_DICT[lexicon_type]))) > 0)

# ============ location (left, right, between) ==============
def left(args, key, w, inputs):
    args.update({'direction': 'left'})
    return left_or_right(args, key, w, inputs)

def right(args, key, w, inputs):
    args.update({'direction': 'right'})
    return left_or_right(args, key, w, inputs)

def left_or_right(args, key, w, inputs):
    """w is in the left/right of key,
    args['direction'] specify the direction
    args['dist'] specify the distance within."""
    soft = inputs['soft']
    s, w, key = update_inputs(inputs, ['Context', w, key])
    ref_q = inputs['reference']
    pretrained_modules = inputs['pretrained_modules']
    #print("key:", key)
    occurrences_of_key = find(ref_q, key, s, pretrained_modules, soft)
    #print("occurence_of_key:", occurrences_of_key)
    #print("w:", w)
    occurrences_of_w = find(ref_q, w, s, pretrained_modules, soft)
    #print("occurence_of_w:", occurrences_of_key)
    max_prob = torch.tensor(0.0)
    for okey in occurrences_of_key:
        for ow in occurrences_of_w:
            if args['direction'] == 'left':
                dist = okey.begin_idx - ow.end_idx - 1
            else:
                dist = ow.begin_idx - okey.end_idx - 1
            #print("dist:", dist)
            #print("w:", ow.span, ow.begin_idx, ow.end_idx)
            #print("key:", okey.span, okey.begin_idx, okey.end_idx)
            current_prob = geq(dist, 0, soft)
            if 'dist' in args:
                #print("request_dist:", args['dist'])
                prob1 = leq(dist, args['dist'], soft)
                current_prob = _and([current_prob, prob1, okey.confidence, ow.confidence], soft)
            if isinstance(current_prob, float):
                current_prob = torch.tensor(current_prob)
            max_prob = torch.max(max_prob, current_prob)
    return max_prob

def dep_dist(args, key, w, inputs):
    soft = inputs['soft']
    s, w, key = update_inputs(inputs, ['Context', w, key])
    ref_q = inputs['reference']
    pretrained_modules = inputs['pretrained_modules']
    # print("key:", key)
    occurrences_of_key = find(ref_q, key, s, pretrained_modules, soft)
    # print("occurence_of_key:", occurrences_of_key)
    # print("w:", w)
    occurrences_of_w = find(ref_q, w, s, pretrained_modules, soft)
    # print("occurence_of_w:", occurrences_of_key)
    max_prob = torch.tensor(0.0)
    for okey in occurrences_of_key:
        for ow in occurrences_of_w:
            # print("w:", ow.span, ow.begin_idx, ow.end_idx)
            # print("key:", okey.span, okey.begin_idx, okey.end_idx)
            inputs_copy = copy.copy(inputs)
            dist = get_dependency_distance(okey, ow, inputs_copy)
            # print("dist:", dist)
            current_prob = geq(dist, 0, soft)
            if 'dist' in args:
                # print("request_dist:", args['dist'])
                prob1 = leq(dist, args['dist'], soft)
                current_prob = _and([current_prob, prob1, okey.confidence, ow.confidence], soft)
            if isinstance(current_prob, float):
                current_prob = torch.tensor(current_prob)
            max_prob = torch.max(max_prob, current_prob)
    return max_prob

def between(args, keys, w, inputs):
    """w is between key1 and key2,
    args['tight'], sandwich"""
    soft = inputs['soft']
    key1, key2 = keys
    s, w, key1, key2 = update_inputs(inputs, ['Context', w, key1, key2])
    ref_q = inputs['reference']
    pretrained_modules = inputs['pretrained_modules']
    occurrences_of_key1 = find(ref_q, key1, s, pretrained_modules, soft)
    occurrences_of_key2 = find(ref_q, key2, s, pretrained_modules, soft)
    occurrences_of_w = find(ref_q, w, s, pretrained_modules, soft)
    max_prob = torch.tensor(0.0)
    for okey1 in occurrences_of_key1:
        for okey2 in occurrences_of_key2:
            for ow in occurrences_of_w:
                d0 = distance(okey1, ow) + distance(okey2, ow) + ow.length
                d1 = distance(okey1, okey2)
                current_prob = _and([okey1.confidence, okey2.confidence, eq(d0, d1, soft)], soft)
                if 'tight' in args:
                    d2 = distance(okey1, ow)
                    d3 = distance(okey2, ow)
                    current_prob = _and([eq(d2, 0, soft), eq(d3, 0, soft), current_prob], soft)
                if isinstance(current_prob, float):
                    current_prob = torch.tensor(current_prob)
                max_prob = torch.max(max_prob, current_prob)
    return max_prob

# ============ start with or end with ===========
def starts_with(key1, key2, inputs):
    return float(base_with(key1, key2, inputs, st=True))

def ends_with(key1, key2, inputs):
    return float(base_with(key1, key2, inputs, st=False))

def base_with(key1, key2, inputs, st=True):
    key1, key2 = update_inputs(inputs, [key1, key2])
    if isinstance(key1, dict):
        key1 = ' '.join(key1['lemmas']).lower()
    else:
        key1 = ' '.join(key1).lower()
    key2 = ' '.join(key2['lemmas']).lower()
    return key1.startswith(key2) if st else key1.endswith(key2)

# ============ deprecated ==============
def get_st_ed0(score):
    st, ed = -1, -1
    flag1 = True
    flag2 = True
    for idx, item in enumerate(score):
        if flag1 and item == 1:
            st = idx
            flag1 = False
        if flag2 and not flag1 and item == 0:
            ed = idx - 1
            flag2 = False
            break
    if flag2:
        ed = len(score)
    return st, ed

def sent_strip(sent):
    return sent.rstrip('.').rstrip('?').rstrip('!')

def compare(dist, scalar, n, hard):
    prob = 0
    if hard:
        idx = scalar + n
        prob = np.sum(dist[:idx])
    return prob

def distance_counting(score1, score2, n, hard):
    ret = [0] * (2 * n)
    if hard:
        # distance can be -n to n
        st1, ed1 = get_st_ed(score1)
        st2, ed2 = get_st_ed(score2)
        if st1 > ed2:
            dist = ed2 - st1 - 1
            ret[n + dist] = 1
        elif st2 > ed1:
            dist = st2 - ed1 - 1
            ret[n + dist] = 1
        else:
            raise Exception
    return ret

def string_matching(query, sentence, hard):
    query = query.split(" ")
    sentence = sent_strip(sentence)
    sentence = sentence.split(" ")
    if hard:
        for start in range(len(sentence) - len(query) + 1):
            flag = True
            for j in range(len(query)):
                if query[j] != sentence[start + j]:
                    flag = False
                    break
            if flag:
                end = start + len(query)
                ret = [1 if start <= idx < end else 0 for idx in range(len(sentence))]
                return ret
    return [0 for _ in range(len(sentence))]



if __name__ == "__main__":
    sentence = "How is packet switching characterized?"
    query1 = "packet switching"
    ret1 = string_matching(query1, sentence, hard=True)
    assert ret1 == [0, 0, 1, 1, 0]

    sentence = "How is packet switching characterized?"
    query2 = "characterized"
    ret2 = string_matching(query2, sentence, hard=True)
    assert ret2 == [0, 0, 0, 0, 1]

    ret3 = distance_counting(ret1, ret2, len(sentence.split(" ")), hard=True)
    assert ret3 == [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]

    ret4 = compare(ret3, 3, len(sentence.split(" ")), hard=True)
    print(ret4)
