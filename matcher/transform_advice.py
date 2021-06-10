"""
Transform sentence and advice span from spacy tokens to BERT tokens
"""
import argparse
import ast
import logging
import spacy
import string
import time

logger = logging.getLogger(__name__)
printable = set(string.printable)

def transform_advice(sentence, advice, max_seq_length, nlp, cls_token, sep_token):
    def get_mapping(toks):
        mapping = {}
        bert_id = 0
        for i, tok in enumerate(toks):
            if tok != cls_token and tok != sep_token:
                mapping[i] = bert_id
                bert_id += 1
        return mapping

    def get_bert_id(span):
        st, ed = span
        try:
            st, ed = spacy_alignment[st][0], spacy_alignment[ed][-1]
        except:
            logger.warning('List index out of range')
            print('st', st, 'ed', ed, 'spacy_alignment', len(spacy_alignment))
            return None
        return (spacy_to_hf[st], spacy_to_hf[ed])
    
    def transform_single(single):
        single = [get_bert_id(e) for e in single]
        single = [e for e in filter(None, single)]
        return single

    def transform_pair(pair):
        res = []
        for p, q in pair:
            p, q = get_bert_id(p), get_bert_id(q)
            if p is not None and q is not None:
                res.append((p, q))
        return res
    
    doc = nlp(sentence)

    advice = ''.join(filter(lambda x: x in printable, advice))

    try:
        advice_literal = ast.literal_eval(advice)
    except:
        advice_literal = [[], [], [], [], [], [], [], []]
        logger.warning('Advice literal error')
        print(sentence)
        print(repr(advice))

    single_inc_0, single_dec_0, pair_inc_0, pair_dec_0, \
            single_inc_1, single_dec_1, pair_inc_1, pair_dec_1 = advice_literal
    
    spacy_to_hf = get_mapping(doc._.trf_word_pieces_)
    spacy_alignment = doc._.trf_alignment
    space_ids = []
    for i, tok in enumerate(doc):
        if tok.is_space:
            space_ids.append(i)
    
    spacy_alignment = [e for i, e in enumerate(spacy_alignment) if i not in space_ids]

    # transform span index from spacy to huggingface BERT
    single_inc_0 = transform_single(single_inc_0)
    single_dec_0 = transform_single(single_dec_0)
    single_inc_1 = transform_single(single_inc_1)
    single_dec_1 = transform_single(single_dec_1)
    pair_inc_0 = transform_pair(pair_inc_0)
    pair_dec_0 = transform_pair(pair_dec_0)
    pair_inc_1 = transform_pair(pair_inc_1)
    pair_dec_1 = transform_pair(pair_dec_1)

    bert_advice = [single_inc_0, single_dec_0, pair_inc_0, pair_dec_0,
                    single_inc_1, single_dec_1, pair_inc_1, pair_dec_1]
    
    return str(bert_advice)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path',
                        default=None, type=str, required=True, help='Advice file path')
    parser.add_argument('--labeled', action='store_true')
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument('--base', choices=['bert', 'roberta'], default='bert')
    args = parser.parse_args()

    if args.base == 'bert':
        nlp = spacy.load("en_trf_bertbaseuncased_lg")
        cls_token = '[CLS]'
        sep_token = '[SEP]'
    elif args.base == 'roberta':
        nlp = spacy.load("en_trf_robertabase_lg")
        cls_token = '<s>'
        sep_token = '</s>'
    else:
        raise NotImplementedError
    
    with open(args.path, encoding='utf-8') as f:
        advice_lines = f.readlines()
    
    result = []

    start_time = time.time()

    for i, line in enumerate(advice_lines):
        if args.labeled:
            sentence, advice, label, score = line.strip('\n').split('\t')
        else:
            sentence, advice = line.split('\t')
        
        advice = transform_advice(sentence, advice, args.max_seq_length, nlp, cls_token, sep_token)

        if args.labeled:
            transformed_line = sentence + '\t' + advice + '\t' + label + '\t' + score + '\n'
        else:
            transformed_line = sentence + '\t' + advice + '\n'
        
        result.append(transformed_line)

    end_time = time.time()
    print('Elapsed time in seconds:', end_time - start_time)
    
    with open(args.path + '_' + args.base, 'w', encoding='utf-8') as f:
        for line in result:
            f.write(line)

if __name__ == '__main__':
    main()