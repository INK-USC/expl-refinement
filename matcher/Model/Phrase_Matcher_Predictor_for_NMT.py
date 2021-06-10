from Model.Phrase_Matcher_Predictor import Phrase_Matcher_Predictor
from spacy.tokens import Doc
import spacy
import torch

class _WhitespaceSpacyTokenizer:
    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, words):
        # words = text.split(" ")
        spaces = [True] * len(words)
        return Doc(self.vocab, words=words, spaces=spaces)

def padding(sents):
    max_len = max([len(r) for r in sents])
    if max_len > 500:
        print('Exceeding max length ...')
        raise IndexError
    return [r + ['[PAD]'] * (max_len - len(r)) for r in sents]

def get_aligned(span, alignment):
    try:
        return [alignment[span[0]][0], alignment[span[1] - 1][-1] + 1]
    except IndexError:
        # Some tokens don't have word alignment?
        # spans = [(0, 13), (0, 6), (0, 4), (0, 1), (7, 8), (9, 10)]
        # alignment = [[1], [2], [3, 4, 5, 6, 7], [8], [9], *[]*, [10], [11, 12, 13, 14], [15], [16, 17, 18, 19, 20], [21], [22], [23]]
        # tokens = ['°', 'N', '1.91667', '°', 'E', '\ufeff', '/', '50.98500', ';', '1.91667', ')', ',', 'and']
        return None

class Phrase_Matcher_Predictor_NMT(Phrase_Matcher_Predictor):
    def __init__(self, batch_size):
        super(Phrase_Matcher_Predictor_NMT, self).__init__(batch_size)
        self.spacy = spacy.load("en_trf_bertbaseuncased_lg", disable=['tagger', 'parser', 'ner'])
        self.spacy.tokenizer = _WhitespaceSpacyTokenizer(self.spacy.vocab)
        self.spacy.remove_pipe("trf_tok2vec")

    def add_cache(self, instance, require_grad=False):
        self.cached_target_sentence_original = [instance.context_info['tokens']]
        self.target_sentences = [self.spacy(sent) for sent in self.cached_target_sentence_original]
        self.target_sentence_alignment = [sent._.trf_alignment for sent in self.target_sentences]
        self.target_sentence_word_pieces = [sent._.trf_word_pieces_ for sent in self.target_sentences]
        '''
        query_sentence = instance.question_info['tokens']
        self.query_sentence = self.spacy(query_sentence)
        self.query_sentence_alignment = self.query_sentence._.trf_alignment
        self.query_sentence_word_pieces = self.query_sentence._.trf_word_pieces_
        self.cached_query_sentence_original = query_sentence
        '''
        padded = padding(self.target_sentence_word_pieces)

        gradient_context = torch.enable_grad() if require_grad else torch.no_grad()
        with gradient_context:
            encodings = self.model_.bert(padded)

        #self.query_sentence_encoding = encodings[0, :, :]
        self.target_sentence_encoding = encodings[0:, :, :]

        # self.query_sentence_encoding = self.model_.bert([self.query_sentence_word_pieces])[0, :, :]
        # self.target_sentence_encoding = self.model_.bert(padding(self.target_sentence_word_pieces))
    '''
    def add_cache_nq(self, question_dict, context_dict, require_grad=False):
        self.cached_target_sentence_original = [sentence['tokens'] for sentence in context_dict]
        self.target_sentences = [self.spacy(sent) for sent in self.cached_target_sentence_original]
        self.target_sentence_alignment = [sent._.trf_alignment for sent in self.target_sentences]
        self.target_sentence_word_pieces = [sent._.trf_word_pieces_ for sent in self.target_sentences]

        query_sentence = question_dict['tokens']
        self.query_sentence = self.spacy(query_sentence)
        self.query_sentence_alignment = self.query_sentence._.trf_alignment
        self.query_sentence_word_pieces = self.query_sentence._.trf_word_pieces_
        self.cached_query_sentence_original = query_sentence

        padded = padding([self.query_sentence_word_pieces] + self.target_sentence_word_pieces)
        gradient_context = torch.enable_grad() if require_grad else torch.no_grad()
        with gradient_context:
            encodings = self.model_.bert(padded)

        self.query_sentence_encoding = encodings[0, :, :]
        self.target_sentence_encoding = encodings[1:, :, :]
    '''
    def clear_cache(self):
        """release cached bert encoded context"""
        del self.target_sentence_encoding
        #del self.query_sentence_encoding

        self.target_sentence = None
        self.target_sentence_alignment = None
        self.target_sentence_word_pieces = None
        self.cached_target_sentence_original = None
        '''
        self.query_sentence = None
        self.query_sentence_alignment = None
        self.query_sentence_word_pieces = None
        self.cached_query_sentence_original = None
        '''
    def predict_find_custom(self, idx, ref_q, query_idx, target_idxs, topn=None, require_grad=False):
        if len(target_idxs) == 0:
            return []

        # preprocess query sentence
        query_sentence = ref_q['tokens']
        self.query_sentence = self.spacy(query_sentence)
        self.query_sentence_alignment = self.query_sentence._.trf_alignment
        self.query_sentence_word_pieces = self.query_sentence._.trf_word_pieces_
        self.cached_query_sentence_original = query_sentence
        padded = padding([self.query_sentence_word_pieces])
        gradient_context = torch.enable_grad() if require_grad else torch.no_grad()
        with gradient_context:
            encodings = self.model_.bert(padded)
        self.query_sentence_encoding = encodings[0, :, :]

        # align piecewise tokenization
        aligned_query_idx = get_aligned(query_idx, self.query_sentence_alignment)

        aligned_target_idxs = [get_aligned(target_idx, self.target_sentence_alignment[idx])
                                 for target_idx in target_idxs]
        aligned_target_idxs = list(filter(None, aligned_target_idxs))

        if len(aligned_target_idxs) == 0:
            return []

        gradient_context = torch.enable_grad() if require_grad else torch.no_grad()
        with gradient_context:
            query_span = self.model_.encode_phrases_for_nmt(
                sent_encoding=self.query_sentence_encoding,
                spans=[aligned_query_idx]
            )

            target_spans = self.model_.encode_phrases_for_nmt(
                sent_encoding=self.target_sentence_encoding[idx],
                spans=aligned_target_idxs
            )

            query_span = query_span.expand_as(target_spans)
            scores = self.model_.predict(query_span, target_spans)
            scores = scores * 4 - 3 # scaling
            span_scores = sorted(zip(target_idxs, scores), key=lambda x: x[1], reverse=True)

        if topn:
            span_scores = span_scores[:topn]
        return span_scores

    def add_cache_fill(self, ref_sentence, new_sentence, require_grad=False):
        self.ref_sentence = self.spacy(ref_sentence['tokens'])
        self.ref_sentence_alignment = self.ref_sentence._.trf_alignment
        self.ref_sentence_word_pieces = self.ref_sentence._.trf_word_pieces_
        self.cached_ref_sentence_original = ref_sentence

        self.new_sentence = self.spacy(new_sentence['tokens'])
        self.new_sentence_alignment = self.new_sentence._.trf_alignment
        self.new_sentence_word_pieces = self.new_sentence._.trf_word_pieces_
        self.cached_new_sentence_original = new_sentence

        padded = padding([self.ref_sentence_word_pieces, self.new_sentence_word_pieces])
        gradient_context = torch.enable_grad() if require_grad else torch.no_grad()
        with gradient_context:
            encodings = self.model_.bert(padded)

        self.ref_sentence_encoding = encodings[0, :, :]
        self.new_sentence_encoding = encodings[1, :, :]

    def clear_cache_fill(self):
        del self.ref_sentence_encoding
        del self.new_sentence_encoding

        self.ref_sentence = None
        self.ref_sentence_alignment = None
        self.ref_sentence_word_pieces = None
        self.cached_ref_sentence_original = None

        self.new_sentence = None
        self.new_sentence_alignment = None
        self.new_sentence_word_pieces = None
        self.cached_new_sentence_original = None

    def predict_fill_custom(self, query_sent, target_sent, query_idx, target_idxs, topn=None, require_grad=False):
        if len(target_idxs) == 0:
            return []

        self.add_cache_fill(query_sent, target_sent)

        # align piecewise tokenization
        aligned_ref_idx = get_aligned(query_idx, self.ref_sentence_alignment)
        aligned_target_idxs = [get_aligned(target_idx, self.new_sentence_alignment)
                             for target_idx in target_idxs]
        aligned_target_idxs = list(filter(None, aligned_target_idxs))

        if len(aligned_target_idxs) == 0:
            return []

        gradient_context = torch.enable_grad() if require_grad else torch.no_grad()
        with gradient_context:
            query_span = self.model_.encode_phrases_for_nmt(
                sent_encoding=self.ref_sentence_encoding,
                spans=[aligned_ref_idx]
            )

            target_spans = self.model_.encode_phrases_for_nmt(
                sent_encoding=self.new_sentence_encoding,
                spans=aligned_target_idxs
            )

            query_span = query_span.expand_as(target_spans)
            scores = self.model_.predict(query_span, target_spans)
            span_scores = sorted(zip(target_idxs, scores), key=lambda x: x[1], reverse=True)

        self.clear_cache_fill()

        if topn:
            span_scores = span_scores[:topn]
        return span_scores