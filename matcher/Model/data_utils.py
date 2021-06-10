import numpy as np

def padding(sents):
    max_len = max([len(r) for r in sents])
    return [r+['[PAD]']*(max_len-len(r)) for r in sents]

# for FIND module
# dev & test: sents that all query phrases have at least one non-exact matched target phrases
# train: rest of the data
def split_data(data, train_mode):
    train, test = [], []
    for d in data:
        if_all_em = []
        for i in range(len(d[1])):
            query_phrase = d[0][d[1][i][0]:d[1][i][1]]
            if train_mode == 'sent':
                all_target_spans_flat = [[i, r[0], r[1]] for i in range(len(d[3])) for r in d[3][i]]
                labels_flat_i = [r for rr in d[4][i] for r in rr]
                target_spans = [all_target_spans_flat[j] for j in range(len(all_target_spans_flat)) if labels_flat_i[j]]
            else:
                target_spans = [d[3][j] for j in range(len(d[3])) if d[4][i][j]]
            all_em = True
            for s in target_spans:
                if train_mode == 'sent':
                    target_phrase = d[2][s[0]][s[1]:s[2]]
                else:
                    target_phrase = d[2][s[0]:s[1]]
                if query_phrase != target_phrase:
                    all_em = False 
                    break
            
            if_all_em.append(all_em)
        
        if not any(if_all_em):
            test.append(d)
        else:
            train.append(d)
    
    return train, test

# for FIND module
# filter out exact matches
def filter_em_data(data):
    non_em_data = []
    for d in data:
        query_sent, query_spans, target_sent, target_spans, labels, types, sources = d
        query_spans = [query_spans[i] for i in range(len(query_spans)) if sources[i]!='exact match']
        labels = [labels[i] for i in range(len(labels)) if sources[i]!='exact match']
        sources = [r for r in sources if r != 'exact match']
        non_em_data.append([query_sent, query_spans, target_sent, target_spans, labels, types, sources])
    
    return non_em_data

# for training data: keep a ratio of neg to pos spans
def neg_pos_rebalance(data, neg_pos_ratio, train_mode):
    result = []
    for d in data:
        if train_mode == 'phrase':
            query_spans, target_spans, labels = d
        else:
            query_sent, query_spans, target_sent, target_spans, labels, types, sources = d
        true_spans_idx = list(set(np.where(np.array(labels))[1]))
        neg_spans_idx = [r for r in range(len(target_spans)) if not r in true_spans_idx]
        n_neg_spans = int(neg_pos_ratio * len(true_spans_idx)) if train_mode != 'phrase' else 30
        if len(neg_spans_idx) > n_neg_spans:
            selected_neg_spans_idx = list(np.random.choice(neg_spans_idx, n_neg_spans, replace=False))
            all_spans_idx = true_spans_idx + selected_neg_spans_idx
            target_spans = [target_spans[i] for i in range(len(target_spans)) if i in all_spans_idx]
            labels = [[l[i] for i in range(len(l)) if i in all_spans_idx] for l in labels]
            if train_mode != 'phrase':
                types = [types[i] for i in range(len(types)) if i in all_spans_idx]
        
        if train_mode == 'phrase':
            result.append([query_spans, target_spans, labels])
        else:
            result.append([query_sent, query_spans, target_sent, target_spans, labels, types, sources])
    
    return result

# process training data for fill module
def process_train_enum_fill(data, neg_pos_ratio, batch_size, diff_threshold = 10):
    np.random.seed(0)
    processed = {}
    all_q = [r for rr in data['q2q_matching'] for r in rr]
    all_q = [list(r) for r in set([tuple(r) for r in all_q])]
    processed['q_batchs'] = []
    for i in range(int(np.ceil(len(all_q)/batch_size))):
        curr_batch = {}
        batch_all_q = all_q[(batch_size*i):(batch_size*(i+1))]
        q2idx = {k:v for v,k in enumerate(tuple(q) for q in batch_all_q)}
        q_groups = [[i for i in range(len(data['q2q_matching'])) if r in data['q2q_matching'][i]] for r in batch_all_q]
        max_q_len = max([len(r) for r in batch_all_q])
        curr_batch['q2q_matching'] = []
        for i in range(len(batch_all_q)-1):
            for j in range(i+1, len(batch_all_q)):
                if set(q_groups[i]).intersection(q_groups[j]):
                    curr_batch['q2q_matching'].append([i, j, 1])
                else:
                    curr_batch['q2q_matching'].append([i, j, 0])
        
        curr_batch['q2q_phrase_matching'] = []
        for d in data['q2q_phrase_matching']:
            d = [r for r in d if r[0] in batch_all_q]
            d = d[:200] # avoid having too many pairs
            for i in range(len(d)):
                for j in range(len(d)):
                    q1, spans1 = d[i]
                    q2, spans2 = d[j]
                    for k in range(len(spans1)):    
                        len_span1_k = spans1[k][1]-spans1[k][0]
                        max_len_span2 = len_span1_k + diff_threshold
                        all_spans2 = [[i,j] for i in range(len(q2)-1) for j in range(i+1,len(q2)) if j-i <= max_len_span2]
                        if neg_pos_ratio:
                            selected_all_spans2_idx = np.random.choice([i for i in range(len(all_spans2)) if all_spans2[i] != spans2[k]], int(neg_pos_ratio))
                            all_spans2 = [all_spans2[i] for i in selected_all_spans2_idx] + [spans2[k]]
                        else:
                            if not spans2[k] in all_spans2:
                                all_spans2.append(spans2[k])
                        
                        all_labels = [1 if r == spans2[k] else 0 for r in all_spans2]
                        curr_batch['q2q_phrase_matching'].append([q2idx[tuple(q1)], spans1[k], q2idx[tuple(q2)], all_spans2, all_labels])
        
        batch_all_q = [r + ['[PAD]']*(max_q_len-len(r)) for r in batch_all_q]
        curr_batch['all_q'] = batch_all_q
        processed['q_batchs'].append(curr_batch)
    
    all_c = [r[0] for rr in data['c2c_answer_matching'] for r in rr]
    all_c = [list(r) for r in set([tuple(r) for r in all_c])]
    processed['c_batchs'] = []
    for i in range(int(np.ceil(len(all_c)/batch_size))):
        curr_batch = {}
        batch_all_c = all_c[(batch_size*i):(batch_size*(i+1))]
        c2idx = {k:v for v,k in enumerate(tuple(c) for c in batch_all_c)}
        max_c_len = max([len(r) for r in batch_all_c])
        curr_batch['c2c_answer_matching'] = []
        for d in data['c2c_answer_matching']:
            d = [r for r in d if r[0] in batch_all_c]
            d = d[:200]
            for i in range(len(d)):
                for j in range(len(d)):
                    c1, span1 = d[i]
                    c2, span2 = d[j]
                    len_span1 = span1[1]-span1[0]
                    max_len_span2 = len_span1 + diff_threshold
                    all_spans2 = [[i,j] for i in range(len(c2)-1) for j in range(i+1,len(c2)) if j-i <= max_len_span2]
                    if neg_pos_ratio:
                        selected_all_spans2_idx = np.random.choice([i for i in range(len(all_spans2)) if all_spans2[i] != span2], int(neg_pos_ratio))
                        all_spans2 = [all_spans2[i] for i in selected_all_spans2_idx] + [span2]
                    else:
                        if not span2 in all_spans2:
                            all_spans2.append(span2)
                    
                    all_labels = [1 if r == span2 else 0 for r in all_spans2]
                    curr_batch['c2c_answer_matching'].append([c2idx[tuple(c1)], span1, c2idx[tuple(c2)], all_spans2, all_labels])
        
        batch_all_c = [r + ['[PAD]']*(max_c_len-len(r)) for r in batch_all_c]
        curr_batch['all_c'] = batch_all_c
        processed['c_batchs'].append(curr_batch)
    
    return processed


def process_single_train_enum_fill(sent1, span1, sent2, diff_threshold = 10):
    all_sents = [sent1, sent2]
    max_q_len = max([len(r) for r in all_sents])
    all_sents = [r + ['[PAD]']*(max_q_len-len(r)) for r in all_sents]
    len_span1 = span1[1]-span1[0]
    max_len_span2 = len_span1 + diff_threshold
    all_spans2 = [[i,j] for i in range(len(sent2)-1) for j in range(i+1,len(sent2)) if j-i <= max_len_span2]
    phrase_matching = [[0,span1,1,all_spans2]]
    sent_matching = [[0,1]]
    return all_sents, phrase_matching, sent_matching


# process training data for find module
def process_train_enum_find(data, neg_pos_ratio, text_len_threshold = 150, query_len_threshold = 6, diff_threshold = 5):
    processed = []
    for query_sentence, query_span, target_sentence, target_spans in data:
        if len(target_sentence) > text_len_threshold:
            continue
        
        query_len = query_span[1]-query_span[0]
        target_spans = [r for r in target_spans if (r[1]-r[0])-(query_len) <= diff_threshold]
        if query_len > query_len_threshold or not target_spans:
            continue
        
        sentence_boundaries, curr_start = [], 0
        for i in range(len(target_sentence)):
            if target_sentence[i] == '[SEP]':
                sentence_boundaries.append([curr_start, i+1])
                curr_start = i+1
        
        target_max_len = query_len + diff_threshold
        all_target_spans = [[i,j] for r in sentence_boundaries for i in range(r[0],r[1]) for j in range(i+1,min(i+target_max_len+1, r[1]+1))]
        if not all([r in all_target_spans for r in target_spans]):
            print("Error in process_train")
        
        labels = [1 if r in target_spans else 0 for r in all_target_spans]
        if neg_pos_ratio:
            pos_count = len(target_spans)
            max_neg_count = int(pos_count * neg_pos_ratio)
            neg_indices = [i for i in range(len(labels)) if labels[i]==0]
            if len(neg_indices) > max_neg_count:
                keep_neg_indices = np.random.choice(neg_indices, max_neg_count, replace=False)
                all_target_spans = [all_target_spans[i] for i in range(len(all_target_spans)) if i in keep_neg_indices or labels[i]==1]
                labels = [labels[i] for i in range(len(labels)) if i in keep_neg_indices or labels[i]==1]
        
        processed.append([query_sentence, query_span, target_sentence, all_target_spans, labels])
    
    return processed


# process training data for phrase matcher
def process_train_seqt(data, batch_size, text_len_threshold = 150, query_len_threshold = 6, diff_threshold = 5):
    processed, curr_batch = [], []
    for query_sentence, query_span, target_sentence, target_spans in data:
        if len(target_sentence) > text_len_threshold:
            continue
        
        query_len = query_span[1]-query_span[0]
        target_spans = [r for r in target_spans if (r[1]-r[0])-(query_len) <= diff_threshold]
        if query_len > query_len_threshold or not target_spans:
            continue
                
        labels = np.zeros(len(target_sentence), dtype=int)
        for s in target_spans:
            labels[s[0]:s[1]] = 1
        
        curr_batch.append([query_sentence, query_span, target_sentence, target_spans, list(labels)])
        if len(curr_batch) == batch_size:
            curr_batch = [[r[i] for r in curr_batch] for i in range(len(curr_batch[0]))]
            processed.append(curr_batch)
            curr_batch = []
    
    if curr_batch:
        curr_batch = [[r[i] for r in curr_batch] for i in range(len(curr_batch[0]))]
        processed.append(curr_batch)
    
    # padding sentences
    for i in range(len(processed)):
        query_sentences = processed[i][0]
        target_sentences = processed[i][2]
        labels = processed[i][4]
        max_len_qs = max([len(r) for r in query_sentences])
        max_len_ts = max([len(r) for r in target_sentences])
        query_sentences = [r + ['[PAD]']*(max_len_qs-len(r)) for r in query_sentences]
        target_sentences = [r + ['[PAD]']*(max_len_ts-len(r)) for r in target_sentences]
        labels = [r + [0]*(max_len_ts-len(r)) for r in labels]
        processed[i][0] = query_sentences
        processed[i][2] = target_sentences
        processed[i][4] = labels
    
    return processed


def find_chunk(sent_ann):
    chunking_hierachy, chunking_strings = chunking_result(sent_ann)
    # find the most top and largest N* chunk
    selected_span = []
    for level in chunking_hierachy:
        N_chunks = [c for c in level if c[0][0]=='N']
        if N_chunks:
            selected_span = sorted(N_chunks, key=lambda x:(x[1][1]-x[1][0]), reverse=True)[0][1]
            break
    
    return selected_span

if __name__ == "__main__":
    pass
    # nlp = StanfordCoreNLP("./stanfordnlp_resources/stanford-corenlp-full-2018-10-05", memory='8g', timeout=15000)
    
    # raw_sentence = "What is the French name of the Canadian Armed Forces?"
    # phrase_span = [2,5] # "the French name"
    # phrase_span2 = []

    # ann = json.loads(nlp.annotate(raw_sentence))

    # assert len(ann['sentences']) == 1
    # sent_ann = ann['sentences'][0]
    # phrase_span = find_chunk(sent_ann)
    # if phrase_span:
        # tokens = [r['originalText'] for r in sent_ann['tokens']]
        # tokens = ['[CLS]'] + tokens + ['[SEP]']
        # phrase_span = [phrase_span[0]+1, phrase_span[1]+1]

