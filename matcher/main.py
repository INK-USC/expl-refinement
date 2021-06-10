import copy
import random
import os
import argparse
from tqdm import tqdm
import pickle
from rule import Rule, AnsFunc, Variable
from parser import Parser, preprocess_sent, tokenize
from read_instance import InstancePreprocess, Instance
import json
import string
import logging
from multiprocessing import Process, Manager
import time
import sys
import numpy as np
import torch.multiprocessing as mp
from functools import reduce
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from Model.Phrase_Matcher_Predictor_for_NMT import Phrase_Matcher_Predictor_NMT

def get_ans_func(row, parser, instance_reader):
    context = row['Text'].replace("``", "\"").replace("''", "\"").replace("<font color=\\\"red\\\"> ", "").replace(" </font>", "").replace("<font color=\"red\">", "").replace("</font>", "")
    label = row["Label"]
    instance = instance_reader.read_one(context, label)

    ansFunc = AnsFunc()
    ansFunc.instantiate(instance)

    vars = []
    Exp = row['Part A'].strip(' ')
    X = row['X']
    Y = row['Y']
    Z = row['Z']
    if len(X) > 0 and X.lower() in context.lower():
        vars.append('X')
        ansFunc.add_variable(Variable.get_variable(instance, X, 'X'))
    if len(Y) > 0 and Y.lower() in context.lower():
        vars.append('Y')
        ansFunc.add_variable(Variable.get_variable(instance, Y, 'Y'))
    if len(Z) > 0 and Z.lower() in context.lower():
        vars.append('Z')
        ansFunc.add_variable(Variable.get_variable(instance, Z, 'Z'))

    version = row['version']
    ansFunc.set_version(version)

    part_b = row['Part B']
    ansFunc.set_advice(part_b)

    #quality_score = float(row['precision'])
    #ansFunc.set_quality(quality_score)

    inputs = {'Label': label,
              'X': X,
              'Y': Y,
              'Z': Z,
              'instance': instance,
              'reference': instance.context_info,
              'pretrained_modules': None,
              'soft': False
              }

    sentences = preprocess_sent(Exp)
    tokenized_sentences = [tokenize(sentence) for sentence in sentences]
    for sentence in tokenized_sentences:
        rule_list_sentence = []
        for potential_sentence in sentence:
            all_possible_parses = parser.parse(potential_sentence)
            if len(all_possible_parses) > 0:
                rule_list_sentence += all_possible_parses
        if len(rule_list_sentence) > 0:
            sorted_rules = sorted(rule_list_sentence, key=lambda x: parser.forward_single(x), reverse=True)
            for a_rule in sorted_rules:
                inputs_copy0 = copy.copy(inputs)
                rule = Rule(inputs_copy0, a_rule)
                inputs_copy = copy.copy(inputs)
                #rule = Rule(a_rule)
                if np.equal(1, rule.execute(inputs_copy)):
                    ansFunc.add_rule(inputs_copy0, a_rule)
                #else:
                #    print("CANNOT MATCH!!!")
                #    print("results:", rule.execute(inputs_copy))
                #    print(inputs['instance'].original_context)
                #    print(a_rule)
    return ansFunc

def match_one_instance(instance, ansFuncs, pretrained_modules, soft, thres):
    # print(type(instance))
    # print(instance.question_info)
    if soft and pretrained_modules:
        if 'find' in pretrained_modules:
            pretrained_modules['find'].add_cache(instance)

    # consensus_answer, (st, ed, confidence) = '', (-1, -1, 0.0)
    ans_for_this_instance = []
    rules_used = []
    for idx, ans_func in ansFuncs:
        answers = ans_func.answer(instance, pretrained_modules, thres=thres, soft=soft)

        # one_answer = answer with confidence 1.0
        # in hard mode one_answer should be the same as answers
        one_answers = [answer for answer in answers if np.equal(answer['confidence'], 1.0)]
        #assert len(answers) == len(one_answers)
        #if len(answers) > 0:
        if len(one_answers) > 0: # do not require only one hard match
            ans_for_this_instance += one_answers
            for i in range(len(one_answers)):
                rules_used.append(idx)

        elif soft and len(answers) > 0: # soft-mode, no hard-match, keep the one answer with max confidence
            answers = sorted(answers, key=lambda x: x['confidence'], reverse=True)
            for answer in answers:
                ans_for_this_instance.append(answer)
                rules_used.append(idx)

    if soft and pretrained_modules:
        if 'find' in pretrained_modules:
            pretrained_modules['find'].clear_cache()

    assert len(ans_for_this_instance) == len(rules_used)
    map_list = []
    for idx, answer in enumerate(ans_for_this_instance):
        map_dict = {}
        idx0, ans_func = ansFuncs[rules_used[idx]]
        map_dict['rule_idx'] = idx0
        map_dict['rule_label'] = ans_func.label
        map_dict['instance_label'] = instance.label
        map_dict['rule_advice'] = ans_func.advice
        map_dict['original_text'] = instance.original_context
        map_dict['instance_chunk'] = instance.context_info['constituency']
        map_dict['confidence'] = answer['confidence']
        map_dict['quality'] = ans_func.score
        for j in ['X', 'Y', 'Z']:
            if answer.__contains__(j):
                #map_dict[j] = (answer[j].span)
                map_dict[j] = (answer[j].span, answer[j].begin_idx, answer[j].end_idx)
        map_list.append(map_dict)
    run_function = lambda x,y: x if y in x else x+[y]
    map_list = reduce(run_function, [[],]+ map_list)
    #if len(ans_for_this_instance) > 0:
    #    consensus_answer, (st, ed, confidence) = get_majority(ans_for_this_instance, soft=0)
    return map_list

def match_multiple_instances(idx0, instances, ansFuncs, return_dict, soft, thres, ngpu):
    start_time = time.time()

    for idx, ans_func in ansFuncs:
        ans_func.reload_rules()

    pretrained_modules = None
    if soft:
        find_predictor = Phrase_Matcher_Predictor_NMT(batch_size=1)
        find_predictor.load_model(idx0%ngpu, module='find')  # args: gpu_id, module
        find_predictor.model_.eval()

        #q2q_predictor = Phrase_Matcher_Predictor_NMT(batch_size=1)
        #q2q_predictor.load_model(idx0 % opt['n_gpu'], module='fill')
        #q2q_predictor.model_.eval()
        # q2c_predictor = Find_Trf(gpu_id = idx0 % opt['n_gpu'])
        # q2c_predictor.eval()
        # d = torch.load('./playground/best_model.pth')
        #
        # q2c_predictor.load_state_dict(d['model'])
        # q2c_predictor.eval()
        # q2c_predictor.to(torch.device("cuda:{}".format(str(idx0 % 2))))

        pretrained_modules = {
            'find': find_predictor,
        }

    to_return = []
    for idx, instance in enumerate(instances):
        if idx % 10 == 0:
            sys.stdout.write(
                "Thread {}, processed {} instances , Time: {} sec\r".format(idx0, idx, time.time() - start_time))
            sys.stdout.flush()
        to_return.append(match_one_instance(instance, ansFuncs, pretrained_modules, soft, thres))

    return_dict[idx0] = to_return

def update_list(result, eight_set):
    rule_set = result['rule_advice']
    single_list = [0, 1, 4, 5]

    for idx, advice_list in enumerate(rule_set):
        for candidate in advice_list:
            if idx in single_list:
                eight_set[idx].append(result[candidate])
            else:
                candidateA, candidateB = candidate
                eight_set[idx].append((result[candidateA], result[candidateB]))
        eight_set[idx] = list(set(eight_set[idx]))

def advice_without_inter(advices):
    advices_without_inter = []
    single_list = [0, 1, 4, 5]
    for advice in advices:
        advice_without_inter = []
        for i in range(len(advice)):
            if i in single_list:
                advice_without_inter.append(advice[i])
            else:
                advice_without_inter.append([])
        advices_without_inter.append(advice_without_inter)
    return advices_without_inter

def get_span_advices(instance_advices):
    instance_span_advices = []
    single_list = [0, 1, 4, 5]
    for advices in instance_advices:
        span_advices = []
        for idx, advice in enumerate(advices):
            span_advice = []
            for i in range(len(advice)):
                if idx in single_list:
                    span, _, _ = advice[i]
                    span_advice.append(span)
                else:
                    A, B = advice[i]
                    spanA, _, _ = A
                    spanB, _, _ = B
                    span_advice.append((spanA, spanB))
            span_advices.append(span_advice)
        instance_span_advices.append(span_advices)
    return instance_span_advices

def get_idx_advices(instance_advices):
    instance_idx_advices = []
    single_list = [0, 1, 4, 5]
    for advices in instance_advices:
        idx_advices = []
        for idx, advice in enumerate(advices):
            idx_advice = []
            for i in range(len(advice)):
                if idx in single_list:
                    _, begin_idx, end_idx = advice[i]
                    idx_advice.append((begin_idx, end_idx))
                else:
                    A, B = advice[i]
                    _, begin_idx_A, end_idx_A = A
                    _, begin_idx_B, end_idx_B = B
                    idx_advice.append(((begin_idx_A, end_idx_A), (begin_idx_B, end_idx_B)))
            idx_advices.append(idx_advice)
        instance_idx_advices.append(idx_advices)
    return instance_idx_advices

def per_instance_test(instances, ansFuncs, label, soft, thres, nproc, ngpu):
    instances_per_proc = int(len(instances) / nproc)
    instances_split = [instances[i * instances_per_proc: (i + 1) * instances_per_proc] for i in range(nproc - 1)]
    instances_split.append(instances[(nproc - 1) * instances_per_proc:])
    manager = Manager()
    return_dict = manager.dict()

    if nproc > 1:
        procs = []
        for i in range(nproc):
            p = mp.Process(target=match_multiple_instances,
                           args=(i + 1, instances_split[i], ansFuncs, return_dict, soft, thres, ngpu))
            p.start()
            procs.append(p)

        for proc in procs:
            proc.join()

    else:
        return_dict = {}
        match_multiple_instances(1, instances, ansFuncs, return_dict, soft, thres, ngpu)

    cnt = 0
    cnt_without_ambiguity = 0
    cnt_inconsistency = 0
    cnt_different = 0
    cnt_rules = {rule_idx: (0, 0) for rule_idx in range(0, len(ansFuncs))}

    texts = []
    advice = []
    labels = []
    preds = []
    labels_all = []
    preds_all = []

    for idx, instance_results in tqdm(return_dict.items()):
        assert len(instances_split[idx - 1]) == len(instance_results)
        for idx1, instance_result in enumerate(instance_results):
            eight_set = [[], [], [], [], [], [], [], []]

            if len(instance_result) == 0:
                continue
            cnt += 1

            if len(instance_result) == 1:
                cnt_without_ambiguity += 1

            rule_labels = []
            instance_label = 0
            original_text = str()
            for idx0, result in enumerate(instance_result):
                rule_labels.append(result['rule_label'])
                instance_label = int(result['instance_label'])
                original_text = result['original_text']
                a, b = cnt_rules[result['rule_idx']]
                b += 1
                if result['rule_label'] == int(result['instance_label']):
                    a += 1
                cnt_rules[result['rule_idx']] = (a, b)

            labels.append(instance_label)
            labels_all.append(instance_label)
            if 0 in rule_labels and 1 in rule_labels:
                cnt_inconsistency += 1
            if 1 - instance_label in rule_labels:
                cnt_different += 1
            if rule_labels.count(1) > len(rule_labels)/2:
                preds.append(1)
                preds_all.append(1)
            else:
                preds.append(0)
                preds_all.append(0)
            for idx0, result in enumerate(instance_result):
                if label and result['rule_label'] != result['instance_label']:
                    continue
                update_list(result, eight_set)
            advice.append(eight_set)
            texts.append(original_text)
    for i in range(len(ansFuncs)):
        a,b = cnt_rules[i]
        print("{}\t{}".format(a,b))
    try:
        assert len(texts) == len(advice) == len(preds) == len(labels)
    except:
        print(len(texts), len(advice), len(preds), len(labels))

    return texts, advice, preds, labels

def per_instance_test_with_max_conf(instances, ansFuncs, label, soft, thres, nproc, ngpu):
    instances_per_proc = int(len(instances) / nproc)
    instances_split = [instances[i * instances_per_proc: (i + 1) * instances_per_proc] for i in range(nproc - 1)]
    instances_split.append(instances[(nproc - 1) * instances_per_proc:])
    manager = Manager()
    return_dict = manager.dict()

    if nproc > 1:
        procs = []
        for i in range(nproc):
            p = mp.Process(target=match_multiple_instances,
                           args=(i + 1, instances_split[i], ansFuncs, return_dict, soft, thres, ngpu))
            p.start()
            procs.append(p)

        for proc in procs:
            proc.join()

    else:
        return_dict = {}
        match_multiple_instances(1, instances, ansFuncs, return_dict, soft, thres, ngpu)

    cnt = 0
    cnt_rules = {rule_idx: (0, 0) for rule_idx in range(0, len(ansFuncs))}

    texts = []
    advice = []
    labels = []
    preds = []
    scores = []

    for idx, instance_results in tqdm(return_dict.items()):
        assert len(instances_split[idx - 1]) == len(instance_results)
        for idx1, instance_result in enumerate(instance_results):
            if len(instance_result) == 0:
                continue

            eight_set = [[], [], [], [], [], [], [], []]
            cnt += 1

            sorted_answers = sorted(instance_result, key=lambda x: x['confidence'], reverse=True)
            result = sorted_answers[0]
            #result = get_majority(instance_result, soft=soft)

            rule_labels = result['rule_label']
            instance_label = int(result['instance_label'])
            original_text = result['original_text']
            score = result['confidence']
            a, b = cnt_rules[rule_labels]
            b += 1
            if rule_labels == instance_label:
                a += 1
            cnt_rules[result['rule_idx']] = (a, b)

            if label and rule_labels != instance_label:
                continue

            update_list(result, eight_set)
            labels.append(instance_label)
            preds.append(rule_labels)
            scores.append(score)
            advice.append(eight_set)
            texts.append(original_text)

    for i in range(len(ansFuncs)):
        a,b = cnt_rules[i]
        print("{}\t{}".format(a,b))
    try:
        assert len(texts) == len(advice) == len(preds) == len(labels) == len(scores)
    except:
        print(len(texts), len(advice), len(preds), len(labels)), len(scores)

    return texts, advice, preds, labels, scores

def per_instance_test_with_all_conf(instances, ansFuncs, label, soft, thres, nproc, ngpu):
    instances_per_proc = int(len(instances) / nproc)
    instances_split = [instances[i * instances_per_proc: (i + 1) * instances_per_proc] for i in range(nproc - 1)]
    instances_split.append(instances[(nproc - 1) * instances_per_proc:])
    manager = Manager()
    return_dict = manager.dict()

    if nproc > 1:
        procs = []
        for i in range(nproc):
            p = mp.Process(target=match_multiple_instances,
                           args=(i + 1, instances_split[i], ansFuncs, return_dict, soft, thres, ngpu))
            p.start()
            procs.append(p)

        for proc in procs:
            proc.join()

    else:
        return_dict = {}
        match_multiple_instances(1, instances, ansFuncs, return_dict, soft, thres, ngpu)

    cnt = 0
    cnt_rules = {rule_idx: (0, 0) for rule_idx in range(0, len(ansFuncs))}

    texts = []
    advice = []
    labels = []
    preds = []
    conf_scores = []
    quality_scores = []

    for idx, instance_results in tqdm(return_dict.items()):
        assert len(instances_split[idx - 1]) == len(instance_results)
        for idx1, instance_result in enumerate(instance_results):
            if len(instance_result) == 0:
                continue

            eight_set = [[], [], [], [], [], [], [], []]
            cnt += 1

            for result in instance_result:
                rule_labels = result['rule_label']
                instance_label = int(result['instance_label'])
                original_text = result['original_text']
                conf_score = result['confidence']
                quality_score = result['quality']
                a, b = cnt_rules[rule_labels]
                b += 1
                if rule_labels == instance_label:
                    a += 1
                cnt_rules[result['rule_idx']] = (a, b)

                if label and rule_labels != instance_label:
                    continue

                update_list(result, eight_set)
                labels.append(instance_label)
                preds.append(rule_labels)
                conf_scores.append(conf_score)
                quality_scores.append(quality_score)
                advice.append(eight_set)
                texts.append(original_text)

    for i in range(len(ansFuncs)):
        a,b = cnt_rules[i]
        print("{}\t{}".format(a,b))
    try:
        assert len(texts) == len(advice) == len(preds) == len(labels) == len(conf_scores) == len(quality_scores)
    except:
        print(len(texts), len(advice), len(preds), len(labels)), len(conf_scores), len(quality_scores)

    return texts, advice, preds, labels, conf_scores, quality_scores

def preprocess_data(training_path, instance_reader):
    instances = instance_reader.read_preprocess(training_path)
    return instances

def parsing(dev_path, parser, instance_reader):
    # parsing explanation
    human_expl = json.load(open(dev_path, 'r'))
    human_expl = human_expl['annotations']

    ans_funcs = []

    logging.info("Start parsing {} rules.".format(len(human_expl)))
    for idx, row in enumerate(human_expl):
        ansFunc = get_ans_func(row, parser, instance_reader)
        ans_funcs.append((idx, ansFunc))

    return ans_funcs

def matching(ans_funcs, instances, save_path, label, soft, thres, nproc, ngpu):
    #matching
    texts, advices, preds, labels = per_instance_test(instances=instances, ansFuncs=ans_funcs, label=label, soft=soft, thres=thres, nproc=nproc, ngpu=ngpu)
    advices = get_idx_advices(advices)

    logging.info('#Total: {}'.format(len(instances)))
    logging.info('#Answered: {}'.format(len(texts)))

    logging.info('#Metrics on matched instances only:')
    accuracy = accuracy_score(labels, preds)
    logging.info("Accuracy: %.3f" % accuracy)
    macro_precision = precision_score(labels, preds)
    logging.info("Precision: %.3f" % macro_precision)
    macro_recall = recall_score(labels, preds)
    logging.info("Recall: %.3f" % macro_recall)
    macro_f1score = f1_score(labels, preds)
    logging.info("F1: %.3f" % macro_f1score)

    #save
    with open(save_path, 'w') as f:
        for idx in range(len(advices)):
            f.write("{0}\t{1}\t{2}\t{3}\n".format(texts[idx], advices[idx], preds[idx], labels[idx]))

def matching_with_conf(ans_funcs, instances, save_path, label, soft, thres, nproc, ngpu):
    #matching
    texts, advices, preds, labels, scores = per_instance_test_with_max_conf(instances=instances, ansFuncs=ans_funcs, label=label, soft=soft, thres=thres, nproc=nproc, ngpu=ngpu)
    advices = get_idx_advices(advices)

    logging.info('#Total: {}'.format(len(instances)))
    logging.info('#Answered: {}'.format(len(texts)))

    logging.info('#Metrics on matched instances only:')
    accuracy = accuracy_score(labels, preds)
    logging.info("Accuracy: %.3f" % accuracy)
    macro_precision = precision_score(labels, preds)
    logging.info("Precision: %.3f" % macro_precision)
    macro_recall = recall_score(labels, preds)
    logging.info("Recall: %.3f" % macro_recall)
    macro_f1score = f1_score(labels, preds)
    logging.info("F1: %.3f" % macro_f1score)

    #save
    with open(save_path, 'w') as f:
        for idx in range(len(advices)):
            f.write("{0}\t{1}\t{2}\t{3}\n".format(texts[idx], advices[idx], preds[idx], scores[idx]))

def matching_with_all_conf(ans_funcs, instances, save_path, label, soft, thres, nproc, ngpu):
    #matching
    texts, advices, preds, labels, conf_scores, quality_scores = per_instance_test_with_all_conf(instances=instances, ansFuncs=ans_funcs, label=label, soft=soft, thres=thres, nproc=nproc, ngpu=ngpu)
    advices = get_idx_advices(advices)

    logging.info('#Total: {}'.format(len(instances)))
    logging.info('#Answered: {}'.format(len(texts)))

    logging.info('#Metrics on matched instances only:')
    accuracy = accuracy_score(labels, preds)
    logging.info("Accuracy: %.3f" % accuracy)
    macro_precision = precision_score(labels, preds)
    logging.info("Precision: %.3f" % macro_precision)
    macro_recall = recall_score(labels, preds)
    logging.info("Recall: %.3f" % macro_recall)
    macro_f1score = f1_score(labels, preds)
    logging.info("F1: %.3f" % macro_f1score)

    #save
    with open(save_path, 'w') as f:
        for idx in range(len(advices)):
            f.write("{0}\t{1}\t{2}\t{3}\t{4}\n".format(texts[idx], advices[idx], preds[idx], conf_scores[idx], quality_scores[idx]))

def per_instance_extract(instances, ansFuncs, label, soft):
    nproc = 40
    instances_per_proc = int(len(instances) / nproc)
    instances_split = [instances[i * instances_per_proc: (i + 1) * instances_per_proc] for i in range(nproc - 1)]
    instances_split.append(instances[(nproc - 1) * instances_per_proc:])
    manager = Manager()
    return_dict = manager.dict()

    if nproc > 1:
        procs = []
        for i in range(nproc):
            p = mp.Process(target=match_multiple_instances,
                           args=(i + 1, instances_split[i], ansFuncs, return_dict, soft))
            p.start()
            procs.append(p)
        for proc in procs:
            proc.join()
    else:
        return_dict = {}
        match_multiple_instances(1, instances, ansFuncs, return_dict, soft)

    cnt = 0
    reference_texts = list()
    reference_words = list()
    reference_chunks = list()
    texts = list()
    chunks = list()
    answers = list()

    for idx, instance_results in tqdm(return_dict.items()):
        assert len(instances_split[idx - 1]) == len(instance_results)
        for idx1, instance_result in enumerate(instance_results):
            if len(instance_result) == 0:
                continue
            cnt += 1

            for idx0, result in enumerate(instance_result):
                if label and result['rule_label'] != result['instance_label']:
                    continue
                reference_text = ansFuncs[result['rule_idx']][1].reference_instance.original_context
                text = result['original_text']
                chunk_init = result['instance_chunk']
                chunk = list()
                for element in chunk_init:
                    chunk.append([element['begin_idx'], element['end_idx']])
                reference_chunk_init = ansFuncs[result['rule_idx']][1].reference_instance.context_info['constituency']
                reference_chunk = list()
                for element in reference_chunk_init:
                    reference_chunk.append([element['begin_idx'], element['end_idx']])
                reference_word = list()
                answer = list()
                for j in ['X', 'Y', 'Z']:
                    if result.__contains__(j):
                        element = ansFuncs[result['rule_idx']][1].variables[j].loc_in_context[0]
                        reference_word.append([element['st'], element['ed']-1])
                        answer.append([result[j][1], result[j][2]])
                reference_texts.append(reference_text)
                reference_chunks.append(reference_chunk)
                reference_words.append(reference_word)
                texts.append(text)
                chunks.append(chunk)
                answers.append(answer)

    try:
        assert len(texts) == len(reference_texts) == len(chunks) == len(reference_words) == len(answers) == len(reference_chunks)
    except:
        print(len(texts), len(reference_texts), len(chunks), len(reference_words), len(answers), len(reference_chunks))

    print("Matched number: {}".format(cnt))
    return reference_texts, reference_words, reference_chunks, texts, chunks, answers

def extract_matching(ans_funcs, instances, save_path, label, soft):
    #matching
    reference_texts, reference_words, reference_chunks, texts, chunks, answers = per_instance_extract(instances=instances, ansFuncs=ans_funcs, label=label, soft=soft)
    #save
    with open(save_path, 'w') as f:
        for idx in range(len(texts)):
            f.write("{0}\t{1}\t{2}\t{3}\t{4}\t{5}\n".format(reference_texts[idx], reference_words[idx], reference_chunks[idx], texts[idx], chunks[idx], answers[idx]))

def per_rule_matching(instances, ans_funcs, label, soft, thres, nproc, ngpu, save_path):
    #parsing
    try:
        assert len(ans_funcs) == 1
    except:
        logging.info('validate rule > 1')
        raise AssertionError
    for idx, ans_func in ans_funcs:
        temp_func = [(idx, ans_func)]
        texts, advices, preds, labels = per_instance_test(instances=instances, ansFuncs=temp_func, label=label, soft=soft, thres=thres, nproc=nproc, ngpu=ngpu)
        span_advices = get_span_advices(advices)
        '''
        labels_all = copy.deepcopy(labels)
        preds_all = copy.deepcopy(preds)
        for instance in instances:
            if instance.original_context not in texts:
                labels_all.append(instance.label)
                preds_all.append(1 - temp_func[0][1].label)
        '''
        logging.info('#Total: {}'.format(len(instances)))
        logging.info('#Answered: {}'.format(len(texts)))

        logging.info('#Metrics on matched instances only:')
        matched_accuracy = accuracy_score(labels, preds)
        logging.info("Accuracy: %.3f" % matched_accuracy)
        matched_precision = precision_score(labels, preds)
        logging.info("Precision: %.3f" % matched_precision)
        matched_recall = recall_score(labels, preds)
        logging.info("Recall: %.3f" % matched_recall)
        matched_f1score = f1_score(labels, preds)
        logging.info("F1: %.3f" % matched_f1score)
        '''
        logging.info('#Metrics on all instances:')
        accuracy = accuracy_score(labels_all, preds_all)
        logging.info("Accuracy: %.3f" % accuracy)
        precision = precision_score(labels_all, preds_all)
        logging.info("Precision: %.3f" % precision)
        recall = recall_score(labels_all, preds_all)
        logging.info("Recall: %.3f" % recall)
        f1score = f1_score(labels_all, preds_all)
        logging.info("F1: %.3f" % f1score)
        '''
        with open(save_path, 'a+') as f:
            f.write('=======Rule #{}======='.format(idx))
            f.write('\n')
            for idx0 in range(len(texts)):
                f.write("{0}\t{1}\t{2}\t{3}\n".format(texts[idx0], span_advices[idx0], preds[idx0], labels[idx0]))

def set_save(instances, path):
    all_instances = []
    for instance in instances:
        instance_dict = {"text": instance.original_context,
                         "label": str(instance.label)}
        all_instances.append(instance_dict)
    with open(path,'w') as f:
        for instance in all_instances:
            f.write("{}\n".format(json.dumps(instance)))

def data_resplit(data_path):
    print("loading preprocessed data ...")
    with open(data_path, 'rb') as f:
        instances = pickle.load(f)

    # dataset resplit
    total_cnt = len(instances)
    train_cnt = int(0.8 * total_cnt)
    dev_cnt = int(0.1 * total_cnt)
    test_cnt = total_cnt - train_cnt - dev_cnt
    print("total: {0}, train: {1}, dev: {2}, test: {3}".format(total_cnt, train_cnt, dev_cnt, test_cnt))

    random.shuffle(instances)

    train_instances = instances[:train_cnt]
    dev_instances = instances[train_cnt: train_cnt + dev_cnt]
    test_instances = instances[train_cnt + dev_cnt:]
    return train_instances, dev_instances, test_instances

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--rule_path',
                        default=None, type=str, required=True,
                        help='Rule file path')
    parser.add_argument('--data_path',
                        default='../data/split_1', type=str, required=True,
                        help='dataset path')
    #parser.add_argument('--task',
    #                    type=str, required=True,
    #                    help='task type')
    parser.add_argument('--validate',
                        action='store_true', help='validate the rule')
    parser.add_argument('--validation_set',
                        default='dev', choices=['dev', 'train', 'both', 'test'], type=str, required=False,
                        help='use which set to validate the rule')
    parser.add_argument('--validation_save_path',
                        default=None, type=str, required=False,
                        help='path to save validation details')
    parser.add_argument('--regularization',
                        action='store_true',
                        help='generate regularization advice')
    parser.add_argument('--confidence',
                        action='store_true',
                        help='if need results with max confidence score')
    parser.add_argument('--allconfidence',
                        action='store_true',
                        help='if need results with all results with confidence score')
    parser.add_argument('--extract',
                        action='store_true',
                        help='extract data for training modules')
    parser.add_argument('--extract_save_path',
                        default=None, type=str, required=False,
                        help='path to save extracted data')
    parser.add_argument('--soft',
                        default=0, choices=[0,1], type=int, required=True,
                        help='matching version')
    parser.add_argument('--label',
                        default=0, choices=[0, 1], type=int, required=True,
                        help='if use label to filter out matching results')
    parser.add_argument('--advice_path',
                        default=None, type=str, required=False,
                        help='path to save regularization advice')
    parser.add_argument('--thres',
                        default=1.0, type=float, required=False,
                        help='threshold to filter soft matching results')
    parser.add_argument('--nproc',
                        default=1, type=int, required=False,
                        help='number of processor')
    parser.add_argument('--ngpu',
                        default=1, type=int, required=False,
                        help='number of gpu')

    args = parser.parse_args()
    if args.soft == 0:
        soft = False
    else:
        soft = True


    parser = Parser()
    instance_reader = InstancePreprocess(pre=True)
    mp.set_start_method('spawn')

    logging.basicConfig(format='%(message)s',level=logging.INFO)


    if args.validate:
        logging.info("Starting to validate every rule...")
    elif args.regularization:
        logging.info("Starting to generate regularization advice...")
    elif args.extract:
        logging.info("Starting to extract data for training modules...")
    else:
        logging.info('Error: Operation not define.')
        raise Exception

    # loading rules
    logging.info("loading rules...")
    ans_funcs = parsing(args.rule_path, parser, instance_reader)
    for idx, ans_func in ans_funcs:
        logging.info('=======Rule #{}======='.format(idx))
        logging.info([item.raw_str for item in ans_func.all_rules])
        logging.info(ans_func.advice)

    for idx, ans_func in ans_funcs:
        ans_func.clean_rules()

    if args.extract:
        if args.extract_save_path == None:
            logging.info('Error: Save path not define.')
            raise Exception

        training_path = os.path.join(args.data_path, "train.pkl")
        if os.path.exists(training_path):
            logging.info("loading training set...")
            with open(training_path, 'rb') as f:
                train_instances = pickle.load(f)
        else:
            logging.info("preprocessing training set...")
            train_instances = preprocess_data(os.path.join(args.data_path, "train.jsonl"), instance_reader)
            with open(training_path, 'wb') as f:
                pickle.dump(train_instances, f)

        # training set matching
        logging.info("matching {} rules with training set...".format(len(ans_funcs)))
        extract_matching(ans_funcs, train_instances,
                        args.extract_save_path, label=args.label, soft=soft)

    if args.regularization:
        if args.advice_path == None:
            logging.info('Error: Save path not define.')
            raise Exception

        #loading data
        training_path = os.path.join(args.data_path, "train.pkl")
        if os.path.exists(training_path):
            logging.info("loading training set...")
            with open(training_path, 'rb') as f:
                train_instances = pickle.load(f)
        else:
            logging.info("preprocessing training set...")
            train_instances = preprocess_data(os.path.join(args.data_path, "train.jsonl"), instance_reader)
            with open(training_path, 'wb') as f:
                pickle.dump(train_instances, f)

        # training set matching
        logging.info("matching {} rules with training set...".format(len(ans_funcs)))
        if args.confidence:
            matching_with_conf(ans_funcs, train_instances,
                 args.advice_path,
                 label = args.label, soft=soft, thres=args.thres,
                 nproc = args.nproc, ngpu = args.ngpu)
        elif args.allconfidence:
            matching_with_all_conf(ans_funcs, train_instances,
                               args.advice_path,
                               label=args.label, soft=soft, thres=args.thres,
                               nproc=args.nproc, ngpu=args.ngpu)
        else:
            matching(ans_funcs, train_instances,
                 args.advice_path,
                 label = args.label, soft=soft, thres=args.thres,
                 nproc = args.nproc, ngpu = args.ngpu)

    if args.validate:
        if args.validation_save_path == None:
            logging.info('Error: Save path not define.')
            raise Exception

        logging.info("using {} as validation set".format(args.validation_set))

        if args.validation_set == 'test':
            # loading data
            test_path = os.path.join(args.data_path, "test.pkl")
            if os.path.exists(test_path):
                logging.info("loading dev set...")
                with open(test_path, 'rb') as f:
                    test_instances = pickle.load(f)
            else:
                logging.info("preprocessing test set...")
                test_instances = preprocess_data(os.path.join(args.data_path, "test.jsonl"), instance_reader)
                with open(test_path, 'wb') as f:
                    pickle.dump(test_instances, f)
            logging.info("matching with dev set...")
            per_rule_matching(instances=test_instances, ans_funcs=ans_funcs,
                              label=args.label, soft=soft, thres=args.thres,
                              nproc=args.nproc, ngpu=args.ngpu,
                              save_path=args.validation_save_path)

        if args.validation_set == 'dev' or args.validation_set == 'both':
            # loading data
            dev_path = os.path.join(args.data_path, "dev.pkl")
            if os.path.exists(dev_path):
                logging.info("loading dev set...")
                with open(dev_path, 'rb') as f:
                    dev_instances = pickle.load(f)
            else:
                logging.info("preprocessing dev set...")
                dev_instances = preprocess_data(os.path.join(args.data_path, "dev.jsonl"), instance_reader)
                with open(dev_path, 'wb') as f:
                    pickle.dump(dev_instances, f)
            logging.info("matching with dev set...")
            per_rule_matching(instances=dev_instances, ans_funcs=ans_funcs,
                              label=args.label, soft=soft, thres=args.thres,
                              nproc=args.nproc, ngpu=args.ngpu,
                              save_path=args.validation_save_path)

        if args.validation_set == 'train' or args.validation_set == 'both':
            # loading data
            training_path = os.path.join(args.data_path, "train.pkl")
            if os.path.exists(training_path):
                logging.info("loading training set...")
                with open(training_path, 'rb') as f:
                    train_instances = pickle.load(f)
            else:
                logging.info("preprocessing training set...")
                train_instances = preprocess_data(os.path.join(args.data_path, "train.jsonl"), instance_reader)
                with open(training_path, 'wb') as f:
                    pickle.dump(train_instances, f)
            logging.info("matching with training set...")
            per_rule_matching(instances=train_instances, ans_funcs=ans_funcs,
                              label = args.label, soft=soft, thres=args.thres,
                              nproc=args.nproc, ngpu=args.ngpu,
                              save_path=args.validation_save_path)

