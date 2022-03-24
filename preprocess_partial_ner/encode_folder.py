import pickle
import argparse
import os
import random
import numpy as np
from tqdm import tqdm

import itertools
import functools

def filter_words(w_map, emb_array, ck_filenames):
    """ delete word in w_map but not in the current corpus """
    vocab = set()
    for filename in ck_filenames:
        for line in open(filename, 'r'):
            if not (line.isspace() or (len(line) > 10 and line[0:10] == '-DOCSTART-')):
                line = line.rstrip('\n').split()
                assert len(line) >= 3, 'wrong ck file format'
                word = line[0]
                vocab.add(word)
                word = word.lower()
                vocab.add(word)
    new_w_map = {}
    new_emb_array = []
    for (word, idx) in w_map.items():
        if word in vocab or word in ['<unk>', '<s>', '< >', '<\n>']:
            assert word not in new_w_map
            new_w_map[word] = len(new_emb_array)
            new_emb_array.append(emb_array[idx])
    print('filtered %d --> %d' % (len(emb_array), len(new_emb_array)))
    return new_w_map, new_emb_array


def build_label_mapping(train_file, dev_file, test_file):
    ret = {'None': 0} # None must be 0
    for filename in [train_file, dev_file, test_file]:
        for line in open(filename):
            if not (line.isspace() or (len(line) > 10 and line[0:10] == '-DOCSTART-')):
                line = line.rstrip('\n').split()
                assert len(line) >= 3 and len(line) <= 4, "the format of noisy corpus"
                # The format should be
                # 0. Token
                # 1. I/O (I means Break, O means Connected)
                # 2. Type (separated by comma)
                # 3. Safe or dangerous?   <-- this is optional
                token = line[0]
                chunk_gap = line[1]
                entity_type = line[2]
                if entity_type not in ret:
                    type_id = len(ret)
                    ret[entity_type] = type_id
                    print('\tlabel mapping: %s --> %d' % (entity_type, type_id))
    return ret


def read_noisy_corpus(lines):
    features, gap_labels, safe_labels, gap_ids, type_labels = list(), list(), list(), list(), list()

    tmp_tokens, tmp_gap_ids, tmp_safe_labels, tmp_gap_labels, tmp_type_lst = (
        list(), list(), list(), list(), list())

    for line in lines:
        if not (line.isspace() or (len(line) > 10 and line[0:10] == '-DOCSTART-')):
            line = line.rstrip('\n').split()
            
            assert len(line) >= 3 and len(line) <= 4, "the format of noisy corpus"
            # The format should be
            # 0. Token
            # 1. I/O (I means Break, O means Connected)
            # 2. Type (separated by comma)
            # 3. Safe or dangerous?   <-- this is optional
            token = line[0]
            chunk_gap = line[1]
            entity_types = line[2]

            if len(line) == 3:
                safe = 1
            else:
                safe = int(line[3] == 'S')

            tmp_tokens.append(token)
            tmp_safe_labels.append(safe)
            if safe:
                tmp_gap_labels.append(chunk_gap)
                if 'I' == chunk_gap:
                    tmp_gap_ids.append(1)
                    tmp_type_lst.append(entity_types.split(','))
                else:
                    tmp_gap_ids.append(0)
        elif len(tmp_tokens) > 0:
            features.append(tmp_tokens)
            gap_labels.append(tmp_gap_labels)
            safe_labels.append(tmp_safe_labels)
            gap_ids.append(tmp_gap_ids)
            type_labels.append(tmp_type_lst)
            tmp_tokens, tmp_gap_ids, tmp_safe_labels, tmp_gap_labels, tmp_type_lst = (
                list(), list(), list(), list(), list())

    if len(tmp_tokens) > 0:
        features.append(tmp_tokens)
        gap_labels.append(tmp_gap_labels)
        safe_labels.append(tmp_safe_labels)
        gap_ids.append(tmp_gap_ids)
        type_labels.append(tmp_type_lst)

    return features, gap_labels, safe_labels, gap_ids, type_labels


def encode_folder(input_file, output_folder, w_map, char_map, gap_label_to_id, type_label_map, 
    char_threshold = 5):
    """  
    1. use char_threshold to filter out rare count char and treat them as '<unk>'.
    """
    w_st, w_unk, w_con, w_pad = w_map['<s>'], w_map['<unk>'], w_map['< >'], w_map['<\n>']
    c_st, c_unk, c_con, c_pad = char_map['<s>'], char_map['<unk>'], char_map['< >'], char_map['<\n>']
    # list_dirs = os.walk(input_folder)
    range_ind = 0
    # for root, dirs, files in list_dirs:
        # print('loading from ' + ', '.join(files))
        # for file in tqdm(files):
            # with open(os.path.join(root, file), 'r') as fin:
    with open(input_file, 'r') as fin:
        lines = fin.readlines()

    # use sentence as per group
    features, gap_labels, safe_labels, gap_ids, type_labels = read_noisy_corpus(lines)

    # initial char_map = {'<s>': 0, '<unk>': 1, '< >': 2, '<\n>': 3}
    if char_threshold > 0:
        c_count = dict()
        for line in features:
            for token in line:
                for t_char in token:
                    c_count[t_char] = c_count.get(t_char, 0) + 1
        char_set = [k for k, v in c_count.items() if v > char_threshold]
        for key in char_set:
            if key not in char_map:
                char_map[key] = len(char_map)

    dataset = list()
    # sentence level lists: f_l, sub_gap_labels, sub_safe_labels, sub_gap_ids, sub_type_labels
    for f_l, sub_gap_labels, sub_safe_labels, sub_gap_ids, sub_type_labels in zip(
        features, gap_labels, safe_labels, gap_ids, type_labels):
        tmp_w = [w_st, w_con]
        tmp_c = [c_st, c_con]
        tmp_mc = [0, 1]

        for i_f, i_m in zip(f_l[1:-1], sub_safe_labels[1:-1]):
            tmp_w = tmp_w + [w_map.get(i_f, w_map.get(i_f.lower(), w_unk))] * len(i_f) + [w_con]
            tmp_c = tmp_c + [char_map.get(t, c_unk) for t in i_f] + [c_con]
            tmp_mc = tmp_mc + [0] * len(i_f) + [i_m]

        tmp_w.append(w_pad)
        tmp_c.append(c_pad)
        tmp_mc.append(0)

        # gap_label_to_id = {'I': 0, 'O': 1}
        ### tmp_lc is the opposite of tmp_mt
        tmp_lc = [gap_label_to_id[tup] for tup in sub_gap_labels[1:]]
        tmp_mt = sub_gap_ids[1:]
        tmp_lt = list()
        for tup_list in sub_type_labels:
            tmp_mask = [0] * len(type_label_map)
            for tup in tup_list:
                tmp_mask[type_label_map[tup]] = 1
            tmp_lt.append(tmp_mask)

        dataset.append([tmp_w, tmp_c, tmp_mc, tmp_lc, tmp_mt, tmp_lt])

    dataset.sort(key=lambda t: len(t[0]), reverse=True)

    with open(output_folder+'train_'+ str(range_ind) + '.pk', 'wb') as f:
        pickle.dump(dataset, f)

    range_ind += 1
    return range_ind


def read_corpus(lines):
    features, gap_labels, gap_ids, type_labels = list(), list(), list(), list()

    tmp_tokens, tmp_gap_ids, tmp_gap_labels, tmp_type_lst = list(), list(), list(), list()

    for line in lines:
        if not (line.isspace() or (len(line) > 10 and line[0:10] == '-DOCSTART-')):
            line = line.rstrip('\n').split()

            assert len(line) == 3, "the format of corpus"
            # The format should be
            # 0. Token
            # 1. I/O (I means Break, O means Connected)
            # 2. Type (separated by comma)
            token = line[0]
            chunk_gap = line[1]
            entity_types = line[2]

            tmp_tokens.append(token)
            tmp_gap_labels.append(chunk_gap)
            if 'I' == chunk_gap:
                tmp_gap_ids.append(1)
                tmp_type_lst.append(entity_types)
            else:
                tmp_gap_ids.append(0)
        elif len(tmp_tokens) > 0:
            features.append(tmp_tokens)
            gap_labels.append(tmp_gap_labels)
            gap_ids.append(tmp_gap_ids)
            type_labels.append(tmp_type_lst)
            tmp_tokens, tmp_gap_ids, tmp_gap_labels, tmp_type_lst = list(), list(), list(), list()

    if len(tmp_tokens) > 0:
        features.append(tmp_tokens)
        gap_labels.append(tmp_gap_labels)
        gap_ids.append(tmp_gap_ids)
        type_labels.append(tmp_type_lst)

    return features, gap_labels, gap_ids, type_labels


def encode_dataset(input_file, w_map, char_map, gap_label_to_id, type_label_map):

    print('loading from ' + input_file)

    with open(input_file, 'r') as f:
        lines = f.readlines()

    features, gap_labels, gap_ids, type_labels = read_corpus(lines)

    w_st, w_unk, w_con, w_pad = w_map['<s>'], w_map['<unk>'], w_map['< >'], w_map['<\n>']
    c_st, c_unk, c_con, c_pad = char_map['<s>'], char_map['<unk>'], char_map['< >'], char_map['<\n>']

    dataset = list()

    for f_l, sub_gap_labels, sub_gap_ids, sub_type_labels in zip(
        features, gap_labels, gap_ids, type_labels):
        tmp_w = [w_st, w_con]
        tmp_c = [c_st, c_con]
        tmp_mc = [0, 1]

        for i_f in f_l[1:-1]:
            tmp_w = tmp_w + [w_map.get(i_f, w_map.get(i_f.lower(), w_unk))] * len(i_f) + [w_con]
            tmp_c = tmp_c + [char_map.get(t, c_unk) for t in i_f] + [c_con]
            tmp_mc = tmp_mc + [0] * len(i_f) + [1]

        tmp_w.append(w_pad)
        tmp_c.append(c_pad)
        tmp_mc.append(0)

        tmp_lc = [gap_label_to_id[tup] for tup in sub_gap_labels[1:]]
        tmp_mt = sub_gap_ids[1:]
        tmp_lt = [type_label_map[tup] for tup in sub_type_labels]

        dataset.append([tmp_w, tmp_c, tmp_mc, tmp_lc, tmp_mt, tmp_lt])

    dataset.sort(key=lambda t: len(t[0]), reverse=True)

    return dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_train', default="./annotations/debug.ck")
    parser.add_argument('--input_testa', default="./data/ner/eng.testa.ck")
    parser.add_argument('--input_testb', default="./data/ner/eng.testb.ck")
    parser.add_argument('--pre_word_emb', default="./data/glove.100.pk")
    parser.add_argument('--output_folder', default="./data/hqner/")
    args = parser.parse_args()

    with open(args.pre_word_emb, 'rb') as f:
        w_emb = pickle.load(f)
        w_map = w_emb['w_map']
        emb_array = w_emb['emb_array']

    w_map, emb_array = filter_words(w_map, emb_array, [args.input_train, args.input_testa, args.input_testb])
    assert len(w_map) == len(emb_array)
    
    #four special char/word, <s>, <unk>, < > and <\n>
    char_map = {'<s>': 0, '<unk>': 1, '< >': 2, '<\n>': 3}
    # tl_map = {'None': 0,
    #           'PER': 1, 'ORG': 2, 'LOC': 3,
    #           'Chemical': 1, 'Disease': 2, 'Gene' : 3, 'Pathway' : 4, 'Protein' : 5, 'Mutation' : 6, 'Species': 7,
    #           'AspectTerm': 1}
    type_label_map = build_label_mapping(args.input_train, args.input_testa, args.input_testb)
    gap_label_to_id = {'I': 0, 'O': 1}

    range_ind = encode_folder(args.input_train, args.output_folder, w_map, char_map, gap_label_to_id, 
        type_label_map, char_threshold=5)
    testa_dataset = encode_dataset(args.input_testa, w_map, char_map, gap_label_to_id, type_label_map)
    testb_dataset = encode_dataset(args.input_testb, w_map, char_map, gap_label_to_id, type_label_map)

    with open(args.output_folder+'test.pk', 'wb') as f:
        pickle.dump({'emb_array': emb_array, 'w_map': w_map, 'c_map': char_map, 'tl_map': type_label_map, 'cl_map': gap_label_to_id, 'range': range_ind, 'test_data':testb_dataset, 'dev_data': testa_dataset}, f)

    print('dumped to the folder: ' + args.output_folder)
    print('done!')
