import os
import pickle as pkl
import argparse
import re
import json
import numpy as np
import nltk
from collections import defaultdict



def generate_answer_nl(in_dir, dtype):
    with open(os.path.join(in_dir, '{}_answer.src'.format(dtype)),'r') as f:
        subgraph_answer = f.readlines()

    ans_list = []
    for line in subgraph_answer:
        triples = line.split()

        tmp = []
        find_answer = False
        for each in triples:
            if '￨A' in each:
                tmp.append(each.split('￨A')[0])
                find_answer = True
            else:
                assert find_answer == False
                find_answer = False

        tmp = ' '.join(tmp)
        ans_list.append([tmp])

    print('generate_answer_nl')
    return ans_list


def build_input_seq(in_dir, dtype):
    with open(os.path.join(in_dir, '{}.src'.format(dtype)),'r') as f2:
        subgraphf = f2.readlines()

    num_triples = []
    all_input_seqs = []

    for line in subgraphf:
        path_list = line.strip().split()
        num_triples.append((len(path_list) - 1) // 2)

        input_seq = []
        for idx, element in enumerate(path_list):
            if idx % 2 == 0:
                input_seq.append(' '.join(element.split('_')))
            else:
                element = element.split('__')[-1]
                element = ' '.join(element.replace('_', ' ').split())
                input_seq.append(element)
        input_seq = ' '.join(input_seq)
        all_input_seqs.append(input_seq)

    print('average # of triples', np.mean(num_triples))
    return all_input_seqs


def process_querys(in_dir, dtype):
    with open(os.path.join(in_dir, '{}.tgt'.format(dtype)),'r') as f:
        querys = f.readlines()

    new_queries = []
    for idx, line in enumerate(querys):
        line = line.lower().replace('_', ' ').strip()
        new_queries.append(line)

    return new_queries


def prepare_output_data(input_seqs, answer_lists, queries, dtype, out_dir):
    count = 0
    with open(os.path.join(out_dir, '{}.seq.json'.format(dtype)), 'w') as outf:
        for i in range(len(input_seqs)):
            example = {}
            example['answers'] = answer_lists[i]
            example['outSeq'] = queries[i]
            example['qId'] = count + 1
            example['inSeq'] = input_seqs[i]

            outf.write(json.dumps(example) + '\n')
            count += 1
    return count


def prepare_output_data_example_per_line(input_seqs, answer_lists, queries, dtype, out_dir):
    count = 0
    with open(os.path.join(out_dir, 'src-{}.txt'.format(dtype)), 'w') as outf:
        for i in range(len(input_seqs)):
            outf.write(input_seqs[i] + '\n')

    with open(os.path.join(out_dir, 'tgt-{}.txt'.format(dtype)), 'w') as outf:
        for i in range(len(queries)):
            outf.write(queries[i] + '\n')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_dir', required=True, type=str, help='path to the input dir')
    parser.add_argument('-o', '--output_dir', required=True, type=str, help='path to the output dir')
    opt = vars(parser.parse_args())

    for dtype in ['train', 'test', 'dev']:
        new_input_seqs = build_input_seq(opt['input_dir'], dtype)
        new_ans_list = generate_answer_nl(opt['input_dir'], dtype)
        new_queries = process_querys(opt['input_dir'], dtype)
        assert len(new_input_seqs) == len(new_ans_list) == len(new_queries)

        prepare_output_data(new_input_seqs, new_ans_list, new_queries, dtype, opt['output_dir'])
        # prepare_output_data_example_per_line(new_input_seqs, new_ans_list, new_queries, dtype, opt['output_dir'])



