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


def build_subgraph_nl(in_dir, dtype):
    with open(os.path.join(in_dir, '{}.src'.format(dtype)),'r') as f2:
        subgraphf = f2.readlines()

    new_subgraph = []

    all_ents = set()
    all_rels = set()
    num_triples = []
    formatted_subgraphs = []

    for line in subgraphf:
        path_list = line.strip().split()

        g_node_names = {}
        # g_node_types = {}
        g_edge_types = {}
        g_adj = defaultdict(dict)

        assert len(path_list) % 2 == 1
        num_triples.append((len(path_list) - 1) // 2)

        triple = []
        for idx, element in enumerate(path_list):
            if idx % 2 == 0:
                all_ents.add(element)
                g_node_names[element] = ' '.join(element.split('_'))
            else:
                element = element.replace('__', '/')
                all_rels.add(element)
                g_edge_types[element] = element

            triple.append(element)
            if idx > 0 and idx % 2 == 0:
                triple = triple[-3:]
                if triple[2] in g_adj[triple[0]]:
                    g_adj[triple[0]][triple[2]].append(triple[1])
                else:
                    g_adj[triple[0]][triple[2]] = [triple[1]]

        subgraph = {'g_node_names': g_node_names,
                    'g_edge_types': g_edge_types,
                    'g_adj': g_adj}

        assert len(g_adj) > 0
        formatted_subgraphs.append(subgraph)

    print('build_subgraph_nl')
    print('all_ents', len(all_ents))
    print('all_rels', len(all_rels))
    print('# of triples: min: {}, max: {}, mean: {}'.format(np.min(num_triples), np.max(num_triples), np.mean(num_triples)))
    return formatted_subgraphs


def process_querys(in_dir, dtype):
    with open(os.path.join(in_dir, '{}.tgt'.format(dtype)),'r') as f:
        querys = f.readlines()

    new_queries = []
    for idx, line in enumerate(querys):
        line = line.lower().replace('_', ' ').strip()
        new_queries.append(line)

    return new_queries


def prepare_output_data(subgraphs, answer_lists, queries, dtype, out_dir):
    count = 0
    with open(os.path.join(out_dir, '{}.json'.format(dtype)), 'w') as outf:
        for i in range(len(subgraphs)):
            example = {}
            example['answers'] = answer_lists[i]
            example['outSeq'] = queries[i]
            example['qId'] = count + 1
            # example['topicEntityID'] = None
            example['inGraph'] = subgraphs[i]

            outf.write(json.dumps(example) + '\n')
            count += 1
    return count



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_dir', required=True, type=str, help='path to the input dir')
    parser.add_argument('-o', '--output_dir', required=True, type=str, help='path to the output dir')
    opt = vars(parser.parse_args())

    for dtype in ['train', 'test', 'dev']:
        new_subgraph = build_subgraph_nl(opt['input_dir'], dtype)
        new_ans_list = generate_answer_nl(opt['input_dir'], dtype)
        new_queries = process_querys(opt['input_dir'], dtype)
        assert len(new_subgraph) == len(new_ans_list) == len(new_queries)

        prepare_output_data(new_subgraph, new_ans_list, new_queries, dtype, opt['output_dir'])



