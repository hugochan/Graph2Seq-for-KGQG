'''
Created on Oct, 2019

@author: hugo

'''
import os
import argparse
from collections import defaultdict

from utils import *


def prepare_webquestions(data, kg, out_path):
    count = 0
    with open(out_path, 'w') as outf:
        for each in data:
            example = {}
            example['answers'] = each['answers']
            example['outSeq'] = each['qText']
            example['qId'] = each['qId']

            subgraph = extract_kg_subgraph(kg.get(each['freebaseKey'], None), [x[0] for x in each['relPaths']], each['answers'], each['qText'])
            if subgraph is not None:
                example['topicEntityID'] = kg[each['freebaseKey']]['id']
                example['inGraph'] = subgraph
                outf.write(json.dumps(example) + '\n')
                count += 1
    return count


def extract_kg_subgraph(graph, all_rel_paths, all_answers, query):
    if graph is None:
        return None

    g_node_names = {}
    g_node_types = {}
    g_edge_types = {}
    g_adj = defaultdict(dict)

    all_answers = [normalize_answer(x) for x in all_answers]

    topic_key_id = graph['id']

    selected_names = (graph['name'] + graph['alias'])[:1]
    topic_key_name = selected_names[0] if len(selected_names) > 0 else ''

    selected_types = (graph['notable_types'] + graph['type'])[:1]
    topic_key_type = selected_types[0] if len(selected_types) > 0 else ''

    # We only consider the alias relations of topic entityies
    if ['/common/topic/alias'] in all_rel_paths:
        for each in graph.get('alias', []):
            if normalize_answer(each) in all_answers:
                g_node_names[each] = each
                g_node_types[each] = 'str'
                g_edge_types['/common/topic/alias'] = '/common/topic/alias'
                g_adj[topic_key_id][each] = '/common/topic/alias'


    if not 'neighbors' in graph and len(g_adj) == 0:
        return None



    mentioned_rels = {rel for path in all_rel_paths for rel in path}

    for k, v in graph['neighbors'].items():
        # If the rel is not part of the paths, we discard this rel and the subsequent rels
        if not k in mentioned_rels:
            continue


        rel_path = [k]
        for nbr in v:
            if isinstance(nbr, str) or isinstance(nbr, bool) or isinstance(nbr, float):
                if rel_path in all_rel_paths and normalize_answer(str(nbr)) in all_answers: # Find a matched path
                    node_id = str(nbr)
                    g_node_names[node_id] = str(nbr)

                    if isinstance(nbr, str):
                        g_node_types[node_id] = 'str'
                    elif isinstance(nbr, bool):
                        g_node_types[node_id] = 'bool'
                    else:
                        g_node_types[node_id] = 'num'

                    g_edge_types[k] = k
                    g_adj[topic_key_id][node_id] = k

            elif isinstance(nbr, dict):
                node_id = list(nbr.keys())[0]
                nbr_v = nbr[node_id]

                selected_names = (nbr_v['name'] + nbr_v['alias'])[:1]
                node_name = selected_names[0] if len(selected_names) > 0 else ''
                selected_types = (nbr_v['notable_types'] + nbr_v['type'])[:1]
                node_type = selected_types[0] if len(selected_types) > 0 else ''


                if rel_path in all_rel_paths:
                    if normalize_answer(node_name) in all_answers:
                        g_node_names[node_id] = node_name
                        g_node_types[node_id] = node_type
                        g_edge_types[k] = k
                        g_adj[topic_key_id][node_id] = k

                else:
                    good_graph = False

                    tmp_tmp_node_names = {}
                    tmp_tmp_node_types = {}
                    tmp_tmp_edge_types = {}
                    tmp_tmp_adj = defaultdict(dict)

                    for kk, vv in nbr_v.get('neighbors', {}).items(): # 2nd hop
                        # extended_rel_path = rel_path + [kk]
                        # if extended_rel_path in all_rel_paths: # Not used, we need context nodes as constraints for questions
                        #     continue
                        for nbr_nbr in vv:
                            if isinstance(nbr_nbr, str) or isinstance(nbr_nbr, bool) or isinstance(nbr_nbr, float):
                                # Choosing additional context nodes (serving as constraints in a query)
                                # that have overlaps with the query
                                overlapped_sub_string = get_text_overlap(query, str(nbr_nbr))
                                if len(overlapped_sub_string) > 0 or normalize_answer(str(nbr_nbr)) in all_answers:
                                    new_node_id = str(nbr_nbr)
                                    tmp_tmp_node_names[new_node_id] = str(nbr_nbr)

                                    if isinstance(nbr_nbr, str):
                                        tmp_tmp_node_types[new_node_id] = 'str'
                                    elif isinstance(nbr_nbr, bool):
                                        tmp_tmp_node_types[new_node_id] = 'bool'
                                    else:
                                        tmp_tmp_node_types[new_node_id] = 'num'

                                    tmp_tmp_edge_types[kk] = kk
                                    tmp_tmp_adj[node_id][new_node_id] = kk

                                    if normalize_answer(str(nbr_nbr)) in all_answers:
                                        good_graph = True

                            elif isinstance(nbr_nbr, dict):
                                new_node_id = list(nbr_nbr.keys())[0]
                                nbr_nbr_v = nbr_nbr[new_node_id]

                                selected_names = (nbr_nbr_v['name'] + nbr_nbr_v['alias'])[:1]
                                new_node_name = selected_names[0] if len(selected_names) > 0 else ''

                                # Choosing additional context nodes (serving as constraints in a query)
                                # that have overlaps with the query
                                overlapped_sub_string = get_text_overlap(query, new_node_name)
                                if len(overlapped_sub_string) > 0 or normalize_answer(new_node_name) in all_answers:
                                    tmp_tmp_node_names[new_node_id] = new_node_name

                                    selected_types = (nbr_nbr_v['notable_types'] + nbr_nbr_v['type'])[:1]
                                    tmp_tmp_node_types[new_node_id] = selected_types[0] if len(selected_types) > 0 else ''

                                    tmp_tmp_edge_types[kk] = kk
                                    tmp_tmp_adj[node_id][new_node_id] = kk

                                    if normalize_answer(new_node_name) in all_answers:
                                        good_graph = True
                            else:
                                raise RuntimeError('Unknown type: %s' % type(nbr_nbr))

                    if good_graph:
                        g_node_names[node_id] = node_name
                        g_node_types[node_id] = node_type
                        g_edge_types[k] = k
                        g_adj[topic_key_id][node_id] = k

                        g_node_names.update(tmp_tmp_node_names)
                        g_node_types.update(tmp_tmp_node_types)
                        g_edge_types.update(tmp_tmp_edge_types)
                        g_adj = update_adj(g_adj, tmp_tmp_adj)


    if len(g_adj) > 0:
        g_node_names[topic_key_id] = topic_key_name
        g_node_types[topic_key_id] = topic_key_type
        assert len(g_node_names) == len(g_node_types)

        subgraph = {}
        subgraph['g_node_names'] = g_node_names
        subgraph['g_node_types'] = g_node_types
        subgraph['g_edge_types'] = g_edge_types
        subgraph['g_adj'] = g_adj
        return subgraph

    else:
        return None


def update_adj(tmp_adj, tmp_tmp_adj):
    for tmp_node in tmp_tmp_adj:
        tmp_adj[tmp_node].update(tmp_tmp_adj[tmp_node])
    return tmp_adj



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_dir', required=True, type=str, help='path to the raw data dir')
    parser.add_argument('-o', '--out_dir', required=True, type=str, help='path to the output dir')
    args = parser.parse_args()


    freebase = load_ndjson(os.path.join(args.input_dir, 'freebase.json'), return_type='dict')
    for dtype in ('train', 'dev', 'test'):
        qa_data = load_json(os.path.join(args.input_dir, '{}.json'.format(dtype)))
        out_path = os.path.join(args.out_dir, '{}.json'.format(dtype))
        n_examples = prepare_webquestions(qa_data, freebase, out_path)
        print('Saved {} {} examples to {}'.format(n_examples, dtype, out_path))
