import os
import pickle as pkl
import argparse
import re
import json
import nltk
import numpy as np
from collections import defaultdict
# import matplotlib.pyplot as plt

# Train/test/dev: 18989/2000/2000



def build_mid2ent(in_dir):
    mid2ent = {}
    with open(os.path.join(in_dir, 'ComplexWebQuestions_train.json'),'r') as f:
        ComplexWebQuestions_train = json.load(f)

    # with open(os.path.join(in_dir, 'ComplexWebQuestions_test.json'),'r') as f:
    #     ComplexWebQuestions_test = json.load(f)

    with open(os.path.join(in_dir, 'ComplexWebQuestions_dev.json'),'r') as f:
        ComplexWebQuestions_dev = json.load(f)

    for dct in ComplexWebQuestions_train + ComplexWebQuestions_dev:
        for ans in dct['answers']:
            if ans['answer'] != None:
                mid2ent['/'+ans['answer_id'].replace('.','/')]=[ans['answer']]

    return mid2ent

def generate_answer_nl(in_dir, out_dir):
    with open(os.path.join(in_dir, 'entity_dict.txt'),'r') as f:
        entity_dict = f.readlines()

    with open(os.path.join(in_dir, 'subgraph_answer.txt'),'r') as f:
        subgraph_answer = f.readlines()

    node1 = []
    node2 = []
    edge = []
    ans_list = []
    for line in subgraph_answer:
        line = line.replace('￨O','')
        line = line.replace('￨A','|A')
        triples = line.split('<t>')
        subjs = ""
        objs = ""
        preds = ""
        ans = []
        for triple in triples:
            subj, pred, obj = triple.strip().split()
            if '|A' in subj:
                ans.append(int(subj[:-3]))
            if '|A' in obj:
                ans.append(int(obj[:-3]))
        ans_list.append(list(set(ans)))

    new_entity_dict = {}
    for line in entity_dict:
        line = line.strip()
        k,v = line.split('\t')
        new_entity_dict[int(k)] = '/'+v.replace('.','/')

    mid2ent_hand = build_mid2ent(in_dir)

    with open(os.path.join(in_dir, 'mid2ents.pkl'),'rb') as f3:
        mid2ents = pkl.load(f3)

    with open(os.path.join(in_dir, 'mid2ents_extra.pkl'),'rb') as f4:
        mid2ents_extra = pkl.load(f4)

    merge_mid2ents = {**mid2ent_hand,**mid2ents,**mid2ents_extra}
    print('Num of merged mid2ents: {}'.format(len(merge_mid2ents)))

    new_ans_list = []
    miss1=0
    miss2=0
    cnt=0
    cnt_tmp_ans=0
    for ans in ans_list:
        cnt+=len(ans)
        tmp = []
        tmp_id = []
        for x in ans:
            if x==0:
                miss1+=1
                continue

            tmp_id.append(new_entity_dict[x])
            if new_entity_dict[x] in merge_mid2ents.keys():
                tmp.append(merge_mid2ents[new_entity_dict[x]][0])
            else:
                miss2+=1
        new_ans_list.append([tmp, tmp_id])
        if len(tmp)==0:
            cnt_tmp_ans+=1

    # with open(os.path.join(out_dir, 'answer_list.txt'),'w+') as f:
    #     for line, line_id in new_ans_list:
    #         f.write(' | '.join(line).lower()+'\n')

    print('generate_answer_nl')
    print('miss1', miss1)
    print('miss2', miss2)
    print('cnt', cnt)
    print('cnt_tmp_ans', cnt_tmp_ans)
    return new_ans_list


def build_subgraph_nl(in_dir, out_dir):
    mid2ent_hand = build_mid2ent(in_dir)

    # with open('subgraph_answer.txt','r') as f1:
    #     subgraph_answer = f1.readlines()

    with open(os.path.join(in_dir, 'subgraph.txt'),'r') as f2:
        subgraphf = f2.readlines()


    with open(os.path.join(in_dir, 'mid2ents.pkl'),'rb') as f3:
        mid2ents = pkl.load(f3)


    with open(os.path.join(in_dir, 'mid2ents_extra.pkl'),'rb') as f4:
        mid2ents_extra = pkl.load(f4)

    merge_mid2ents = {**mid2ent_hand,**mid2ents,**mid2ents_extra}
    print('Num of merged mid2ents: {}'.format(len(merge_mid2ents)))


    new_subgraph = []

    miss_cnt = 0
    miss_list=[]
    all_ents = []
    num_triples = []
    formatted_subgraphs = []

    for idx, line in enumerate(subgraphf):
        triple_list = line.strip().split('<t>')
        new_triple_list = []
        tmp_dict={}
        unk_ent_cnt=1
        unk_ent_dict = {}

        g_node_names = {}
        # g_node_types = {}
        g_edge_types = {}
        g_adj = defaultdict(dict)

        for triple in triple_list:
            # if triple=='' or len(triple.split())!=3:
            if triple=='':
                break

            elements = triple.split()
            assert 'm.' in elements[0] or 'g.' in elements[0]

            if len(elements)!=3:
                subj = elements[0]
                pred = elements[1]
                obj = ' '.join(elements[2:])
            else:
                subj, pred, obj = elements

            all_ents.append(subj)
            all_ents.append(obj)
            subj = '/'+subj.replace('.','/')
            if '/m/' in subj or '/g/' in subj:
                if subj in tmp_dict.keys() or subj in merge_mid2ents.keys():
                    tmp_dict[subj]=merge_mid2ents[subj][0]
                    subj_name = tmp_dict[subj]
                else:
                    miss_cnt+=1
                    miss_list.append(subj)
                    if subj in unk_ent_dict.keys():
                        subj_name = unk_ent_dict[subj]
                    else:
                        # unk_ent_dict[subj] = 'unk_ent_'+str(unk_ent_cnt)
                        unk_ent_dict[subj] = 'none'
                        subj_name = unk_ent_dict[subj]
                        unk_ent_cnt += 1

            pred = '/' + pred.replace('.','/')

            if 'm.' in obj or 'g.' in obj:
                obj = '/' + obj.replace('.', '/')
                if obj in tmp_dict.keys() or obj in merge_mid2ents.keys():
                    tmp_dict[obj] = merge_mid2ents[obj][0]
                    obj_name = tmp_dict[obj]
                else:
                    miss_cnt += 1
                    miss_list.append(obj)

                    if obj in unk_ent_dict.keys():
                        obj_name = unk_ent_dict[obj]
                    else:
                        # unk_ent_dict[obj] = 'unk_ent_'+str(unk_ent_cnt)
                        unk_ent_dict[obj] = 'none'
                        obj_name = unk_ent_dict[obj]
                        unk_ent_cnt += 1
            else:
                obj = ' '.join(obj.split('-'))
                obj_name = obj

            new_triple = [subj_name, pred, obj_name]
            g_node_names.update({subj: subj_name, obj: obj_name})
            g_edge_types[pred] = pred
            g_adj[subj][obj] = pred

            new_triple_list.append(new_triple)
        subgraph = {'g_node_names': g_node_names,
                    'g_edge_types': g_edge_types,
                    'g_adj': g_adj}

        assert len(g_adj) > 0

        formatted_subgraphs.append(subgraph)
        new_subgraph.append(new_triple_list)
        num_triples.append(len(new_triple_list))

    with open(os.path.join(out_dir, 'subgraph_nl.pkl'),'wb+') as f:
        pkl.dump(new_subgraph,f)

    print('build_subgraph_nl')
    all_ents=set(all_ents)
    print('miss_cnt', miss_cnt)
    print('miss_list', len(set(miss_list)))
    print('all_ents', len(all_ents))
    print('sum(num_triples)/len(num_triples)', sum(num_triples)/len(num_triples))
    print('# of triples: min: {}, max: {}, mean: {}'.format(np.min(num_triples), np.max(num_triples), np.mean(num_triples)))
    return formatted_subgraphs


def process_querys(in_dir, out_dir):
    with open(os.path.join(in_dir, 'querys.txt'),'r') as f:
        querys = f.readlines()

    with open(os.path.join(in_dir, 'ent_dict_list.txt'),'r') as f:
        ent_dict_list = f.readlines()

    new_ent_dict_list = []
    for line in ent_dict_list:
        try:
            new_line = eval(line)
        except:
            a=0
        new_ent_dict_list.append(new_line)

    new_queries = []
    for idx, line in enumerate(querys):
        for k,v in new_ent_dict_list[idx].items():
            line = line.replace(k,v[0])

        line = ' '.join(nltk.word_tokenize(line.lower()))
        new_queries.append(line)

    # with open(os.path.join(out_dir, 'new_queries.txt'),'w+') as f:
    #     for line in new_queries:
    #         line = ' '.join(nltk.word_tokenize(line))+'\n'
    #         f.write(line)
    return new_queries


def prepare_output_data(subgraphs, answer_lists, queries, out_dir):
    count = 0
    with open(os.path.join(out_dir, 'data.json'), 'w') as outf:
        for i in range(len(subgraphs)):
            example = {}
            example['answers'] = answer_lists[i][0]
            example['answer_ids'] = answer_lists[i][1]
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


    new_subgraph = build_subgraph_nl(opt['input_dir'], opt['output_dir'])
    new_ans_list = generate_answer_nl(opt['input_dir'], opt['output_dir'])
    new_queries = process_querys(opt['input_dir'], opt['output_dir'])
    assert len(new_subgraph) == len(new_ans_list) == len(new_queries)

    prepare_output_data(new_subgraph, new_ans_list, new_queries, opt['output_dir'])



