'''
Created on Oct, 2019

@author: hugo

'''
import os
import json
import argparse

from ..core.utils.io_utils import *


def fetch_meta(path):
    try:
        data = load_gzip_json(path)
    except:
        return {}
    content = {}
    properties = data['property']
    if '/type/object/name' in properties:
        content['name'] = [x['value'] for x in properties['/type/object/name']['values']]
    else:
        content['name'] = []
    if '/common/topic/alias' in properties:
        content['alias'] = [x['value'] for x in properties['/common/topic/alias']['values']]
    else:
        content['alias'] = []
    if '/common/topic/notable_types' in properties:
        content['notable_types'] = [x['id'] for x in properties['/common/topic/notable_types']['values']]
    else:
        content['notable_types'] = []
    if '/type/object/type' in properties:
        content['type'] = [x['id'] for x in properties['/type/object/type']['values']]
    else:
        content['type'] = []
    return content

def fetch(data, data_dir):
    if not 'id' in data:
        return data['value']
    mid = data['id']
    # meta data might not be in the subgraph, get it from target files
    meta = fetch_meta(os.path.join(data_dir, '{}.json.gz'.format(mid.strip('/').replace('/', '.'))))
    if meta == {}:
        if not 'property' in data:
            if 'text' in data:
                return data['text']
            else:
                import pdb;pdb.set_trace()
        properties = data['property']
        if '/type/object/name' in properties:
            meta['name'] = [x['value'] for x in properties['/type/object/name']['values']]
        else:
            meta['name'] = []
        if '/common/topic/alias' in properties:
            meta['alias'] = [x['value'] for x in properties['/common/topic/alias']['values']]
        else:
            meta['alias'] = []
        if '/common/topic/notable_types' in properties:
            meta['notable_types'] = [x['id'] for x in properties['/common/topic/notable_types']['values']]
        else:
            meta['notable_types'] = []
        if '/type/object/type' in properties:
            meta['type'] = [x['id'] for x in properties['/type/object/type']['values']]
        else:
            meta['type'] = []
    graph = {mid: meta}
    if not 'property' in data: # we stop at the 2nd hop
        return graph
    properties = data['property']
    neighbors = {}
    for k, v in properties.items():
        if k.startswith('/common') or k.startswith('/type') \
            or k.startswith('/freebase') or k.startswith('/user') \
            or k.startswith('/imdb'):
            continue
        if len(v['values']) > 0:
            neighbors[k] = []
            for nbr in v['values']:
                nbr_graph = fetch(nbr, data_dir)
                neighbors[k].append(nbr_graph)
    graph[mid]['neighbors'] = neighbors
    return graph



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-data_dir', '--data_dir', required=True, type=str, help='path to the data dir')
    parser.add_argument('-fbkeys', '--freebase_keys', required=True, type=str, help='path to the freebase key file')
    parser.add_argument('-out_dir', '--out_dir', type=str, required=True, help='path to the output dir')
    args = parser.parse_args()

    ids = load_json(args.freebase_keys)
    total = len(ids)
    print('Fetching {} entities and their 2-hop neighbors.'.format(total))
    print_bar_len = 50
    cnt = 0
    missing_ids = set()
    with open(os.path.join(args.out_dir, 'freebase.json'), 'a') as out_f:
        for id_ in ids:
            try:
                data = load_gzip_json(os.path.join(args.data_dir, '{}.json.gz'.format(id_)))
            except:
                missing_ids.add(id_)
                continue
            graph = fetch(data, args.data_dir)
            graph2 = {id_: list(graph.values())[0]}
            graph2[id_]['id'] = list(graph.keys())[0]
            line = json.dumps(graph2) + '\n'
            out_f.write(line)
            cnt += 1
            if cnt % int(total / print_bar_len) == 0:
                printProgressBar(cnt, total, prefix='Progress:', suffix='Complete', length=print_bar_len)
        printProgressBar(cnt, total, prefix='Progress:', suffix='Complete', length=print_bar_len)

    print('Missed %s mids' % len(missing_ids))
    # dump_json(list(missing_ids), os.path.join(args.out_dir, 'missing_fbids.json'))
