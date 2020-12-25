import sys
import pickle as pkl
import json

def load_ndjson_to_dict(file):
    data = {}
    try:
        with open(file, 'r') as f:
            for line in f:
                data.update(json.loads(line.strip()))
    except Exception as e:
        raise e
    return data


if __name__ == '__main__':
    kg = load_ndjson_to_dict(sys.argv[1])

    mid2name = {}
    for _, graph in kg.items():
        selected_names = (graph['name'] + graph['alias'])[:1]
        if len(selected_names) > 0:
            mid2name[graph['id']] = selected_names

        for k, v in graph['neighbors'].items():
            for nbr in v:
                if isinstance(nbr, dict):
                    nbr_k = list(nbr.keys())[0]
                    nbr_v = nbr[nbr_k]
                    selected_names = (nbr_v['name'] + nbr_v['alias'])[:1]
                    if len(selected_names) > 0:
                        mid2name[nbr_k] = selected_names

                    if not 'neighbors' in nbr_v:
                        continue

                    for kk, vv in nbr_v['neighbors'].items(): # 2nd hop
                        for nbr_nbr in vv:
                            if isinstance(nbr_nbr, dict):
                                nbr_nbr_k = list(nbr_nbr.keys())[0]
                                nbr_nbr_v = nbr_nbr[nbr_nbr_k]
                                selected_names = (nbr_nbr_v['name'] + nbr_nbr_v['alias'])[:1]
                                if len(selected_names) > 0:
                                    mid2name[nbr_nbr_k] = selected_names

    print('len(mid2name)', len(mid2name))
    with open(sys.argv[2],'wb+') as f:
        pkl.dump(mid2name, f)

