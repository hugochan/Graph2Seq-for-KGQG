import os
import json
import argparse
import numpy as np



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-emb', '--emb_dir', required=True, type=str, help='path to the input dir')
    parser.add_argument('-dict', '--dict_dir', required=True, type=str, help='path to the input dir')
    parser.add_argument('-ent_vocab', '--ent_vocab', required=True, type=str, help='path to the input dir')
    parser.add_argument('-rel_vocab', '--rel_vocab', required=True, type=str, help='path to the input dir')
    parser.add_argument('-o', '--output_dir', required=True, type=str, help='path to the output dir')
    opt = vars(parser.parse_args())

    num_ents = 54865939
    num_rels = 10024
    emb_dim = 50
    scale = 0.1

    # Relation
    rel2id = {}
    with open(os.path.join(opt['dict_dir'], 'relation2id.txt'), 'r') as f:
        f.readline()
        for line in f:
            rel, id_ = line.split('\t')
            rel = '/' + rel.replace('.', '/')
            rel2id[rel] = id_
    rel2vec = np.memmap(os.path.join(opt['emb_dir'], 'relation2vec.bin'), dtype='float32', mode='r', shape=(num_rels, emb_dim))

    rel_vocab = json.load(open(opt['rel_vocab'], 'r'))
    hit_ratio = 0
    with open(os.path.join(opt['output_dir'], 'rel_emb.ndjson'), 'w') as f:
        for rel in rel_vocab:
            if rel in rel2id and int(rel2id[rel]) < len(rel2vec):
                emb = rel2vec[int(rel2id[rel])]
                hit_ratio += 1
            else:
                emb = np.array(np.random.uniform(low=-scale, high=scale, size=(emb_dim,)), dtype=np.float32)
            # f.write(json.dumps({'id':rel, 'vec':emb.tolist()}) + '\n')
            f.write(json.dumps({rel:emb.tolist()}) + '\n')
    print('hit_ratio: {}'.format(hit_ratio / len(rel_vocab)))



    # Entity
    ent2id = {}
    with open(os.path.join(opt['dict_dir'], 'entity2id.txt'), 'r') as f:
        f.readline()
        for line in f:
            ent, id_ = line.split('\t')
            ent = '/' + ent.replace('.', '/')
            ent2id[ent] = id_
    ent2vec = np.memmap(os.path.join(opt['emb_dir'], 'entity2vec.bin'), dtype='float32', mode='r', shape=(num_ents, emb_dim))

    ent_vocab = json.load(open(opt['ent_vocab'], 'r'))
    hit_ratio = 0
    with open(os.path.join(opt['output_dir'], 'ent_emb.ndjson'), 'w') as f:
        for ent in ent_vocab:
            if ent in ent2id and int(ent2id[ent]) < len(ent2vec):
                emb = ent2vec[int(ent2id[ent])]
                hit_ratio += 1
            else:
                emb = np.array(np.random.uniform(low=-scale, high=scale, size=(emb_dim,)), dtype=np.float32)
            # f.write(json.dumps({'id':ent, 'vec':emb.tolist()}) + '\n')
            f.write(json.dumps({ent:emb.tolist()}) + '\n')
    print('hit_ratio: {}'.format(hit_ratio / len(ent_vocab)))

