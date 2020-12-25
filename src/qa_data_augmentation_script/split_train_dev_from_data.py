import argparse
import random
import os
import json


def load_ndjson(file):
    data = []
    try:
        with open(file, 'r') as f:
            for line in f:
                data.append(json.loads(line.strip()))
    except Exception as e:
        raise e
    return data

def dump_ndjson(data, file):
    try:
        with open(file, 'w') as f:
            for each in data:
                f.write(json.dumps(each) + '\n')
    except Exception as e:
        raise e


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', required=True, type=str, help='path to the data dir')
    parser.add_argument('-out_dir', '--out_dir', required=True, type=str, help='path to the output dir')
    parser.add_argument('-train_ratio', '--train_ratio', default=0.8, type=float, help='training data sampling ratio')
    opt = vars(parser.parse_args())

    random.seed(123)
    data = load_ndjson(opt['input'])
    train_ratio = min(1, opt['train_ratio'])
    n_train = int(len(data) * train_ratio)
    random.shuffle(data)
    random.shuffle(data)
    random.shuffle(data)
    random.shuffle(data)
    random.shuffle(data)
    train_data = data[:n_train]
    dev_data = data[n_train:]
    dump_ndjson(train_data, os.path.join(opt['out_dir'], 'train.json'))
    dump_ndjson(dev_data, os.path.join(opt['out_dir'], 'dev.json'))
    print('total size: {}, train size: {}, dev size: {}'.format(len(data), len(train_data), len(dev_data)))
