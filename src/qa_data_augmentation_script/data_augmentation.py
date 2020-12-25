import argparse
import random
import os
import json
import copy


def dump_ndjson(data, file):
    try:
        with open(file, 'w') as f:
            for each in data:
                f.write(json.dumps(each) + '\n')
    except Exception as e:
        raise e

def load_ndjson(file):
    data = []
    try:
        with open(file, 'r') as f:
            for line in f:
                data.append(json.loads(line.strip()))
    except Exception as e:
        raise e
    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', required=True, type=str, help='path to the data dir')
    parser.add_argument('-q', '--question', required=True, type=str, help='path to the data dir')
    parser.add_argument('-out_dir', '--out_dir', required=True, type=str, help='path to the output dir')
    opt = vars(parser.parse_args())

    data = load_ndjson(opt['input'])
    new_data = copy.deepcopy(data)
    print('{} examples in the original data'.format(len(data)))

    fin = open(opt['question'], 'r')
    questions = fin.readlines()
    assert len(data) == len(questions)
    for i, each in enumerate(new_data):
        each['outSeq'] = questions[i].strip()

    new_data = data + new_data
    random.seed(123)
    random.shuffle(new_data)
    random.shuffle(new_data)
    random.shuffle(new_data)
    random.shuffle(new_data)
    random.shuffle(new_data)
    dump_ndjson(new_data, os.path.join(opt['out_dir'], 'train.json'))
    print('{} examples in the augmented data'.format(len(new_data)))
