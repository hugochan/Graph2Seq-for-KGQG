import argparse
import random
import os
import json


def write_lines(data, path):
    with open(path, 'w') as fout:
        for line in data:
            fout.write(line)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', required=True, type=str, help='path to the data')
    parser.add_argument('-out_dir', '--out_dir', required=True, type=str, help='path to the output dir')
    parser.add_argument('-sample_ratio', '--sample_ratio', type=float, help='training data sampling ratio')
    parser.add_argument('-train_ratio', '--train_ratio', default=0.8, type=float, help='training data sampling ratio')
    parser.add_argument('-prefix', '--prefix', required=True, type=str, help='prefix')
    opt = vars(parser.parse_args())

    with open(opt['input'], 'r') as fin:
        data = fin.readlines()

    random.seed(6666)
    sample_ratio = min(1, opt['sample_ratio'])
    n_examples = int(len(data) * sample_ratio)
    data = list(enumerate(data))
    random.shuffle(data)
    random.shuffle(data)
    random.shuffle(data)
    random.shuffle(data)
    random.shuffle(data)
    sampled_train_ids, data = list(zip(*data[:n_examples]))
    data = list(data)
    write_lines(data, os.path.join(opt['out_dir'], 'sampled_{}_train_{}.txt'.format(opt['prefix'], sample_ratio)))
    print('randomly sampled {}% training data'.format(sample_ratio * 100))



    random.seed(123)
    train_ratio = min(1, opt['train_ratio'])
    n_train = int(len(data) * train_ratio)
    random.shuffle(data)
    random.shuffle(data)
    random.shuffle(data)
    random.shuffle(data)
    random.shuffle(data)
    train_data = data[:n_train]
    dev_data = data[n_train:]
    write_lines(train_data, os.path.join(opt['out_dir'], '{}-train.txt'.format(opt['prefix'])))
    write_lines(dev_data, os.path.join(opt['out_dir'], '{}-val.txt'.format(opt['prefix'])))
    print('total size: {}, train size: {}, dev size: {}'.format(len(data), len(train_data), len(dev_data)))

