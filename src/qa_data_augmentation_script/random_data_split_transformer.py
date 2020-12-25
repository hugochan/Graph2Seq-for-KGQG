import argparse
import random
import os


def write_lines(data, path):
    with open(path, 'w') as fout:
        for line in data:
            fout.write(line)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', required=True, type=str, nargs='+', help='path to the input datasets')
    parser.add_argument('-out_dir', '--out_dir', required=True, type=str, help='path to the output dir')
    parser.add_argument('-prefix', '--prefix', required=True, type=str, help='prefix')
    parser.add_argument('-ratio', '--ratio', type=float, nargs='+', help='training data sampling ratio')
    opt = vars(parser.parse_args())

    random.seed(123)
    data = []
    for path in opt['input']:
        with open(path, 'r') as fin:
            data.extend(fin.readlines())

    assert len(opt['ratio']) == 2 and sum(opt['ratio']) < 1 and min(opt['ratio']) > 0

    train_ratio, dev_ratio = opt['ratio']
    n_train = int(len(data) * train_ratio)
    n_dev = int(len(data) * dev_ratio)

    random.shuffle(data)
    random.shuffle(data)
    random.shuffle(data)
    random.shuffle(data)
    random.shuffle(data)

    train_data = data[:n_train]
    dev_data = data[n_train:n_train + n_dev]
    test_data = data[n_train + n_dev:]
    write_lines(train_data, os.path.join(opt['out_dir'], '{}-train.txt'.format(opt['prefix'])))
    write_lines(dev_data, os.path.join(opt['out_dir'], '{}-val.txt'.format(opt['prefix'])))
    write_lines(test_data, os.path.join(opt['out_dir'], '{}-test.txt'.format(opt['prefix'])))
    print('total size: {}, train size: {}, dev size: {}, test size: {}'.format(len(data), len(train_data), len(dev_data), len(test_data)))
