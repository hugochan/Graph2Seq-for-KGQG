# -*- coding: utf-8 -*-
"""
Module to handle getting data loading classes and helper functions.
"""

import json
import re
import random
import io
from nltk.tokenize import wordpunct_tokenize, word_tokenize

import torch
import numpy as np
from scipy.sparse import *
from collections import Counter, defaultdict
from .timer import Timer

from .bert_utils import *
from . import padding_utils
from . import constants


tokenize = lambda s: wordpunct_tokenize(s)

def vectorize_input(batch, config, bert_model, training=True, device=None):
    # Check there is at least one valid example in batch (containing targets):
    if not batch:
        return None

    # Relevant parameters:
    batch_size = len(batch.sent1_word)

    src_seqs = torch.LongTensor(batch.sent1_word)
    src_lens = torch.LongTensor(batch.sent1_length)

    out_seqs = torch.LongTensor(batch.sent2_word)
    out_seq_lens = torch.LongTensor(batch.sent2_length)


    with torch.set_grad_enabled(training):
        example = {'batch_size': batch_size,
                   'in_graphs': None,
                   'src_seqs': src_seqs.to(device) if device else src_seqs,
                   'src_lens': src_lens.to(device) if device else src_lens,
                   'targets': out_seqs.to(device) if device else out_seqs,
                   'target_lens': out_seq_lens.to(device) if device else out_seq_lens,
                   'target_src': batch.sent2_src,
                   'oov_dict': batch.oov_dict}

        if config['f_ans'] or config.get('f_ans_pool', None):
            answers = torch.LongTensor(batch.answers)
            answer_lens = torch.LongTensor(batch.answer_lens)
            num_answers = torch.LongTensor(batch.num_answers)
            example['answers'] = answers.to(device) if device else answers
            example['answer_lens'] = answer_lens.to(device) if device else answer_lens
            example['num_answers'] = num_answers.to(device) if device else num_answers

        return example


def prepare_datasets(config):
    if config['trainset'] is not None:
        train_set, train_src_len, train_tgt_len = load_data(config['trainset'], isLower=True)
        print('# of training examples: {}'.format(len(train_set)))
        print('Training source sentence length: {}'.format(train_src_len))
        print('Training target sentence length: {}'.format(train_tgt_len))
    else:
        train_set = None

    if config['devset'] is not None:
        dev_set, dev_src_len, dev_tgt_len = load_data(config['devset'], isLower=True)
        print('# of dev examples: {}'.format(len(dev_set)))
        print('Dev source sentence length: {}'.format(dev_src_len))
        print('Dev target sentence length: {}'.format(dev_tgt_len))
    else:
        dev_set = None

    if config['testset'] is not None:
        test_set, test_src_len, test_tgt_len = load_data(config['testset'], isLower=True)
        print('# of testing examples: {}'.format(len(test_set)))
        print('Testing source sentence length: {}'.format(test_src_len))
        print('Testing target sentence length: {}'.format(test_tgt_len))
    else:
        test_set = None
    return {'train': train_set, 'dev': dev_set, 'test': test_set}


def load_data(inpath, isLower=True):
    all_instances = []
    src_len = []
    tgt_len = []

    with open(inpath, 'r') as f:
        for line in f:
            line = line.strip()
            instance = json.loads(line)

            all_instances.append([Sequence(instance['inSeq'], isLower=isLower), Sequence(instance['outSeq'], isLower=isLower, end_sym=constants._EOS_TOKEN), [Sequence(x, isLower=isLower) for x in instance['answers']]])
            src_len.append(len(all_instances[-1][0].words))
            tgt_len.append(len(all_instances[-1][1].words))

    src_len_stats = {'min': np.min(src_len), 'max': np.max(src_len), 'mean': np.mean(src_len)}
    tgt_len_stats = {'min': np.min(tgt_len), 'max': np.max(tgt_len), 'mean': np.mean(tgt_len)}
    return all_instances, src_len_stats, tgt_len_stats


class DataStream(object):
    def __init__(self, all_instances, word_vocab, edge_vocab, POS_vocab=None, NER_vocab=None, config=None,
                 isShuffle=False, isLoop=False, isSort=True, batch_size=-1, ext_vocab=False, bert_tokenizer=None):
        self.config = config
        if batch_size == -1: batch_size = config['batch_size']
        # sort instances based on length
        if isSort:
            all_instances = sorted(all_instances, key=lambda question: (len(question[0].words), len(question[1].words)))
        else:
            random.shuffle(all_instances)
            random.shuffle(all_instances)
        self.num_instances = len(all_instances)

        # distribute questions into different buckets
        batch_spans = padding_utils.make_batches(self.num_instances, batch_size)
        self.batches = []
        for batch_index, (batch_start, batch_end) in enumerate(batch_spans):
            cur_instances = all_instances[batch_start: batch_end]
            cur_batch = InstanceBatch(cur_instances, config, word_vocab, ext_vocab=ext_vocab)
            self.batches.append(cur_batch)

        self.num_batch = len(self.batches)
        self.index_array = np.arange(self.num_batch)
        self.isShuffle = isShuffle
        if self.isShuffle: np.random.shuffle(self.index_array)
        self.isLoop = isLoop
        self.cur_pointer = 0

    def nextBatch(self):
        if self.cur_pointer >= self.num_batch:
            if not self.isLoop: return None
            self.cur_pointer = 0
            if self.isShuffle: np.random.shuffle(self.index_array)
        cur_batch = self.batches[self.index_array[self.cur_pointer]]
        self.cur_pointer += 1
        return cur_batch

    def reset(self):
        if self.isShuffle: np.random.shuffle(self.index_array)
        self.cur_pointer = 0

    def get_num_batch(self):
        return self.num_batch

    def get_num_instance(self):
        return self.num_instances

    def get_batch(self, i):
        if i >= self.num_batch: return None
        return self.batches[i]

class InstanceBatch(object):
    def __init__(self, instances, config, word_vocab, ext_vocab=False):
        self.instances = instances
        self.batch_size = len(instances)
        self.oov_dict = None # out-of-vocabulary dict

        self.has_sent3 = False
        if instances[0][2] is not None: self.has_sent3 = True

        # Create word representation and length
        self.sent1_word = [] # [batch_size, sent1_len]
        self.sent1_length = [] # [batch_size]

        self.sent2_src = []
        self.sent2_word = [] # [batch_size, sent2_len]
        self.sent2_length = [] # [batch_size]


        if config['f_ans']:
            self.answers = [] # [batch_size, num_answers, seq3_len]
            self.answer_lens = [] # [batch_size, num_answers]
            self.num_answers = [] # [batch_size]


        if ext_vocab:
            base_oov_idx = len(word_vocab)
            self.oov_dict = OOVDict(base_oov_idx)


        for i, (sent1, sent2, sent3) in enumerate(instances):
            sent1_idx = []
            for word in sent1.words:
                idx = word_vocab.getIndex(word)
                if ext_vocab and idx == word_vocab.UNK:
                    idx = self.oov_dict.add_word(i, word)
                sent1_idx.append(idx)
            self.sent1_word.append(sent1_idx)
            self.sent1_length.append(len(sent1.words))


            sent2_idx = []
            for word in sent2.words:
                idx = word_vocab.getIndex(word)
                if ext_vocab and idx == word_vocab.UNK:
                    idx = self.oov_dict.word2index.get((i, word), idx)
                sent2_idx.append(idx)

            self.sent2_word.append(sent2_idx)
            self.sent2_src.append(sent2.src)
            self.sent2_length.append(len(sent2.words))


            if config['f_ans']:
                tmp_answer_idx = []
                tmp_answer_lens = []
                for answer in seq3:
                    tmp_answer_idx.append([word_vocab.getIndex(word) for word in answer.words])
                    tmp_answer_lens.append(max(len(answer.words), 1))

                self.answers.append(tmp_answer_idx)
                self.answer_lens.append(tmp_answer_lens)
                self.num_answers.append(len(seq3))



        self.sent1_word = padding_utils.pad_2d_vals_no_size(self.sent1_word)

        self.sent1_length = np.array(self.sent1_length, dtype=np.int32)
        self.sent2_word = padding_utils.pad_2d_vals_no_size(self.sent2_word)
        self.sent2_length = np.array(self.sent2_length, dtype=np.int32)

        if config['f_ans']:
            self.answers = padding_utils.pad_3d_vals_no_size(self.answers, fills=word_vocab.PAD)
            self.answer_lens = padding_utils.pad_2d_vals_no_size(self.answer_lens, fills=1)
            self.num_answers = np.array(self.num_answers, dtype=np.int32)


class Sequence(object):
    def __init__(self, data, isLower=False, end_sym=None):
        self.src = data
        self.tokText = self.src

        if isLower:
            self.src = self.src.lower()
            self.tokText = self.tokText.lower()

        self.src = ' '.join(tokenize(self.src))
        self.words = tokenize(self.tokText)

        # it's the output sequence
        if end_sym != None:
            self.tokText += ' ' + end_sym
            self.words.append(end_sym)



class OOVDict(object):
    def __init__(self, base_oov_idx):
        self.word2index = {}  # type: Dict[Tuple[int, str], int]
        self.index2word = {}  # type: Dict[Tuple[int, int], str]
        self.next_index = {}  # type: Dict[int, int]
        self.base_oov_idx = base_oov_idx
        self.ext_vocab_size = base_oov_idx

    def add_word(self, idx_in_batch, word) -> int:
        key = (idx_in_batch, word)
        index = self.word2index.get(key, None)
        if index is not None: return index
        index = self.next_index.get(idx_in_batch, self.base_oov_idx)
        self.next_index[idx_in_batch] = index + 1
        self.word2index[key] = index
        self.index2word[(idx_in_batch, index)] = word
        self.ext_vocab_size = max(self.ext_vocab_size, index + 1)
        return index



