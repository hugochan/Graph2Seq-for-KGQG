# -*- coding: utf-8 -*-
"""
Module to handle getting data loading classes and helper functions.
"""

import json
import re
import io
import random
import string
from collections import Counter, defaultdict, OrderedDict
from nltk.tokenize import wordpunct_tokenize, word_tokenize

import numpy as np
from scipy.sparse import *
import torch

from .timer import Timer
from .bert_utils import *
from .generic_utils import normalize_answer
from . import padding_utils
from . import constants


# tokenize = lambda s: wordpunct_tokenize(re.sub('[%s]' % re.escape(string.punctuation), ' ', s))
# tokenize = lambda s: word_tokenize(s)
tokenize = lambda s: wordpunct_tokenize(s)


def vectorize_input(batch, config, bert_model, training=True, device=None):
    # Check there is at least one valid example in batch (containing targets):
    if not batch:
        return None

    # Relevant parameters:
    batch_size = len(batch.out_seqs)

    in_graphs = {}
    for k, v in batch.in_graphs.items():
        if k in ['node2edge', 'edge2node', 'max_num_graph_nodes']:
            in_graphs[k] = v
        else:
            in_graphs[k] = torch.LongTensor(v).to(device) if device else torch.LongTensor(v)

    out_seqs = torch.LongTensor(batch.out_seqs)
    out_seq_lens = torch.LongTensor(batch.out_seq_lens)


    with torch.set_grad_enabled(training):
        example = {'batch_size': batch_size,
                   'in_graphs': in_graphs,
                   'targets': out_seqs.to(device) if device else out_seqs,
                   'target_lens': out_seq_lens.to(device) if device else out_seq_lens,
                   'target_src': batch.out_seq_src,
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
    levi_graph = config.get('levi_graph', True)

    if config['trainset'] is not None:
        train_set, train_seq_lens = load_data(config['trainset'], isLower=True, levi_graph=levi_graph)
        print('# of training examples: {}'.format(len(train_set)))
        print('[ Max training seq length: {} ]'.format(np.max(train_seq_lens)))
        print('[ Min training seq length: {} ]'.format(np.min(train_seq_lens)))
        print('[ Mean training seq length: {} ]'.format(int(np.mean(train_seq_lens))))
    else:
        train_set = None

    if config['devset'] is not None:
        dev_set, dev_seq_lens = load_data(config['devset'], isLower=True, levi_graph=levi_graph)
        print('# of dev examples: {}'.format(len(dev_set)))
        print('[ Max dev seq length: {} ]'.format(np.max(dev_seq_lens)))
        print('[ Min dev seq length: {} ]'.format(np.min(dev_seq_lens)))
        print('[ Mean dev seq length: {} ]'.format(int(np.mean(dev_seq_lens))))
    else:
        dev_set = None

    if config['testset'] is not None:
        test_set, test_seq_lens = load_data(config['testset'], isLower=True, levi_graph=levi_graph)
        print('# of testing examples: {}'.format(len(test_set)))
        print('[ Max testing seq length: {} ]'.format(np.max(test_seq_lens)))
        print('[ Min testing seq length: {} ]'.format(np.min(test_seq_lens)))
        print('[ Mean testing seq length: {} ]'.format(int(np.mean(test_seq_lens))))
    else:
        test_set = None

    return {'train': train_set, 'dev': dev_set, 'test': test_set}


def load_data(inpath, isLower=True, levi_graph=True):
    all_instances = []
    all_seq_lens = []

    with open(inpath, 'r') as f:
        for line in f:
            line = line.strip()
            jo = json.loads(line, object_pairs_hook=OrderedDict)
            assert len(jo['inGraph']['g_adj']) > 0


            topic_entity_name = jo['inGraph']['g_node_names'][jo['topicEntityID']] if jo.get('topicEntityID', None) else ''
            answers = jo['answers']
            g_node_ans = set(jo.get('answer_ids', []))
            normalized_answers = {normalize_answer(x) for x in answers}
            out_seq = jo['outSeq']

            graph = {'g_node_ids': {}, 'g_node_name_words': [], 'g_node_type_words': [], 'g_node_type_ids': [], 'g_node_ans_match': [], 'g_edge_type_words': [], 'g_edge_type_ids': [], 'g_adj': defaultdict(dict)}
            for idx, nid in enumerate(jo['inGraph']['g_node_names']):
                graph['g_node_ids'][nid] = idx
                graph['g_node_name_words'].append(jo['inGraph']['g_node_names'][nid])

                if 'g_node_types' in jo['inGraph']:
                    graph['g_node_type_words'].append(' '.join(jo['inGraph']['g_node_types'][nid].split('/')[-1].split('_')))
                    graph['g_node_type_ids'].append(jo['inGraph']['g_node_types'][nid])


                if len(g_node_ans) > 0:
                    graph['g_node_ans_match'].append(1 if nid in g_node_ans else 2)
                else:
                    graph['g_node_ans_match'].append(1 if normalize_answer(jo['inGraph']['g_node_names'][nid]) in normalized_answers else 2)


            if levi_graph:
                # Levi Graph Transformation
                # We treat all edges in the original graph as new nodes and add new edges connecting original nodes and new nodes
                g_num_nodes = len(graph['g_node_ids'])
                edge_index = g_num_nodes
                virtual_edge_index = 0
                for nid, val in jo['inGraph']['g_adj'].items():
                    idx1 = graph['g_node_ids'][nid]
                    for nid2, edge_id_list in val.items():
                        idx2 = graph['g_node_ids'][nid2]
                        edge_id_list = [edge_id_list] if isinstance(edge_id_list, str) else edge_id_list
                        # We treat an edge as an additional node in a graph, hence
                        # a node pair (idx1, edge_index, idx2) can be represented as idx1 -> edge_index -> idx2
                        for edge_id in edge_id_list:
                            graph['g_adj'][idx1][edge_index] = virtual_edge_index
                            virtual_edge_index += 1
                            graph['g_adj'][edge_index][idx2] = virtual_edge_index
                            virtual_edge_index += 1

                            graph['g_edge_type_words'].append(' '.join(jo['inGraph']['g_edge_types'][edge_id].split('/')[-1].split('_')))
                            graph['g_edge_type_ids'].append(jo['inGraph']['g_edge_types'][edge_id])
                            edge_index += 1

                assert len(graph['g_edge_type_words']) == edge_index - g_num_nodes

                graph['num_virtual_nodes'] = edge_index
                graph['num_virtual_edges'] = virtual_edge_index
            else:

                edge_index = 0
                for nid, val in jo['inGraph']['g_adj'].items():
                    idx1 = graph['g_node_ids'][nid]
                    for nid2, edge_id_list in val.items():
                        idx2 = graph['g_node_ids'][nid2]
                        edge_id_list = [edge_id_list] if isinstance(edge_id_list, str) else edge_id_list
                        for edge_id in edge_id_list:
                            graph['g_adj'][idx1][idx2] = edge_index

                            graph['g_edge_type_words'].append(' '.join(jo['inGraph']['g_edge_types'][edge_id].split('/')[-1].split('_')))
                            graph['g_edge_type_ids'].append(jo['inGraph']['g_edge_types'][edge_id])
                            edge_index += 1

                assert len(graph['g_edge_type_words']) == edge_index

            all_instances.append([Sequence(graph, is_graph=True, isLower=isLower), Sequence(out_seq, isLower=isLower, end_sym=constants._EOS_TOKEN), [Sequence(x, isLower=isLower) for x in answers]])
            all_seq_lens.append(len(all_instances[-1][1].words))
    return all_instances, all_seq_lens



class DataStream(object):
    def __init__(self, all_instances, word_vocab, node_vocab, node_type_vocab, edge_type_vocab, config=None,
                 isShuffle=False, isLoop=False, isSort=True, batch_size=-1, ext_vocab=False, bert_tokenizer=None):

        self.config = config
        if batch_size == -1: batch_size = config['batch_size']
        # sort instances based on length
        if isSort:
            all_instances = sorted(all_instances, key=lambda instance: len(instance[0].graph['g_node_name_words']))

        self.num_instances = len(all_instances)

        # distribute questions into different buckets
        batch_spans = padding_utils.make_batches(self.num_instances, batch_size)
        self.batches = []
        for batch_index, (batch_start, batch_end) in enumerate(batch_spans):
            cur_instances = all_instances[batch_start: batch_end]
            cur_batch = InstanceBatch(cur_instances, config, word_vocab, node_vocab, node_type_vocab, edge_type_vocab, ext_vocab=ext_vocab)
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
    def __init__(self, instances, config, word_vocab, node_vocab, node_type_vocab, edge_type_vocab, ext_vocab=False):
        self.instances = instances
        self.batch_size = len(instances)
        self.oov_dict = None # out-of-vocabulary dict

        # Create word representation and length
        self.out_seq_src = []
        self.out_seqs = [] # [batch_size, seq2_len]
        self.out_seq_lens = [] # [batch_size]

        if config['f_ans'] or config.get('f_ans_pool', None):
            self.answers = [] # [batch_size, num_answers, seq3_len]
            self.answer_lens = [] # [batch_size, num_answers]
            self.num_answers = [] # [batch_size]
            if config['use_bert']:
                self.answer_bert = []

        if ext_vocab:
            base_oov_idx = len(word_vocab)
            self.oov_dict = OOVDict(base_oov_idx)


        batch_graph = [each[0].graph for each in instances]
        # Build graph
        self.in_graphs = vectorize_batch_graph(batch_graph, word_vocab, node_vocab, node_type_vocab,\
                             edge_type_vocab, self.oov_dict, kg_emb=config['kg_emb'], f_node_type=config.get('f_node_type', False), ext_vocab=ext_vocab)



        for i, (_, seq2, seq3) in enumerate(instances):
            if ext_vocab:
                seq2_idx = seq2ext_vocab_id(i, seq2.words, word_vocab, self.oov_dict)
            else:
                seq2_idx = []
                for word in seq2.words:
                    idx = word_vocab.getIndex(word)
                    seq2_idx.append(idx)

            self.out_seq_src.append(seq2.src)
            self.out_seqs.append(seq2_idx)
            self.out_seq_lens.append(len(seq2_idx))

            if config['f_ans'] or config.get('f_ans_pool', None):
                tmp_answer_idx = []
                tmp_answer_lens = []
                tmp_answer_bert = []
                for answer in seq3:
                    tmp_answer_idx.append([word_vocab.getIndex(word) for word in answer.words])
                    tmp_answer_lens.append(max(len(answer.words), 1))

                    if config['use_bert']:
                        bert_answer_features = convert_text_to_bert_features(answer.words, bert_tokenizer, config['bert_max_seq_len'], config['bert_doc_stride'])
                        tmp_answer_bert.append(bert_answer_features)

                self.answers.append(tmp_answer_idx)
                self.answer_lens.append(tmp_answer_lens)
                self.num_answers.append(len(seq3))

                if config['use_bert']:
                    self.answer_bert.append(tmp_answer_bert)

        self.out_seqs = padding_utils.pad_2d_vals_no_size(self.out_seqs, fills=word_vocab.PAD)
        self.out_seq_lens = np.array(self.out_seq_lens, dtype=np.int32)

        if config['f_ans'] or config.get('f_ans_pool', None):
            self.answers = padding_utils.pad_3d_vals_no_size(self.answers, fills=word_vocab.PAD)
            self.answer_lens = padding_utils.pad_2d_vals_no_size(self.answer_lens, fills=1)
            self.num_answers = np.array(self.num_answers, dtype=np.int32)


class Sequence(object):
    def __init__(self, data, is_graph=False, isLower=False, end_sym=None):
        self.graph = data if is_graph else None
        if is_graph:
            if isLower:
                self.graph['g_node_name_words'] = [tokenize(each.lower()) for each in self.graph['g_node_name_words']]
                self.graph['g_node_type_words'] = [tokenize(each.lower()) for each in self.graph['g_node_type_words']]
                self.graph['g_edge_type_words'] = [tokenize(each.lower()) for each in self.graph['g_edge_type_words']]
            else:
                self.graph['g_node_name_words'] = [tokenize(each) for each in self.graph['g_node_name_words']]
                self.graph['g_node_type_words'] = [tokenize(each) for each in self.graph['g_node_type_words']]
                self.graph['g_edge_type_words'] = [tokenize(each) for each in self.graph['g_edge_type_words']]


        else:
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



def vectorize_batch_graph(graphs, word_vocab, node_vocab, node_type_vocab, edge_type_vocab, oov_dict, kg_emb=False, f_node_type=False, ext_vocab=False):

    max_num_graph_nodes = max([g.get('num_virtual_nodes', len(g['g_node_name_words'])) for g in graphs])
    max_num_graph_edges = max([g.get('num_virtual_edges', len(g['g_edge_type_words'])) for g in graphs])


    # batch_num_virtual_nodes = []
    batch_num_nodes = []
    batch_num_edges = []

    batch_node_name_words = []
    batch_node_name_lens = []

    batch_node_type_words = []
    batch_node_type_lens = []

    batch_node_ans_match = []

    batch_edge_type_words = []
    batch_edge_type_lens = []

    if kg_emb:
        batch_node_ids = []
        batch_node_type_ids = []
        batch_edge_type_ids = []

    if ext_vocab:
        batch_g_oov_idx = []

    batch_node2edge = []
    batch_edge2node = []
    for example_id, g in enumerate(graphs):
        # Node names
        node_name_idx = []
        if ext_vocab:
            g_oov_idx = []


        for each in g['g_node_name_words']: # node level
            # Add out of vocab
            if ext_vocab:
                oov_idx = oov_dict.add_word(example_id, tuple(each))
                g_oov_idx.append(oov_idx)

            tmp_node_name_idx = []
            for word in each: # seq level
                idx = word_vocab.getIndex(word)
                tmp_node_name_idx.append(idx)
            node_name_idx.append(tmp_node_name_idx)
        batch_node_name_words.append(node_name_idx)
        batch_node_name_lens.append([max(len(x), 1) for x in node_name_idx])
        batch_node_ans_match.append(g['g_node_ans_match'])



        # Node types
        node_type_idx = []
        for each in g['g_node_type_words']: # node level
            tmp_node_type_idx = []
            for word in each: # seq level
                idx = word_vocab.getIndex(word)
                tmp_node_type_idx.append(idx)
            node_type_idx.append(tmp_node_type_idx)
        batch_node_type_words.append(node_type_idx)
        batch_node_type_lens.append([max(len(x), 1) for x in node_type_idx])


        # Edge types
        edge_type_idx = []
        for each in g['g_edge_type_words']:
            # Add out of vocab
            # if ext_vocab:
            #     oov_idx = oov_dict.add_word(example_id, tuple(each))
            #     g_oov_idx.append(oov_idx)

            tmp_edge_type_idx = []
            for word in each: # seq level
                idx = word_vocab.getIndex(word)
                tmp_edge_type_idx.append(idx)
            edge_type_idx.append(tmp_edge_type_idx)
        batch_edge_type_words.append(edge_type_idx)
        batch_edge_type_lens.append([max(len(x), 1) for x in edge_type_idx])

        # Number of virtual nodes = number of nodes + number of edges
        # batch_num_virtual_nodes.append(len(node_name_idx) + len(edge_type_idx))
        batch_num_nodes.append(len(node_name_idx))
        batch_num_edges.append(len(edge_type_idx))


        if ext_vocab:
            batch_g_oov_idx.append(g_oov_idx)
            # assert len(g_oov_idx) == len(node_name_idx) + len(edge_type_idx)
            assert len(g_oov_idx) == len(node_name_idx)



        # KG embedding
        if kg_emb:
            tmp_node_ids = []
            if f_node_type:
                tmp_node_type_ids = []
            for i, each in enumerate(g['g_node_ids']):
                tmp_node_ids.append(node_vocab.getIndex(each))
                if f_node_type:
                    tmp_node_type_ids.append(node_type_vocab.getIndex(g['g_node_type_ids'][i]))

            batch_node_ids.append(tmp_node_ids)
            if f_node_type:
                batch_node_type_ids.append(tmp_node_type_ids)


            tmp_edge_type_ids = []
            for each in g['g_edge_type_ids']:
                tmp_edge_type_ids.append(edge_type_vocab.getIndex(each))

            batch_edge_type_ids.append(tmp_edge_type_ids)


        # Adjacency matrix
        node2edge = lil_matrix(np.zeros((max_num_graph_edges, max_num_graph_nodes)), dtype=np.float32)
        edge2node = lil_matrix(np.zeros((max_num_graph_nodes, max_num_graph_edges)), dtype=np.float32)
        for node1, val in g['g_adj'].items():
            for node2, edge in val.items(): # node1 -> edge -> node2
                if node1 == node2: # Ignore self-loops for now
                    continue
                node2edge[edge, node1] = 1
                edge2node[node2, edge] = 1

        batch_node2edge.append(node2edge)
        batch_edge2node.append(edge2node)


    # batch_num_virtual_nodes = np.array(batch_num_virtual_nodes, dtype=np.int32)
    batch_num_nodes = np.array(batch_num_nodes, dtype=np.int32)
    batch_num_edges = np.array(batch_num_edges, dtype=np.int32)

    batch_node_name_words = padding_utils.pad_3d_vals_no_size(batch_node_name_words, fills=word_vocab.PAD)
    if f_node_type:
        batch_node_type_words = padding_utils.pad_3d_vals_no_size(batch_node_type_words, fills=word_vocab.PAD)
        batch_node_type_lens = padding_utils.pad_2d_vals_no_size(batch_node_type_lens, fills=1)

    batch_edge_type_words = padding_utils.pad_3d_vals_no_size(batch_edge_type_words, fills=word_vocab.PAD)

    batch_node_name_lens = padding_utils.pad_2d_vals_no_size(batch_node_name_lens, fills=1)

    batch_node_ans_match = padding_utils.pad_2d_vals_no_size(batch_node_ans_match, fills=0)

    batch_edge_type_lens = padding_utils.pad_2d_vals_no_size(batch_edge_type_lens, fills=1)

    batch_graphs = {'max_num_graph_nodes': max_num_graph_nodes,
                    'num_nodes': batch_num_nodes,
                    'num_edges': batch_num_edges,
                    'node_name_words': batch_node_name_words,
                    'edge_type_words': batch_edge_type_words,
                    'node_name_lens': batch_node_name_lens,
                    'node_ans_match': batch_node_ans_match,
                    'edge_type_lens': batch_edge_type_lens,
                    'node2edge': batch_node2edge,
                    'edge2node': batch_edge2node
                    }

    if f_node_type:
        batch_graphs['node_type_words'] = batch_node_type_words
        batch_graphs['node_type_lens'] = batch_node_type_lens


    if kg_emb:
        batch_node_ids = padding_utils.pad_2d_vals_no_size(batch_node_ids, fills=node_vocab.PAD)
        batch_edge_type_ids = padding_utils.pad_2d_vals_no_size(batch_edge_type_ids, fills=edge_type_vocab.PAD)

        batch_graphs['node_ids'] = batch_node_ids
        batch_graphs['edge_type_ids'] = batch_edge_type_ids

        if f_node_type:
            batch_node_type_ids = padding_utils.pad_2d_vals_no_size(batch_node_type_ids, fills=node_type_vocab.PAD)
            batch_graphs['node_type_ids'] = batch_node_type_ids


    if ext_vocab:
        batch_g_oov_idx = padding_utils.pad_2d_vals_no_size(batch_g_oov_idx, fills=word_vocab.PAD)
        batch_graphs['g_oov_idx'] = batch_g_oov_idx
    return batch_graphs



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


def find_sublist(src_list, a_list):
    indices = []
    for i in range(len(src_list)):
        if src_list[i: i + len(a_list)] == a_list:
            start_idx = i
            end_idx = i + len(a_list)
            indices.append((start_idx, end_idx))
    return indices

def seq2ext_vocab_id(idx_in_batch, seq, word_vocab, oov_dict):
    matched_pos = {}
    for key in oov_dict.word2index:
        if key[0] == idx_in_batch:
            indices = find_sublist(seq, list(key[1]))
            for pos in indices:
                matched_pos[pos] = key

    matched_pos = sorted(matched_pos.items(), key=lambda d: d[0][0])

    seq_idx = []
    i = 0
    while i < len(seq):
        if len(matched_pos) == 0 or i < matched_pos[0][0][0]:
            seq_idx.append(word_vocab.getIndex(seq[i]))
            i += 1
        else:
            pos, key = matched_pos.pop(0)
            seq_idx.append(oov_dict.word2index.get(key))
            i += len(key[1])
    return seq_idx
