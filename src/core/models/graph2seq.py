'''
Created on Oct, 2019

@author: hugo

'''
import random
import string
from typing import Union, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..layers.common import EncoderRNN, DecoderRNN, dropout
from ..layers.attention import *
from ..layers.graphs import GraphNN
from ..utils.generic_utils import to_cuda, create_mask
from ..utils.constants import VERY_SMALL_NUMBER


class Graph2SeqOutput(object):

  def __init__(self, encoder_outputs, encoder_state, decoded_tokens, \
          loss=0, loss_value=0, enc_attn_weights=None, ptr_probs=None):
    self.encoder_outputs = encoder_outputs
    self.encoder_state = encoder_state
    self.decoded_tokens = decoded_tokens  # (out seq len, batch size)
    self.loss = loss  # scalar
    self.loss_value = loss_value  # float value, excluding coverage loss
    self.enc_attn_weights = enc_attn_weights  # (out seq len, batch size, src seq len)
    self.ptr_probs = ptr_probs  # (out seq len, batch size)


class Graph2Seq(nn.Module):

  def __init__(self, config, word_embedding, word_vocab, entity_emb=None, relation_emb=None):
    """
    :param word_vocab: mainly for info about special tokens and word_vocab size
    :param config: model hyper-parameters
    :param max_dec_steps: max num of decoding steps (only effective at test time, as during
                          training the num of steps is determined by the `target_tensor`); it is
                          safe to change `self.max_dec_steps` as the network architecture is
                          independent of src/tgt seq lengths

    Create the graph2seq model; its encoder and decoder will be created automatically.
    """
    super(Graph2Seq, self).__init__()
    self.name = 'Graph2Seq'
    self.device = config['device']
    self.levi_graph = config.get('levi_graph', True)
    self.word_dropout = config['word_dropout']
    self.bert_dropout = config['bert_dropout']
    self.eps_label_smoothing = config.get('eps_label_smoothing', 0)
    if self.eps_label_smoothing is not None:
      assert self.eps_label_smoothing >= 0 and self.eps_label_smoothing <= 1

    self.word_vocab = word_vocab
    self.vocab_size = len(word_vocab)
    self.f_ans = config['f_ans']
    self.dan_type = config.get('dan_type', 'all')
    self.f_ans_match = config['f_ans_match']
    self.use_word_emb = config.get('use_word_emb', True)
    self.kg_emb = config['kg_emb']
    self.f_node_type = config.get('f_node_type', False)
    self.max_dec_steps = config['max_dec_steps']
    self.rnn_type = config['rnn_type']
    self.enc_attn = config['enc_attn']
    self.enc_attn_cover = config['enc_attn_cover']
    self.dec_attn = config['dec_attn']
    self.pointer = config['pointer']
    self.cover_loss = config['cover_loss']
    self.cover_func = config['cover_func']
    self.message_function = config['message_function']
    self.use_bert = config['use_bert']
    self.use_bert_weight = config['use_bert_weight']
    self.use_bert_gamma = config['use_bert_gamma']
    self.finetune_bert = config.get('finetune_bert', None)
    bert_dim = (config['bert_dim'] if self.use_bert else 0)
    assert self.use_word_emb or self.kg_emb
    assert config['entity_emb_dim'] == config['relation_emb_dim']
    config['graph_hidden_size'] = config['hidden_size'] if self.use_word_emb else 0

    enc_hidden_size = config['rnn_size']
    input_node_name_cat_dim = config['word_embed_dim']
    self.word_embed = word_embedding
    if config['fix_word_embed']:
      print('[ Fix word embeddings ]')
      for param in self.word_embed.parameters():
        param.requires_grad = False


    if self.kg_emb:
      self.node_embed = entity_emb if entity_emb is not None else nn.Embedding(config['num_entities'], config['entity_emb_dim'], padding_idx=0)
      self.edge_type_embed = relation_emb if relation_emb is not None else nn.Embedding(config['num_relations'], config['relation_emb_dim'], padding_idx=0)
      # if self.f_node_type:
      #   self.node_type_embed = nn.Embedding(config['num_entity_types'], config['entity_type_emb_dim'], padding_idx=0)
      config['graph_hidden_size'] += config['entity_emb_dim']



    if self.use_bert and self.use_bert_weight:
      start_idx, end_idx = config['bert_layer_indexes'].split(',')
      num_bert_layers = int(end_idx) - int(start_idx)
      self.logits_bert_layers = nn.Parameter(nn.init.xavier_uniform_(torch.Tensor(1, num_bert_layers)))
      if self.use_bert_gamma:
          self.gamma_bert_layers = nn.Parameter(nn.init.constant_(torch.Tensor(1, 1), 1.))



    # Answer alignment
    if self.f_ans:
      if self.dan_type in ('all', 'word'):
        self.node2ans_attn = Node2AnswerAttention(config['word_embed_dim'], config['hidden_size'])
        input_node_name_cat_dim += config['word_embed_dim']

      if self.dan_type in ('all', 'hidden'):
        self.answer_encoder = EncoderRNN(config['word_embed_dim'], enc_hidden_size, bidirectional=config['enc_bidi'], num_layers=config['num_enc_rnn_layers'], rnn_type=self.rnn_type,
                                rnn_dropout=config['enc_rnn_dropout'], device=self.device)
        self.node2ans_attn_l2 = Node2AnswerAttention(config['hidden_size'], config['hidden_size'])



      print('[ Using Answer Alignment Network]')


    if self.f_ans_match:
      self.ans_match_embed = nn.Embedding(3, config['ans_match_emb_dim'], padding_idx=0)
      config['graph_hidden_size'] += config['ans_match_emb_dim']
      print('[ Using exact answer matching]')


    if config['dec_hidden_size']:
      dec_hidden_size = config['dec_hidden_size']
      if self.rnn_type == 'lstm':
        self.enc_dec_adapter = nn.ModuleList([nn.Linear(config['graph_hidden_size'], dec_hidden_size) for _ in range(2)])
      else:
        self.enc_dec_adapter = nn.Linear(config['graph_hidden_size'], dec_hidden_size)
    else:
      dec_hidden_size = config['rnn_size']
      self.enc_dec_adapter = None


    self.node_name_word_encoder = EncoderRNN(input_node_name_cat_dim, enc_hidden_size, bidirectional=config['enc_bidi'], num_layers=config['num_enc_rnn_layers'], rnn_type=self.rnn_type,
                              rnn_dropout=config['enc_rnn_dropout'], device=self.device)

    # self.node_type_word_encoder = EncoderRNN(config['word_embed_dim'], enc_hidden_size, bidirectional=config['enc_bidi'], num_layers=config['num_enc_rnn_layers'], rnn_type=self.rnn_type,
    #                           rnn_dropout=config['enc_rnn_dropout'], device=self.device)


    self.edge_type_word_encoder = EncoderRNN(config['word_embed_dim'], enc_hidden_size, bidirectional=config['enc_bidi'], num_layers=config['num_enc_rnn_layers'], rnn_type=self.rnn_type,
                              rnn_dropout=config['enc_rnn_dropout'], device=self.device)


    self.graph_encoder = GraphNN(config)
    self.decoder = DecoderRNN(self.vocab_size, config['word_embed_dim'], dec_hidden_size, rnn_type=self.rnn_type,
                              enc_attn=config['enc_attn'], dec_attn=config['dec_attn'],
                              pointer=config['pointer'], out_embed_size=config['out_embed_size'],
                              tied_embedding=self.word_embed if config['tie_embed'] else None,
                              in_drop=config['dec_in_dropout'], rnn_drop=config.get('dec_rnn_dropout', config['enc_rnn_dropout']),
                              out_drop=config['dec_out_dropout'], enc_hidden_size=config['graph_hidden_size'], device=self.device)



  def filter_oov(self, tensor, ext_vocab_size):
    """Replace any OOV index in `tensor` with UNK"""
    if ext_vocab_size and ext_vocab_size > self.vocab_size:
      result = tensor.clone()
      result[tensor >= self.vocab_size] = self.word_vocab.UNK
      return result
    return tensor

  def get_coverage_vector(self, enc_attn_weights):
    """Combine the past attention weights into one vector"""
    if self.cover_func == 'max':
      coverage_vector, _ = torch.max(torch.cat(enc_attn_weights), dim=0)
    elif self.cover_func == 'sum':
      coverage_vector = torch.sum(torch.cat(enc_attn_weights), dim=0)
    else:
      raise ValueError('Unrecognized cover_func: ' + self.cover_func)
    return coverage_vector

  def forward(self, ex, target_tensor=None, criterion=None, criterion_reduction=True, criterion_nll_only=False, \
              rl_loss=False, *, forcing_ratio=0, partial_forcing=True, \
              ext_vocab_size=None, sample=False, saved_out: Graph2SeqOutput=None, \
              visualize: bool=None, include_cover_loss: bool=False) -> Graph2SeqOutput:
    """
    :param ex:
    :param target_tensor: tensor of word indices, (batch size, tgt seq len)
    :param input_lengths: see explanation in `EncoderRNN`
    :param criterion: the loss function; if set, loss will be returned
    :param forcing_ratio: see explanation in `Params` (requires `target_tensor`, training only)
    :param partial_forcing: see explanation in `Params` (training only)
    :param ext_vocab_size: see explanation in `DecoderRNN`
    :param sample: if True, the returned `decoded_tokens` will be based on random sampling instead
                   of greedily selecting the token of the highest probability at each step
    :param saved_out: the output of this function in a previous run; if set, the encoding step will
                      be skipped and we reuse the encoder states saved in this object
    :param visualize: whether to return data for attention and pointer visualization; if None,
                      return if no `criterion` is provided
    :param include_cover_loss: whether to include coverage loss in the returned `loss_value`

    Run the graph2seq model for training or testing.
    """
    input_graphs = ex['in_graphs']

    batch_size, max_num_nodes = input_graphs['node_name_words'].shape[:2]
    max_num_edges = input_graphs['edge_type_words'].shape[1]
    num_nodes = input_graphs['num_nodes']
    num_edges = input_graphs['num_edges']


    if self.levi_graph:
      max_num_graph_elements = input_graphs['max_num_graph_nodes']
      num_virtual_nodes = num_nodes + num_edges
      input_mask = create_mask(num_virtual_nodes, max_num_graph_elements, self.device)
      input_node_mask = create_mask(num_nodes, max_num_graph_elements, self.device)

    else:
      # max_num_graph_elements = input_graphs['max_num_graph_nodes']
      input_mask = create_mask(num_nodes, max_num_nodes, self.device)
      input_node_mask = create_mask(num_nodes, max_num_nodes, self.device)



    log_prob = not (sample or self.decoder.pointer)  # don't apply log too soon in these cases
    if visualize is None:
      visualize = criterion is None
    if visualize and not (self.enc_attn or self.pointer):
      visualize = False  # nothing to visualize

    if target_tensor is None:
      target_length = self.max_dec_steps
      target_mask = None
    else:
      target_tensor = target_tensor.transpose(1, 0)
      target_length = target_tensor.size(0)
      target_mask = create_mask(ex['target_lens'], target_length, self.device)


    if forcing_ratio == 1:
      # if fully teacher-forced, it may be possible to eliminate the for-loop over decoder steps
      # for generality, this optimization is not investigated
      use_teacher_forcing = True
    elif forcing_ratio > 0:
      if partial_forcing:
        use_teacher_forcing = None  # decide later individually in each step
      else:
        use_teacher_forcing = random.random() < forcing_ratio
    else:
      use_teacher_forcing = False

    if saved_out:  # reuse encoder states of a previous run
      encoder_outputs = saved_out.encoder_outputs
      encoder_state = saved_out.encoder_state
      # assert max_num_graph_elements == encoder_outputs.size(0)
      # assert batch_size == encoder_outputs.size(1)
    else:  # run the encoder

      if self.use_word_emb:
        # Node name
        node_name_word_emb = self.word_embed(self.filter_oov(input_graphs['node_name_words'], ext_vocab_size))
        node_name_word_emb = dropout(node_name_word_emb, self.word_dropout, shared_axes=[-2], training=self.training)
        node_name_lens = input_graphs['node_name_lens'].view(-1)

        input_node_name_cat = [node_name_word_emb]
        # Answer alignment
        if self.f_ans:
          answer_tensor = ex['answers'] # Shape: (batch_size, max_num_answers, L)
          answer_lens = ex['answer_lens'].view(-1)
          # Shape: (batch_size, max_num_answers * L)
          ans_mask = create_mask(ex['answer_lens'], answer_tensor.size(-1), self.device).view(answer_tensor.size(0), -1)

          ans_emb = self.word_embed(self.filter_oov(answer_tensor, ext_vocab_size))
          ans_emb = dropout(ans_emb, self.word_dropout, shared_axes=[-2], training=self.training)
          # Shape: (batch_size, max_num_answers * L, dim)
          ans_emb = ans_emb.view(ans_emb.size(0), -1, ans_emb.size(-1))

          # Shape: (batch_size, max_num_nodes * L, dim)
          node_name_word_emb = node_name_word_emb.view(node_name_word_emb.size(0), -1, node_name_word_emb.size(-1))


          # Word level alignment
          if self.dan_type in ('all', 'word'):
            # Shape: (batch_size, max_num_nodes, L, dim)
            node_aware_ans_emb = self.node2ans_attn(node_name_word_emb, ans_emb, ans_emb, ans_mask).view(input_graphs['node_name_words'].shape + (-1,))
            input_node_name_cat.append(node_aware_ans_emb)


        input_node_name_cat = torch.cat(input_node_name_cat, -1)
        input_node_name_cat = input_node_name_cat.view(-1, input_node_name_cat.size(-2), input_node_name_cat.size(-1))
        node_name_word_vec = self.node_name_word_encoder(input_node_name_cat, node_name_lens)[1]

        if self.rnn_type == 'lstm':
          node_name_word_vec = node_name_word_vec[0]

        # Shape: (batch_size, max_num_nodes, dim)
        node_name_word_vec = node_name_word_vec.squeeze(0).view(input_graphs['node_name_words'].shape[:2] + (-1,))
        input_node_cat = [node_name_word_vec]
      else:
        input_node_cat = []

      # Hidden level alignment
      if self.f_ans and self.dan_type in ('all', 'hidden'):
        ans_len = ex['answer_lens'].view(-1)
        ans_mask = create_mask(ex['num_answers'], answer_tensor.size(1), self.device)
        ans_vec = self.answer_encoder(ans_emb.view(answer_tensor.size(0) * answer_tensor.size(1), answer_tensor.size(2), -1), ans_len)[1]

        if self.rnn_type == 'lstm':
          ans_vec = ans_vec[0]

        ans_vec = ans_vec.squeeze(0).view(answer_tensor.shape[:2] + (-1,))

        node_aware_ans_emb = self.node2ans_attn_l2(node_name_word_vec, ans_vec, ans_vec, ans_mask)
        input_node_cat.append(node_aware_ans_emb)





      # # Node type
      # node_type_word_emb = self.word_embed(self.filter_oov(input_graphs['node_type_words'], ext_vocab_size))
      # node_type_word_emb = dropout(node_type_word_emb, self.word_dropout, shared_axes=[-2], training=self.training)
      # node_type_word_emb = node_type_word_emb.view(-1, node_type_word_emb.size(-2), node_type_word_emb.size(-1))
      # node_type_lens = input_graphs['node_type_lens'].view(-1)

      # node_type_word_emb = self.node_type_word_encoder(node_type_word_emb, node_type_lens)[1]
      # if self.rnn_type == 'lstm':
      #   node_type_word_emb = node_type_word_emb[0]

      # node_type_word_emb = node_type_word_emb.squeeze(0).view(batch_size, max_num_nodes, -1) # (batch_size, max_num_nodes, hidden_size)

      if self.use_word_emb:
        # Edge type
        edge_type_word_emb = self.word_embed(self.filter_oov(input_graphs['edge_type_words'], ext_vocab_size))
        edge_type_word_emb = dropout(edge_type_word_emb, self.word_dropout, shared_axes=[-2], training=self.training)
        edge_type_word_emb = edge_type_word_emb.view(-1, edge_type_word_emb.size(-2), edge_type_word_emb.size(-1))
        edge_type_lens = input_graphs['edge_type_lens'].view(-1)

        edge_type_word_emb = self.edge_type_word_encoder(edge_type_word_emb, edge_type_lens)[1]
        if self.rnn_type == 'lstm':
          edge_type_word_emb = edge_type_word_emb[0]

        edge_type_word_emb = edge_type_word_emb.squeeze(0).view(batch_size, max_num_edges, -1) # (batch_size, max_num_edges, hidden_size)

        # Fuse node & edge info
        input_edge_cat = [edge_type_word_emb]
      else:
        input_edge_cat = []


      if self.kg_emb:
        node_emb = self.node_embed(input_graphs['node_ids'])
        input_node_cat.append(node_emb)

        edge_type_emb = self.edge_type_embed(input_graphs['edge_type_ids'])
        input_edge_cat.append(edge_type_emb)

        # if self.f_node_type:
        #   node_type_emb = self.node_type_embed(input_graphs['node_type_ids'])
        #   input_node_cat.extend([node_type_emb])


      # Answer matching
      if self.f_ans_match:
        node_ans_match_emb = self.ans_match_embed(input_graphs['node_ans_match'])
        input_node_cat.append(node_ans_match_emb)
        input_edge_cat.append(to_cuda(torch.zeros(edge_type_word_emb.shape[:2] + (node_ans_match_emb.size(-1),)), self.device))

      input_node_vec = torch.cat(input_node_cat, -1)
      # if len(input_node_cat) > 1:
        # input_node_vec = self.linear_node_fusion(input_node_vec)

      input_edge_vec = torch.cat(input_edge_cat, -1)
      # if len(input_edge_cat) > 1:
        # input_edge_vec = self.linear_edge_fusion(input_edge_vec)



      if self.levi_graph:
        # Regard edges as nodes
        init_node_vec = self.gather(input_node_vec, input_edge_vec, num_nodes, num_edges, max_num_graph_elements)
        init_edge_vec = None
      else:
        init_node_vec = input_node_vec
        init_edge_vec = input_edge_vec


      node_embedding, graph_embedding = self.graph_encoder(init_node_vec, \
                  init_edge_vec, (input_graphs['node2edge'], input_graphs['edge2node']), \
                  node_mask=input_mask, ans_state=None)
      encoder_outputs = node_embedding
      encoder_state = (graph_embedding, graph_embedding) if self.rnn_type == 'lstm' else graph_embedding


    # initialize return values
    r = Graph2SeqOutput(encoder_outputs, encoder_state,
                      torch.zeros(target_length, batch_size, dtype=torch.long))
    if visualize:
      r.enc_attn_weights = torch.zeros(target_length, batch_size, max_num_graph_elements if self.levi_graph else max_num_nodes)
      if self.pointer:
        r.ptr_probs = torch.zeros(target_length, batch_size)

    if self.enc_dec_adapter is None:
      decoder_state = encoder_state
    else:
      if self.rnn_type == 'lstm':
        decoder_state = tuple([self.enc_dec_adapter[i](x) for i, x in enumerate(encoder_state)])
      else:
        decoder_state = self.enc_dec_adapter(encoder_state)
    decoder_hiddens = []
    enc_attn_weights = []


    enc_context = None
    dec_prob_ptr_tensor = []
    decoder_input = to_cuda(torch.tensor([self.word_vocab.SOS] * batch_size), self.device)
    for di in range(target_length):
      decoder_embedded = self.word_embed(self.filter_oov(decoder_input, ext_vocab_size))
      decoder_embedded = dropout(decoder_embedded, self.word_dropout, shared_axes=[-2], training=self.training)
      if enc_attn_weights:
        coverage_vector = self.get_coverage_vector(enc_attn_weights)
      else:
        coverage_vector = None
      decoder_output, decoder_state, dec_enc_attn, dec_prob_ptr, enc_context = \
        self.decoder(decoder_embedded, decoder_state, encoder_outputs,
                     torch.cat(decoder_hiddens) if decoder_hiddens else None, coverage_vector,
                     input_mask=input_mask,
                     input_node_mask=input_node_mask,
                     encoder_word_idx=input_graphs['g_oov_idx'] if self.pointer else None, ext_vocab_size=ext_vocab_size,
                     log_prob=log_prob,
                     prev_enc_context=enc_context)
      dec_prob_ptr_tensor.append(dec_prob_ptr)
      if self.dec_attn:
        decoder_hiddens.append(decoder_state[0] if self.rnn_type == 'lstm' else decoder_state)

      # save the decoded tokens
      if not sample:
        _, top_idx = decoder_output.data.topk(1)  # top_idx shape: (batch size, k=1)
      else:
        prob_distribution = torch.exp(decoder_output) if log_prob else decoder_output
        top_idx = torch.multinomial(prob_distribution, 1)
      top_idx = top_idx.squeeze(1).detach()  # detach from history as input
      r.decoded_tokens[di] = top_idx


      # decide the next input
      if use_teacher_forcing or (use_teacher_forcing is None and random.random() < forcing_ratio):
        decoder_input = target_tensor[di]  # teacher forcing
      else:
        decoder_input = top_idx

      # compute loss
      if criterion:
        if target_tensor is None:
          gold_standard = top_idx  # for sampling
        else:
          gold_standard = target_tensor[di] if not rl_loss else decoder_input
        if not log_prob:
          decoder_output = torch.log(decoder_output + VERY_SMALL_NUMBER)  # necessary for NLLLoss


        if self.eps_label_smoothing is not None:
          eps = self.eps_label_smoothing
          n_class = decoder_output.size(1)

          one_hot = to_cuda(torch.zeros_like(decoder_output).scatter(1, gold_standard.view(-1, 1), 1), self.device)
          one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)


          non_pad_mask = gold_standard.ne(self.word_vocab.PAD).float()
          nll_loss = -(one_hot * decoder_output).sum(dim=1)
          nll_loss = nll_loss * non_pad_mask

          if criterion_reduction:
            nll_loss = nll_loss.sum() / torch.sum(non_pad_mask)

          r.loss += nll_loss
          r.loss_value += nll_loss

        else:
          if criterion_reduction:
            nll_loss = criterion(decoder_output, gold_standard)
            r.loss += nll_loss
            r.loss_value += nll_loss.item()
          else:
            nll_loss = F.nll_loss(decoder_output, gold_standard, ignore_index=self.word_vocab.PAD, reduction='none')
            r.loss += nll_loss
            r.loss_value += nll_loss


      # update attention history and compute coverage loss
      if self.enc_attn_cover or (criterion and self.cover_loss > 0):
        if not criterion_nll_only and coverage_vector is not None and criterion and self.cover_loss > 0:
          if criterion_reduction:
            coverage_loss = torch.sum(torch.min(coverage_vector, dec_enc_attn)) / batch_size * self.cover_loss
            r.loss += coverage_loss
            if include_cover_loss: r.loss_value += coverage_loss.item()
          else:
            coverage_loss = torch.sum(torch.min(coverage_vector, dec_enc_attn), dim=-1) * self.cover_loss
            r.loss += coverage_loss
            if include_cover_loss: r.loss_value += coverage_loss

        enc_attn_weights.append(dec_enc_attn.unsqueeze(0))
      # save data for visualization
      if visualize:
        r.enc_attn_weights[di] = dec_enc_attn.data
        if self.pointer:
          r.ptr_probs[di] = dec_prob_ptr.squeeze(1).data

    return r

  def gather(self, input_tensor1, input_tensor2, num1, num2, max_num_graph_elements):
    input_tensor = torch.cat([input_tensor1, input_tensor2], 1)
    max_num1 = input_tensor1.size(1)

    index_tensor = []
    for i in range(input_tensor.size(0)):
      selected_index = list(range(num1[i].item())) + list(range(max_num1, max_num1 + num2[i].item()))
      if len(selected_index) < max_num_graph_elements:
        selected_index += [max_num_graph_elements - 1 for _ in range(max_num_graph_elements - len(selected_index))]
      index_tensor.append(selected_index)

    index_tensor = to_cuda(torch.LongTensor(index_tensor).unsqueeze(-1).expand(-1, -1, input_tensor.size(-1)), self.device)
    return torch.gather(input_tensor, 1, index_tensor)
