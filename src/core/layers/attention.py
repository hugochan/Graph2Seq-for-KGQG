'''
Created on Nov, 2018

@author: hugo

'''
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils.constants import INF, VERY_SMALL_NUMBER


class Node2AnswerAttention(nn.Module):
    def __init__(self, dim, hidden_size):
        super(Node2AnswerAttention, self).__init__()

        self.weight_tensor = torch.Tensor(8, dim)
        self.weight_tensor = nn.Parameter(nn.init.xavier_uniform_(self.weight_tensor))

    def forward(self, context, answers, out_answers, ans_mask=None):
        expand_weight_tensor = self.weight_tensor.unsqueeze(1).unsqueeze(1)
        context = context.unsqueeze(0) * expand_weight_tensor
        context = F.normalize(context, p=2, dim=-1)

        answers = answers.unsqueeze(0) * expand_weight_tensor
        answers = F.normalize(answers, p=2, dim=-1)
        attention = torch.matmul(context, answers.transpose(-1, -2)).mean(0)

        attention = (attention + 1) / 2


        if ans_mask is not None:
            attention = attention.masked_fill_(1 - ans_mask.byte().unsqueeze(1), 0)

        # shape: (batch_size, L, dim)
        ques_emb = torch.matmul(attention, out_answers)
        return ques_emb



class AnswerAwareGraphPooling(nn.Module):
    def __init__(self, dim, hidden_size):
        super(AnswerAwareGraphPooling, self).__init__()
        self.linear_sim = nn.Linear(dim, hidden_size, bias=False)

    def forward(self, context, answers, out_context, context_mask=None, ans_mask=None):
        """
        Parameters
        :context, (batch_size, L, dim)
        :answers, (batch_size, N, dim)
        :out_context, (batch_size, N, dim)
        :ans_mask, (batch_size, N)

        Returns
        :ques_emb, (batch_size, L, dim)
        """
        context_fc = torch.relu(self.linear_sim(context))
        questions_fc = torch.relu(self.linear_sim(answers))

        # shape: (batch_size, L, N)
        attention = torch.matmul(context_fc, questions_fc.transpose(-1, -2))
        if context_mask is not None:
            attention = attention.masked_fill_(1 - context_mask.byte().unsqueeze(-1), -INF)

        if ans_mask is not None:
            attention = attention.masked_fill_(1 - ans_mask.byte().unsqueeze(1), -INF)

        attention = F.max_pool1d(attention, kernel_size=attention.size(-1)).squeeze(-1)
        prob = torch.softmax(attention, dim=-1)
        # shape: (batch_size, L)
        graph_emb = torch.matmul(prob.unsqueeze(1), out_context).squeeze(1)
        return graph_emb



class SelfAttention(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(SelfAttention, self).__init__()
        self.W1 = torch.Tensor(input_size, hidden_size)
        self.W1 = nn.Parameter(nn.init.xavier_uniform_(self.W1))
        self.W2 = torch.Tensor(hidden_size, 1)
        self.W2 = nn.Parameter(nn.init.xavier_uniform_(self.W2))

    def forward(self, x, attention_mask=None):
        attention = torch.mm(torch.tanh(torch.mm(x.view(-1, x.size(-1)), self.W1)), self.W2).view(x.size(0), -1)
        if attention_mask is not None:
            # Exclude masked elements from the softmax
            attention = attention.masked_fill_(1 - attention_mask.byte(), -INF)

        probs = torch.softmax(attention, dim=-1).unsqueeze(1)
        weighted_x = torch.bmm(probs, x).squeeze(1)
        return weighted_x

class Attention(nn.Module):
    def __init__(self, hidden_size, h_state_embed_size=None, in_memory_embed_size=None, attn_type='simple'):
        super(Attention, self).__init__()
        self.attn_type = attn_type
        if not h_state_embed_size:
            h_state_embed_size = hidden_size
        if not in_memory_embed_size:
            in_memory_embed_size = hidden_size
        if attn_type in ('mul', 'add'):
            self.W = torch.Tensor(h_state_embed_size, hidden_size)
            self.W = nn.Parameter(nn.init.xavier_uniform_(self.W))
            if attn_type == 'add':
                self.W2 = torch.Tensor(in_memory_embed_size, hidden_size)
                self.W2 = nn.Parameter(nn.init.xavier_uniform_(self.W2))
                self.W3 = torch.Tensor(hidden_size, 1)
                self.W3 = nn.Parameter(nn.init.xavier_uniform_(self.W3))
        elif attn_type == 'simple':
            pass
        else:
            raise RuntimeError('Unknown attn_type: {}'.format(self.attn_type))

    def forward(self, query_embed, in_memory_embed, attn_mask=None, addition_vec=None):
        if self.attn_type == 'simple': # simple attention
            attention = torch.bmm(in_memory_embed, query_embed.unsqueeze(2)).squeeze(2)
            if addition_vec is not None:
                attention = attention + addition_vec
        elif self.attn_type == 'mul': # multiplicative attention
            attention = torch.bmm(in_memory_embed, torch.mm(query_embed, self.W).unsqueeze(2)).squeeze(2)
            if addition_vec is not None:
                attention = attention + addition_vec
        elif self.attn_type == 'add': # additive attention
            attention = torch.mm(in_memory_embed.view(-1, in_memory_embed.size(-1)), self.W2)\
                .view(in_memory_embed.size(0), -1, self.W2.size(-1)) + torch.mm(query_embed, self.W).unsqueeze(1)
            if addition_vec is not None:
                attention = attention + addition_vec
            attention = torch.tanh(attention)
            attention = torch.mm(attention.view(-1, attention.size(-1)), self.W3).view(attention.size(0), -1)
        else:
            raise RuntimeError('Unknown attn_type: {}'.format(self.attn_type))

        if attn_mask is not None:
            # Exclude masked elements from the softmax
            attention = attn_mask * attention - (1 - attn_mask) * INF
        return attention
