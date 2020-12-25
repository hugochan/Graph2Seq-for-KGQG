'''
Created on Oct, 2019

@author: hugo

'''
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..layers.attention import *
from ..utils.generic_utils import to_cuda
from .common import GatedFusion, GRUStep
from ..utils.constants import VERY_SMALL_NUMBER, INF


class GraphNN(nn.Module):
    def __init__(self, config):
        super(GraphNN, self).__init__()
        print('[ Using {}-hop GraphNN ]'.format(config['graph_hops']))
        self.device = config['device']
        self.f_ans_pool = config['f_ans_pool']
        hidden_size = config['graph_hidden_size']
        self.graph_direction = config.get('graph_direction', 'all')
        assert self.graph_direction in ('all', 'forward', 'backward')
        self.graph_type = config['graph_type']
        self.graph_hops = config['graph_hops']
        self.linear_max = nn.Linear(hidden_size, hidden_size, bias=False)

        # Static graph
        self.static_graph_mp = GraphMessagePassing(config)
        self.static_gru_step = GRUStep(hidden_size, hidden_size)

        if self.graph_direction == 'all':
            self.static_gated_fusion = GatedFusion(hidden_size)

        if self.graph_type == 'static':
            self.graph_update = self.static_graph_update
        elif self.graph_type == 'static_gcn':
            self.graph_update = self.static_gcn
            self.gcn_linear = nn.ModuleList([nn.Linear(hidden_size, hidden_size, bias=False) for _ in range(self.graph_hops)])
        else:
            raise RuntimeError('Unknown graph_type: {}'.format(self.graph_type))

        self.graph_pool = self.graph_maxpool

        print('[ Using graph type: {} ]'.format(self.graph_type))
        print('[ Using graph direction: {} ]'.format(self.graph_direction))


    def forward(self, node_state, edge_vec, adj, node_mask=None, ans_state=None):
        node_state, graph_embedding = self.graph_update(node_state, edge_vec, adj, node_mask=node_mask, ans_state=ans_state)
        return node_state, graph_embedding

    def static_graph_update(self, node_state, edge_vec, adj, node_mask=None, ans_state=None):
        '''Static graph update'''
        node2edge, edge2node = adj

        # Shape: (batch_size, num_edges, num_entities)
        node2edge = to_cuda(torch.stack([torch.Tensor(x.A) for x in node2edge], dim=0), self.device)
        # Shape: (batch_size, num_entities, num_edges)
        edge2node = to_cuda(torch.stack([torch.Tensor(x.A) for x in edge2node], dim=0), self.device)

        for _ in range(self.graph_hops):
            bw_agg_state = self.static_graph_mp.mp_func(node_state, edge_vec, node2edge, edge2node)
            fw_agg_state = self.static_graph_mp.mp_func(node_state, edge_vec, edge2node.transpose(1, 2), node2edge.transpose(1, 2))
            if self.graph_direction == 'all':
                agg_state = self.static_gated_fusion(fw_agg_state, bw_agg_state)
                node_state = self.static_gru_step(node_state, agg_state)
            elif self.graph_direction == 'forward':
                node_state = self.static_gru_step(node_state, fw_agg_state)
            else:
                node_state = self.static_gru_step(node_state, bw_agg_state)

        if self.f_ans_pool:
            graph_embedding = self.graph_pool(node_state, ans_state[0], node_state, context_mask=node_mask, ans_mask=ans_state[1]).unsqueeze(0)
        else:
            graph_embedding = self.graph_pool(node_state, node_mask).unsqueeze(0)
        return node_state.transpose(0, 1), graph_embedding

    def graph_maxpool(self, node_state, node_mask=None):
        # Maxpool
        # Shape: (batch_size, hidden_size, num_entities)
        node_embedding_p = self.linear_max(node_state).transpose(-1, -2)
        graph_embedding = F.max_pool1d(node_embedding_p, kernel_size=node_embedding_p.size(-1)).squeeze(-1)
        return graph_embedding

    def static_gcn(self, node_state, edge_vec, adj, node_mask=None, ans_state=None):
        '''Static GCN update'''
        node2edge, edge2node = adj

        # Shape: (batch_size, num_edges, num_entities)
        node2edge = to_cuda(torch.stack([torch.Tensor(x.A) for x in node2edge], dim=0), self.device)
        # Shape: (batch_size, num_entities, num_edges)
        edge2node = to_cuda(torch.stack([torch.Tensor(x.A) for x in edge2node], dim=0), self.device)

        adj = torch.bmm(edge2node, node2edge)
        adj = adj + adj.transpose(1, 2)
        adj = adj + to_cuda(torch.eye(adj.shape[1], adj.shape[2]), self.device)
        adj = torch.clamp(adj, max=1)

        rowsum = torch.sum(adj, -1)
        d_inv_sqrt = torch.pow(rowsum, -0.5)
        d_mat_inv_sqrt = torch.stack([torch.diagflat(d_inv_sqrt[i]) for i in range(d_inv_sqrt.shape[0])], dim=0)

        adj = torch.bmm(d_mat_inv_sqrt, torch.bmm(adj, d_mat_inv_sqrt))

        for _ in range(self.graph_hops):
            node_state = F.relu(self.gcn_linear[_](torch.bmm(adj, node_state)))

        graph_embedding = self.graph_maxpool(node_state, node_mask).unsqueeze(0)
        return node_state.transpose(0, 1), graph_embedding


class GraphMessagePassing(nn.Module):
    def __init__(self, config):
        super(GraphMessagePassing, self).__init__()
        self.config = config
        hidden_size = config['graph_hidden_size']
        if config['message_function'] == 'edge_pair':
            self.mp_func = self.msg_pass
            self.linear_fuse = nn.Linear(hidden_size * 2, hidden_size, bias=True)
        elif config['message_function'] == 'edge_network':
            self.edge_network = torch.Tensor(hidden_size, hidden_size, hidden_size)
            self.edge_network = nn.Parameter(nn.init.xavier_uniform_(self.edge_network))
            self.mp_func = self.msg_pass_edge_network
        elif config['message_function'] == 'no_edge':
            self.mp_func = self.msg_pass
        else:
            raise RuntimeError('Unknown message_function: {}'.format(config['message_function']))

    def msg_pass(self, node_state, edge_vec, node2edge, edge2node):
        node2edge_emb = torch.bmm(node2edge, node_state) # batch_size x num_edges x hidden_size
        if edge_vec is not None and self.config['message_function'] == 'edge_pair':
            # node2edge_emb = node2edge_emb + edge_vec
            node2edge_emb = torch.relu(self.linear_fuse(torch.cat([node2edge_emb, edge_vec], -1)))

        # Add self-loop
        norm_ = torch.sum(edge2node, 2, keepdim=True) + 1
        agg_state = (torch.bmm(edge2node, node2edge_emb) + node_state) / norm_
        return agg_state

    def msg_pass_maxpool(self, node_state, edge_vec, node2edge, edge2node, fc_maxpool):
        node2edge_emb = torch.bmm(node2edge, node_state) # batch_size x num_edges x hidden_size
        node2edge_emb = fc_maxpool(node2edge_emb)
        # Expand + mask
        # batch_size x num_entities x num_edges x hidden_size
        node2edge_emb = node2edge_emb.unsqueeze(1) * edge2node.unsqueeze(-1) - (1 - edge2node).unsqueeze(-1) * INF
        node2edge_emb = node2edge_emb.view(-1, node2edge_emb.size(-2), node2edge_emb.size(-1)).transpose(-1, -2)
        agg_state = F.max_pool1d(node2edge_emb, kernel_size=node2edge_emb.size(-1)).squeeze(-1).view(node_state.size())
        agg_state = agg_state * (torch.sum(edge2node, dim=-1, keepdim=True) != 0).float()
        return agg_state


    def msg_pass_edge_network(self, node_state, edge_vec, node2edge, edge2node):
        node2edge_emb = torch.bmm(node2edge, node_state) # batch_size x num_edges x hidden_size

        new_node2edge_emb = []
        for i in range(node2edge_emb.size(1)):
            edge_weight = torch.mm(edge_vec[:, i], self.edge_network.view(self.edge_network.size(0), -1)).view((-1,) + self.edge_network.shape[-2:])
            new_node2edge_emb.append(torch.matmul(edge_weight, node2edge_emb[:, i].unsqueeze(-1)).squeeze(-1))

        new_node2edge_emb = torch.stack(new_node2edge_emb, dim=1) # batch_size x num_edges x hidden_size

        # Add self-loop
        norm_ = torch.sum(edge2node, 2, keepdim=True) + 1
        agg_state = (torch.bmm(edge2node, new_node2edge_emb) + node_state) / norm_ # TODO: apply LP to node_state itself
        return agg_state
