from __future__ import absolute_import
from transformers import BertModel, BertConfig
from operator import itemgetter
from kge import KGEModel, KGE, KGEcalculate, KGELoss
import os
from tqdm import tqdm
import time
import itertools
import collections
import math
import pickle
import random
from dataloader import TestDataset, TrainDataset, SingledirectionalOneShotIterator
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.nn as nn
from html import entities

import logging
import struct
from textwrap import indent
from tkinter import E
import numpy as np
import torch
torch.cuda.set_device(0)

structure_sequence = {
    ('e', ('r',)): ['[1', '[0', '<e>', ']0', '<r>', None, ']1', '<e>'],
    ('e', ('r', 'r')): ['[1', '[0', '<e>', ']0', '<r>', None, '<r>', None, ']1', '<e>'],
    ('e', ('r', 'r', 'r')): ['[1', '[0', '<e>', ']0', '<r>', None, '<r>', None, '<r>', None, ']1', '<e>'],
    (('e', ('r',)), ('e', ('r',))): ['[1', '[0', '<e>', ']0', '<r>', None, ']1', '<inter>', '[1', '[0', '<e>', ']0', '<r>', None, ']1', '<e>'],
    (('e', ('r',)), ('e', ('r',)), ('e', ('r',))): ['[1', '[0', '<e>', ']0', '<r>', None, ']1', '<inter>', '[1', '[0', '<e>', ']0', '<r>', None, ']1', '<inter>', '[1', '[0', '<e>', ']0', '<r>', None, ']1', '<e>'],

    ((('e', ('r',)), ('e', ('r',))), ('r',)): ['[2', '[1', '[0', '<e>', ']0', '<r>', None, ']1', '<inter>', '[1', '[0', '<e>', ']0', '<r>', None, ']1', ']2', '<r>', None, '<e>'],
    (('e', ('r', 'r')), ('e', ('r',))): ['[1', '[0', '<e>', ']0', '<r>', None, '<r>', None, ']1', '<inter>', '[1', '[0', '<e>', ']0', '<r>', None, ']1', '<e>'],
    (('e', ('r',)), ('e', ('r', 'n'))): ['[1', '[0', '<e>', ']0', '<r>', None, ']1', '<inter>', '[1', '[0', '<e>', ']0', '<r>', None, '<neg>', ']1', '<e>'],
    (('e', ('r',)), ('e', ('r',)), ('e', ('r', 'n'))): ['[1', '[0', '<e>', ']0', '<r>', None, ']1', '<inter>', '[1', '[0', '<e>', ']0', '<r>', None, ']1', '<inter>', '[1', '[0', '<e>', ']0', '<r>', None, '<neg>', ']1', '<e>'],
    ((('e', ('r',)), ('e', ('r', 'n'))), ('r',)): ['[2', '[1', '[0', '<e>', ']0', '<r>', None, ']1', '<inter>', '[1', '[0', '<e>', ']0', '<r>', None, '<neg>', ']1', ']2', '<r>', None, '<e>'],

    (('e', ('r', 'r')), ('e', ('r', 'n'))): ['[1', '[0', '<e>', ']0', '<r>', None, '<r>', None, ']1', '<inter>', '[1', '[0', '<e>', ']0', '<r>', None, '<neg>', ']1', '<e>'],
    (('e', ('r', 'r', 'n')), ('e', ('r',))): ['[1', '[0', '<e>', ']0', '<r>', None, '<r>', None, '<neg>', ']1', '<inter>', '[1', '[0', '<e>', ']0', '<r>', None, ']1', '<e>'],
    (('e', ('r',)), ('e', ('r',)), ('u',)): ['[1', '[0', '<e>', ']0', '<r>', None, ']1', '<union>', '[1', '[0', '<e>', ']0', '<r>', None, ']1', '<e>'],
    ((('e', ('r',)), ('e', ('r',)), ('u',)), ('r',)): ['[2', '[1', '[0', '<e>', ']0', '<r>', None, ']1', '<union>', '[1', '[0', '<e>', ']0', '<r>', None, ']1', ']2', '<r>', None, '<e>']
}

idx_mask = {
    ('e', ('r',)): {0: 2},
    ('e', ('r', 'r')): {0: 2},
    ('e', ('r', 'r', 'r')): {0: 2},
    (('e', ('r',)), ('e', ('r',))): {0: 2, 2: 10},
    (('e', ('r',)), ('e', ('r',)), ('e', ('r',))): {0: 2, 2: 10, 4: 18},

    ((('e', ('r',)), ('e', ('r',))), ('r',)): {0: 3, 2: 11},
    (('e', ('r', 'r')), ('e', ('r',))): {0: 2, 3: 12},
    (('e', ('r',)), ('e', ('r', 'n'))): {0: 2, 2: 10},
    (('e', ('r',)), ('e', ('r',)), ('e', ('r', 'n'))): {0: 2, 2: 10, 4: 18},
    ((('e', ('r',)), ('e', ('r', 'n'))), ('r',)): {0: 3, 2: 11},

    (('e', ('r', 'r')), ('e', ('r', 'n'))): {0: 2, 3: 12},
    (('e', ('r', 'r', 'n')), ('e', ('r',))): {0: 2, 4: 13},
    (('e', ('r',)), ('e', ('r',)), ('u',)): {0: 2, 2: 10},
    ((('e', ('r',)), ('e', ('r',)), ('u',)), ('r',)): {0: 3, 2: 11}
}

relation_inplace = {
    ('e', ('r',)): {1: 5},
    ('e', ('r', 'r')): {1: 5, 2: 7},
    ('e', ('r', 'r', 'r')): {1: 5, 2: 7, 3: 9},
    (('e', ('r',)), ('e', ('r',))): {1: 5, 3: 13},
    (('e', ('r',)), ('e', ('r',)), ('e', ('r',))): {1: 5, 3: 13, 5: 21},

    ((('e', ('r',)), ('e', ('r',))), ('r',)): {1: 6, 3: 14, 4: 18},
    (('e', ('r', 'r')), ('e', ('r',))): {1: 5, 2: 7, 4: 15},
    (('e', ('r',)), ('e', ('r', 'n'))): {1: 5, 3: 13},
    (('e', ('r',)), ('e', ('r',)), ('e', ('r', 'n'))): {1: 5, 3: 13, 5: 21},
    ((('e', ('r',)), ('e', ('r', 'n'))), ('r',)): {1: 6, 3: 14, 5: 19},

    (('e', ('r', 'r')), ('e', ('r', 'n'))): {1: 5, 2: 7, 4: 15},
    (('e', ('r', 'r', 'n')), ('e', ('r',))): {1: 5, 2: 7, 5: 16},
    (('e', ('r',)), ('e', ('r',)), ('u',)): {1: 5, 3: 13},
    ((('e', ('r',)), ('e', ('r',)), ('u',)), ('r',)): {1: 6, 3: 14, 5: 18}
}

class ConeProjection(nn.Module):
    def __init__(self, dim, hidden_dim, num_layers):
        super(ConeProjection, self).__init__()
        self.entity_dim = dim
        self.relation_dim = dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.layer1 = nn.Linear(self.entity_dim + self.relation_dim, self.hidden_dim)  
        self.layer0 = nn.Linear(self.hidden_dim, self.entity_dim + self.relation_dim)  
        for nl in range(2, num_layers + 1):
            setattr(self, "layer{}".format(nl), nn.Linear(self.hidden_dim, self.hidden_dim))
        for nl in range(num_layers + 1):
            nn.init.xavier_uniform_(getattr(self, "layer{}".format(nl)).weight)

    def forward(self, source_embedding_axis, source_embedding_arg, r_embedding_axis, r_embedding_arg):
        x = torch.cat([source_embedding_axis + r_embedding_axis, source_embedding_arg + r_embedding_arg], dim=-1)
        for nl in range(1, self.num_layers + 1):
            x = F.relu(getattr(self, "layer{}".format(nl))(x))
        x = self.layer0(x)

        axis, arg = torch.chunk(x, 2, dim=-1)
        axis_embeddings = convert_to_axis(axis)
        arg_embeddings = convert_to_arg(arg)
        return axis_embeddings, arg_embeddings

class ConeIntersection(nn.Module):
    def __init__(self, dim, drop):
        super(ConeIntersection, self).__init__()
        self.dim = dim
        self.layer_axis1 = nn.Linear(self.dim * 2, self.dim)
        self.layer_arg1 = nn.Linear(self.dim * 2, self.dim)
        self.layer_axis2 = nn.Linear(self.dim, self.dim)
        self.layer_arg2 = nn.Linear(self.dim, self.dim)

        nn.init.xavier_uniform_(self.layer_axis1.weight)
        nn.init.xavier_uniform_(self.layer_arg1.weight)
        nn.init.xavier_uniform_(self.layer_axis2.weight)
        nn.init.xavier_uniform_(self.layer_arg2.weight)

        self.drop = nn.Dropout(p=drop)

    def forward(self, axis_embeddings, arg_embeddings):
        logits = torch.cat([axis_embeddings - arg_embeddings, axis_embeddings + arg_embeddings], dim=-1)
        axis_layer1_act = F.relu(self.layer_axis1(logits))

        axis_attention = F.softmax(self.layer_axis2(axis_layer1_act), dim=0)

        x_embeddings = torch.cos(axis_embeddings)
        y_embeddings = torch.sin(axis_embeddings)
        x_embeddings = torch.sum(axis_attention * x_embeddings, dim=0)
        y_embeddings = torch.sum(axis_attention * y_embeddings, dim=0)

        x_embeddings[torch.abs(x_embeddings) < 1e-3] = 1e-3

        axis_embeddings = torch.atan(y_embeddings / x_embeddings)

        indicator_x = x_embeddings < 0
        indicator_y = y_embeddings < 0
        indicator_two = indicator_x & torch.logical_not(indicator_y)
        indicator_three = indicator_x & indicator_y

        axis_embeddings[indicator_two] = axis_embeddings[indicator_two] + pi
        axis_embeddings[indicator_three] = axis_embeddings[indicator_three] - pi

        arg_layer1_act = F.relu(self.layer_arg1(logits))
        arg_layer1_mean = torch.mean(arg_layer1_act, dim=0)
        gate = torch.sigmoid(self.layer_arg2(arg_layer1_mean))

        arg_embeddings = self.drop(arg_embeddings)
        arg_embeddings, _ = torch.min(arg_embeddings, dim=0)
        arg_embeddings = arg_embeddings * gate

        return axis_embeddings, arg_embeddings

class ConeNegation(nn.Module):
    def __init__(self):
        super(ConeNegation, self).__init__()

    def forward(self, axis_embedding, arg_embedding):
        indicator_positive = axis_embedding >= 0
        indicator_negative = axis_embedding < 0

        axis_embedding[indicator_positive] = axis_embedding[indicator_positive] - pi
        axis_embedding[indicator_negative] = axis_embedding[indicator_negative] + pi

        arg_embedding = pi - arg_embedding

        return axis_embedding, arg_embedding

pi = 3.14159265358979323846

def convert_to_arg(x):
    y = torch.tanh(2 * x) * pi / 2 + pi / 2
    return y

def convert_to_axis(x):
    y = torch.tanh(x) * pi
    return y

class AngleScale:
    def __init__(self, embedding_range):
        self.embedding_range = embedding_range

    def __call__(self, axis_embedding, scale=None):
        if scale is None:
            scale = pi
        return axis_embedding / self.embedding_range * scale


def Identity(x):
    return x


class BoxOffsetIntersection(nn.Module):

    def __init__(self, dim):
        super(BoxOffsetIntersection, self).__init__()
        self.dim = dim
        self.layer1 = nn.Linear(self.dim, self.dim)
        self.layer2 = nn.Linear(self.dim, self.dim)

        nn.init.xavier_uniform_(self.layer1.weight)
        nn.init.xavier_uniform_(self.layer2.weight)

    def forward(self, embeddings):
        layer1_act = F.relu(self.layer1(embeddings))
        layer1_mean = torch.mean(layer1_act, dim=0)
        gate = torch.sigmoid(self.layer2(layer1_mean))
        offset, _ = torch.min(embeddings, dim=0)

        return offset * gate


class CenterIntersection(nn.Module):

    def __init__(self, dim):
        super(CenterIntersection, self).__init__()
        self.dim = dim
        self.layer1 = nn.Linear(self.dim, self.dim)
        self.layer2 = nn.Linear(self.dim, self.dim)

        nn.init.xavier_uniform_(self.layer1.weight)
        nn.init.xavier_uniform_(self.layer2.weight)

    def forward(self, embeddings):
        layer1_act = F.relu(self.layer1(embeddings))
        attention = F.softmax(self.layer2(layer1_act), dim=0)
        embedding = torch.sum(attention * embeddings, dim=0)

        return embedding


class BetaIntersection(nn.Module):

    def __init__(self, dim):
        super(BetaIntersection, self).__init__()
        self.dim = dim
        self.layer1 = nn.Linear(2 * self.dim, 2 * self.dim)
        self.layer2 = nn.Linear(2 * self.dim, self.dim)

        nn.init.xavier_uniform_(self.layer1.weight)
        nn.init.xavier_uniform_(self.layer2.weight)

    def forward(self, alpha_embeddings, beta_embeddings, attention=None):
        if True or type(attention) == type(None):
            all_embeddings = torch.cat([alpha_embeddings, beta_embeddings], dim=-1)
            layer1_act = F.relu(self.layer1(all_embeddings))
            attention = F.softmax(self.layer2(layer1_act), dim=0)
            alpha_embedding = torch.sum(attention * alpha_embeddings, dim=0)
            beta_embedding = torch.sum(attention * beta_embeddings, dim=0)
        else:
            attention = F.softmax(attention, dim=-1).view(attention.shape[0], 1, 1, attention.shape[1])
            alpha_embedding = torch.sum(attention * alpha_embeddings, dim=2)
            beta_embedding = torch.sum(attention * beta_embeddings, dim=2)

        return alpha_embedding, beta_embedding


class BetaProjection(nn.Module):
    def __init__(self, entity_dim, relation_dim, hidden_dim, projection_regularizer, num_layers):
        super(BetaProjection, self).__init__()
        self.entity_dim = entity_dim
        self.relation_dim = relation_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.layer1 = nn.Linear(self.entity_dim + self.relation_dim, self.hidden_dim)
        self.layer0 = nn.Linear(self.hidden_dim, self.entity_dim)
        for nl in range(2, num_layers + 1):
            setattr(self, "layer{}".format(nl), nn.Linear(self.hidden_dim, self.hidden_dim))
        for nl in range(num_layers + 1):
            nn.init.xavier_uniform_(getattr(self, "layer{}".format(nl)).weight)
        self.projection_regularizer = projection_regularizer

    def forward(self, e_embedding, r_embedding):
        x = torch.cat([e_embedding, r_embedding], dim=-1)
        for nl in range(1, self.num_layers + 1):
            x = F.relu(getattr(self, "layer{}".format(nl))(x))
        x = self.layer0(x)
        x = self.projection_regularizer(x)

        return x


class Regularizer():
    def __init__(self, base_add, min_val, max_val):
        self.base_add = base_add
        self.min_val = min_val
        self.max_val = max_val

    def __call__(self, entity_embedding):
        return torch.clamp(entity_embedding + self.base_add, self.min_val, self.max_val)


class FFN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim):
        super(FFN, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, output_dim)

        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.xavier_uniform_(self.linear2.weight)

    def forward(self, input):
        return F.relu(self.linear2(F.relu(self.linear1(input))))


class KGReasoning(nn.Module):
    def __init__(self, nentity, nrelation, hidden_dim, gamma,
                 geo, mode, test_batch_size=1,
                 box_mode=None, use_cuda=False,
                 query_name_dict=None, beta_mode=None, mat=None, inductiveGraph=None, loss_weight=None, args=None):
        super(KGReasoning, self).__init__()
        self.nentity = nentity
        self.nrelation = nrelation
        self.hidden_dim = hidden_dim
        self.epsilon = 2.0
        self.geo = geo
        self.KGEmode = mode
        self.use_cuda = use_cuda
        self.batch_entity_range = torch.arange(nentity).to(torch.float).repeat(test_batch_size, 1).cuda(
        ) if self.use_cuda else torch.arange(nentity).to(torch.float).repeat(test_batch_size, 1)
        self.query_name_dict = query_name_dict
        if self.geo == 'ns':
            self.register_buffer('mat', torch.stack(mat))
        self.loss_weight = loss_weight
        self.args = args
        self.one = torch.tensor([1]).cuda()
        self.thr = torch.Tensor([1e-10]).cuda()

        self.gamma = nn.Parameter(
            torch.Tensor([gamma]),
            requires_grad=False
        )

        self.embedding_range = nn.Parameter(
            torch.Tensor([(self.gamma.item() + self.epsilon) / hidden_dim]),
            requires_grad=False
        )

        self.entity_dim = hidden_dim
        self.relation_dim = hidden_dim

        if self.geo == 'box':
            self.entity_embedding = nn.Parameter(torch.zeros(nentity + 1, self.entity_dim))
            activation, cen = box_mode
            self.cen = cen
            if activation == 'none':
                self.func = Identity
            elif activation == 'relu':
                self.func = F.relu
            elif activation == 'softplus':
                self.func = F.softplus
        elif self.geo == 'vec':
            self.entity_embedding = nn.Parameter(torch.zeros(nentity + 1, self.entity_dim))
        elif self.geo == 'beta':
            self.entity_embedding = nn.Parameter(torch.zeros(nentity + 1, self.entity_dim * 2))
            self.entity_regularizer = Regularizer(1, 0.05, 1e9)
            self.projection_regularizer = Regularizer(1, 0.05, 1e9)
        elif self.geo == 'ns':
            self.entity_embedding, self.relation_embedding, self.offset_embedding = KGE(
                mode, self.nentity + 1, self.nrelation, self.hidden_dim, self.gamma, self.embedding_range)
            self.entity_dim = self.entity_embedding.shape[1]
            self.relation_dim = self.relation_embedding.shape[1]
            self.FFN_vec2emb = FFN(self.entity_dim, 1, hidden_dim)
            self.FFN_emb2vec = FFN(self.nentity + 1, self.nentity + 1, hidden_dim)
            self.mat_degree = []
            for single_mat in mat:
                self.mat_degree.append((torch.sum(single_mat.values()) / len(set(single_mat.indices()[0].cpu().tolist()))).long())
        elif self.geo == 'cone':
            self.entity_embedding = nn.Parameter(torch.zeros(nentity + 1, self.entity_dim))

        if self.geo != 'ns':
            nn.init.uniform_(
                tensor=self.entity_embedding,
                a=-self.embedding_range.item(),
                b=self.embedding_range.item()
            )
            self.relation_embedding = nn.Parameter(torch.zeros(nrelation + 1, self.relation_dim))
            nn.init.uniform_(
                tensor=self.relation_embedding,
                a=-self.embedding_range.item(),
                b=self.embedding_range.item()
            )

        if self.geo == 'box':
            self.offset_embedding = nn.Parameter(torch.zeros(nrelation + 1, self.entity_dim))
            nn.init.uniform_(
                tensor=self.offset_embedding,
                a=0.,
                b=self.embedding_range.item()
            )
            self.center_net = CenterIntersection(self.entity_dim)
            self.offset_net = BoxOffsetIntersection(self.entity_dim)
        elif self.geo == 'vec' or self.geo == 'ns':
            self.center_net = CenterIntersection(self.entity_dim)
        elif self.geo == 'beta':
            hidden_dim, num_layers = beta_mode
            self.center_net = BetaIntersection(self.entity_dim)
            self.projection_net = BetaProjection(self.entity_dim * 2,
                                                 self.relation_dim,
                                                 hidden_dim,
                                                 self.projection_regularizer,
                                                 num_layers)

        self.relation_set_embedding = nn.Parameter(torch.zeros(nrelation + 1, self.relation_dim))
        nn.init.uniform_(
            tensor=self.relation_set_embedding,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )

        if self.geo == 'cone':
            self.angle_scale = AngleScale(self.embedding_range.item())

            self.modulus = nn.Parameter(torch.Tensor([0.5 * self.embedding_range.item()]), requires_grad=True)

            self.cen = args.center_reg

            self.axis_scale = 1.0
            self.arg_scale = 1.0
            self.axis_embedding = nn.Parameter(torch.zeros(nrelation+1, self.relation_dim), requires_grad=True)
            nn.init.uniform_(
                tensor=self.axis_embedding,
                a=-self.embedding_range.item(),
                b=self.embedding_range.item()
            )


            self.arg_embedding = nn.Parameter(torch.zeros(nrelation+1, self.relation_dim), requires_grad=True)
            nn.init.uniform_(
                tensor=self.arg_embedding,
                a=-self.embedding_range.item(),
                b=self.embedding_range.item()
            )

            self.cone_proj = ConeProjection(self.entity_dim, 1600, 2)
            self.cone_intersection = ConeIntersection(self.entity_dim, args.drop)
            self.cone_negation = ConeNegation()

            self.ind_pro1 = nn.Linear(self.relation_dim, self.entity_dim)
            self.ind_pro2 = nn.Linear(self.entity_dim, self.entity_dim)

        self.inductiveGraph = inductiveGraph
        if self.geo != 'beta':
            self.entity_mask = torch.ones(self.nentity+1, self.entity_dim).cuda()
        else:
            self.entity_mask = torch.ones(self.nentity+1, self.entity_dim*2).cuda()
        self.entity_mask.requires_grad = False
        self.entity_mask[-1][:] = 0
        self.relation_mask = torch.ones(self.nrelation+1, self.relation_dim).cuda()
        self.relation_mask.requires_grad = False
        self.relation_mask[-1][:] = 0

        self.inductive_Q = nn.Linear(self.entity_embedding.shape[1], self.entity_embedding.shape[1], bias = False)
        self.inductive_K = nn.Linear(self.entity_embedding.shape[1], self.entity_embedding.shape[1], bias = False)
        self.inductive_V = nn.Linear(self.entity_embedding.shape[1], self.entity_embedding.shape[1], bias = False)
        self.inductive_type_Q = nn.Linear(self.relation_dim, self.entity_embedding.shape[1], bias = False)
        self.inductive_type_K = nn.Linear(self.relation_dim, self.entity_embedding.shape[1], bias = False)
        self.inductive_type_V = nn.Linear(self.relation_dim, self.entity_embedding.shape[1], bias = False)

        nn.init.uniform_(
            tensor=self.inductive_Q.weight,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )
        nn.init.uniform_(
            tensor=self.inductive_K.weight,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )
        nn.init.uniform_(
            tensor=self.inductive_V.weight,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )

        nn.init.uniform_(
            tensor=self.inductive_type_Q.weight,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )
        nn.init.uniform_(
            tensor=self.inductive_type_K.weight,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )
        nn.init.uniform_(
            tensor=self.inductive_type_V.weight,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )

        tokens = list({x for v in structure_sequence.values() for x in v}) + ['<pad>']
        self.tok2id = {tok: i+self.nrelation for i, tok in enumerate(tokens)}
        self.structure_sequence_id = {k: [self.tok2id[x] for x in v] for k, v in structure_sequence.items()}
        query_bert_config = BertConfig(vocab_size=self.nrelation+len(tokens), hidden_size=self.entity_embedding.shape[1], num_hidden_layers=3, num_attention_heads=1,
                                       intermediate_size=512, max_position_embeddings=40, type_vocab_size=1, pad_token_id=self.tok2id['<pad>'])
        self.bert = BertModel(query_bert_config, add_pooling_layer=False)

    def embedding_fusing(self, node, prompt):
        relations, entities = self.get_nbor(node.cpu().numpy().tolist())

        type_embeddings, embeddings = self.predict(relations, entities)

        type_embeddings, embeddings = self.exchange_info(type_embeddings, embeddings)
        fused_embedding = self.query_attn(type_embeddings, embeddings, prompt)

        return fused_embedding

    def get_nbor(self, node):
        info = torch.tensor(itemgetter(*node)(self.inductiveGraph.graph))
        if len(node) == 1:
            info = info.unsqueeze(0)

        relations, entities = info.transpose(0, 1).cuda()
        return relations, entities

    def predict(self, relations, entities):
        self.relation_embedding.data = self.relation_embedding.data * self.relation_mask
        self.relation_set_embedding.data = self.relation_set_embedding.data * self.relation_mask
        self.entity_embedding.data = self.entity_embedding.data * self.entity_mask

        type_embeddings = self.relation_set_embedding[relations]

        if self.geo == 'vec':
            neighbor_embeddings = self.entity_embedding[entities]
            relation_embeddings = self.relation_embedding[relations]
            embeddings = neighbor_embeddings + relation_embeddings
            return type_embeddings, embeddings

        elif self.geo == 'box':
            neighbor_embeddings = self.entity_embedding[entities]
            relation_embeddings = self.relation_embedding[relations]
            embeddings = neighbor_embeddings + relation_embeddings
            return type_embeddings, embeddings

        elif self.geo == 'beta':
            neighbor_embeddings = self.entity_regularizer(self.entity_embedding[entities])
            relation_embeddings = self.relation_embedding[relations]
            embeddings = self.projection_net(neighbor_embeddings, relation_embeddings)
            return type_embeddings, embeddings
        
        elif self.geo == 'cone':
            neighbor_embeddings = self.entity_embedding[entities]
            relation_embeddings = self.axis_embedding[relations]

            type_embeddings = self.relation_set_embedding[relations]
            type_embeddings = self.angle_scale(type_embeddings, self.axis_scale)
            type_embeddings = convert_to_axis(type_embeddings)
            
            axis_embedding = self.angle_scale(neighbor_embeddings, self.axis_scale)
            axis_embedding = convert_to_axis(neighbor_embeddings)
            arg_embedding = torch.zeros_like(axis_embedding).cuda()
            
            axis_r_embedding = self.axis_embedding[relations]
            arg_r_embedding = self.arg_embedding[relations]

            axis_r_embedding = self.angle_scale(axis_r_embedding, self.axis_scale)
            arg_r_embedding = self.angle_scale(arg_r_embedding, self.arg_scale)

            axis_r_embedding = convert_to_axis(axis_r_embedding)
            arg_r_embedding = convert_to_axis(arg_r_embedding)

            embeddings, arg_embedding = self.cone_proj(axis_embedding, arg_embedding, axis_r_embedding, arg_r_embedding)
            return type_embeddings, embeddings


    def exchange_info(self, type_embedding, embeddings):
        padding = embeddings[:, :, 0].bool()
        padding = torch.einsum('bn, bm -> bnm', [padding, padding])
        query = self.inductive_Q(embeddings)
        key = self.inductive_K(embeddings)
        value = self.inductive_V(embeddings)

        key_trans = torch.transpose(key, 1, 2)
        attn = torch.einsum('bnd, bdm -> bnm', [query, key_trans])
        attn = attn / math.sqrt(self.entity_dim)
        attn = attn.masked_fill(~padding, -1e9)
        attn = torch.softmax(attn, dim=-1)
        embeddings = torch.einsum('bnm, bmd -> bnd', [attn, value])

        query = self.inductive_type_Q(type_embedding)
        key = self.inductive_type_K(type_embedding)
        value = self.inductive_type_V(type_embedding)
        attn = torch.einsum('bnd, bdm -> bnm', [query, key_trans])
        attn = attn / math.sqrt(self.entity_dim)
        attn = attn.masked_fill(~padding, -1e9)
        attn = torch.softmax(attn, dim=-1)
        type_embeddings = torch.einsum('bnm, bmd -> bnd', [attn, value])

        return type_embeddings, embeddings

    def query_attn(self, type_embeddings, embeddings, prompt):
        type_embedding = self.induc_inter(type_embeddings, prompt)
        embedding = self.induc_inter(embeddings, prompt)

        embedding = (type_embedding + embedding) / 2

        return embedding

    def induc_inter(self, embeddings, prompt):
        embeddings = embeddings.reshape(prompt.shape[0], embeddings.shape[0]//prompt.shape[0], embeddings.shape[1], embeddings.shape[2])
        if True or self.geo in ['vec', 'box']:
            attn = torch.einsum('ijkd, id -> ijk', [embeddings, prompt])
            attn = F.softmax(attn, dim=-1)
            embedding = torch.einsum('ijkd, ijk -> ijd', [embeddings, attn])
            embedding = embedding.view(-1, prompt.shape[1])
        elif self.geo == 'beta':
            embedding = torch.cat(self.center_net(*torch.chunk(embeddings, 2, dim=-1), prompt), dim=-1)
            embedding = embedding.view(-1, 2*prompt.shape[1])
        return embedding

    def forward(self, positive_sample, negative_sample, subsampling_weight, batch_queries_dict, batch_idxs_dict):
        if self.geo == 'box':
            return self.forward_box(positive_sample, negative_sample, subsampling_weight, batch_queries_dict, batch_idxs_dict)
        elif self.geo == 'vec':
            return self.forward_vec(positive_sample, negative_sample, subsampling_weight, batch_queries_dict, batch_idxs_dict)
        elif self.geo == 'beta':
            return self.forward_beta(positive_sample, negative_sample, subsampling_weight, batch_queries_dict, batch_idxs_dict)
        elif self.geo == 'ns':
            return self.forward_ns(positive_sample, negative_sample, subsampling_weight, batch_queries_dict, batch_idxs_dict)
        elif self.geo == 'cone':
            return self.forward_cone(positive_sample, negative_sample, subsampling_weight, batch_queries_dict, batch_idxs_dict)

    def embed_query_cone(self, queries, query_structure, idx, query_sequence_embedding, whole_query_structure):
        all_relation_flag = True
        for ele in query_structure[-1]:
            if ele not in ['r', 'n']:
                all_relation_flag = False
                break
        if all_relation_flag:
            if query_structure[0] == 'e':
                prompt = query_sequence_embedding[:, idx_mask[whole_query_structure][idx]]
                axis_entity_embedding = self.embedding_fusing(queries[:, idx], prompt)
                axis_entity_embedding = self.angle_scale(axis_entity_embedding, self.axis_scale)
                axis_entity_embedding = convert_to_axis(axis_entity_embedding)

                if self.use_cuda:
                    arg_entity_embedding = torch.zeros_like(axis_entity_embedding).cuda()
                else:
                    arg_entity_embedding = torch.zeros_like(axis_entity_embedding)
                idx += 1

                axis_embedding = axis_entity_embedding
                arg_embedding = arg_entity_embedding
            else:
                axis_embedding, arg_embedding, idx = self.embed_query_cone(queries, query_structure[0], idx, query_sequence_embedding, whole_query_structure)

            for i in range(len(query_structure[-1])):
                if query_structure[-1][i] == 'n':
                    assert (queries[:, idx] == -2).all()
                    axis_embedding, arg_embedding = self.cone_negation(axis_embedding, arg_embedding)
                    

                else:
                    axis_r_embedding = torch.index_select(self.axis_embedding, dim=0, index=queries[:, idx])
                    arg_r_embedding = torch.index_select(self.arg_embedding, dim=0, index=queries[:, idx])

                    axis_r_embedding = self.angle_scale(axis_r_embedding, self.axis_scale)
                    arg_r_embedding = self.angle_scale(arg_r_embedding, self.arg_scale)

                    axis_r_embedding = convert_to_axis(axis_r_embedding)
                    arg_r_embedding = convert_to_axis(arg_r_embedding)

                    axis_embedding, arg_embedding = self.cone_proj(axis_embedding, arg_embedding, axis_r_embedding, arg_r_embedding)
                idx += 1
        else:
            axis_embedding_list = []
            arg_embedding_list = []
            for i in range(len(query_structure)):
                axis_embedding, arg_embedding, idx = self.embed_query_cone(queries, query_structure[0], idx, query_sequence_embedding, whole_query_structure)
                axis_embedding_list.append(axis_embedding)
                arg_embedding_list.append(arg_embedding)

            stacked_axis_embeddings = torch.stack(axis_embedding_list)
            stacked_arg_embeddings = torch.stack(arg_embedding_list)

            axis_embedding, arg_embedding = self.cone_intersection(stacked_axis_embeddings, stacked_arg_embeddings)

        return axis_embedding, arg_embedding, idx

    def embed_query_box(self, queries, query_structure, idx, query_sequence_embedding, whole_query_structure):
        all_relation_flag = True
        for ele in query_structure[-1]:
            if ele not in ['r', 'n']:
                all_relation_flag = False
                break
        if all_relation_flag:
            if query_structure[0] == 'e':
                prompt = query_sequence_embedding[:, idx_mask[whole_query_structure][idx]]
                embedding = self.embedding_fusing(node=queries[:, idx], prompt=prompt)
                if self.use_cuda:
                    offset_embedding = torch.zeros_like(embedding).cuda()
                else:
                    offset_embedding = torch.zeros_like(embedding)
                idx += 1
            else:
                embedding, offset_embedding, idx = self.embed_query_box(
                    queries, query_structure[0], idx, query_sequence_embedding, whole_query_structure)
            for i in range(len(query_structure[-1])):
                if query_structure[-1][i] == 'n':
                    assert False, "box cannot handle queries with negation"
                else:
                    r_embedding = torch.index_select(self.relation_embedding, dim=0, index=queries[:, idx])
                    r_offset_embedding = torch.index_select(self.offset_embedding, dim=0, index=queries[:, idx])
                    embedding += r_embedding
                    offset_embedding += self.func(r_offset_embedding)
                idx += 1
        else:
            embedding_list = []
            offset_embedding_list = []
            for i in range(len(query_structure)):
                embedding, offset_embedding, idx = self.embed_query_box(
                    queries, query_structure[i], idx, query_sequence_embedding, whole_query_structure)
                embedding_list.append(embedding)
                offset_embedding_list.append(offset_embedding)
            embedding = self.center_net(torch.stack(embedding_list))
            offset_embedding = self.offset_net(torch.stack(offset_embedding_list))

        return embedding, offset_embedding, idx

    def embed_query_vec(self, queries, query_structure, idx, query_sequence_embedding, whole_query_structure):
        all_relation_flag = True
        for ele in query_structure[-1]:
            if ele not in ['r', 'n']:
                all_relation_flag = False
                break
        if all_relation_flag:
            if query_structure[0] == 'e':
                prompt = query_sequence_embedding[:, idx_mask[whole_query_structure][idx]]
                embedding = self.embedding_fusing(queries[:, idx], prompt)
                idx += 1
            else:
                embedding, idx = self.embed_query_vec(queries, query_structure[0], idx, query_sequence_embedding, whole_query_structure)
            for i in range(len(query_structure[-1])):
                if query_structure[-1][i] == 'n':
                    assert False, "vec cannot handle queries with negation"
                else:
                    r_embedding = torch.index_select(self.relation_embedding, dim=0, index=queries[:, idx])
                    embedding += r_embedding
                idx += 1
        else:
            embedding_list = []
            for i in range(len(query_structure)):
                embedding, idx = self.embed_query_vec(queries, query_structure[i], idx, query_sequence_embedding, whole_query_structure)
                embedding_list.append(embedding)
            embedding = self.center_net(torch.stack(embedding_list))

        return embedding, idx

    def embed_query_beta(self, queries, query_structure, idx, query_sequence_embedding, whole_query_structure):
        all_relation_flag = True
        for ele in query_structure[-1]:
            if ele not in ['r', 'n']:
                all_relation_flag = False
                break
        if all_relation_flag:
            if query_structure[0] == 'e':
                embedding = self.entity_regularizer(torch.index_select(self.entity_embedding, dim=0, index=queries[:, idx]))
                prompt = query_sequence_embedding[:, idx_mask[whole_query_structure][idx]]

                idx += 1
            else:
                alpha_embedding, beta_embedding, idx = self.embed_query_beta(
                    queries, query_structure[0], idx, query_sequence_embedding, whole_query_structure)
                embedding = torch.cat([alpha_embedding, beta_embedding], dim=-1)
            for i in range(len(query_structure[-1])):
                if query_structure[-1][i] == 'n':
                    assert (queries[:, idx] == -2).all()
                    embedding = 1./embedding
                else:
                    r_embedding = torch.index_select(self.relation_embedding, dim=0, index=queries[:, idx])
                    embedding = self.projection_net(embedding, r_embedding)
                idx += 1
            alpha_embedding, beta_embedding = torch.chunk(embedding, 2, dim=-1)
        else:
            alpha_embedding_list = []
            beta_embedding_list = []
            for i in range(len(query_structure)):
                alpha_embedding, beta_embedding, idx = self.embed_query_beta(
                    queries, query_structure[i], idx, query_sequence_embedding, whole_query_structure)
                alpha_embedding_list.append(alpha_embedding)
                beta_embedding_list.append(beta_embedding)
            alpha_embedding, beta_embedding = self.center_net(torch.stack(alpha_embedding_list), torch.stack(beta_embedding_list))

        return alpha_embedding, beta_embedding, idx

    def embed_query_ns(self, queries, query_structure, idx, query_sequence_embedding, whole_query_structure):
        all_relation_flag = True
        for ele in query_structure[-1]:
            if ele not in ['r', 'n']:
                all_relation_flag = False
                break
        if all_relation_flag:
            if query_structure[0] == 'e':
                prompt = query_sequence_embedding[:, idx_mask[whole_query_structure][idx]]
                embedding = self.embedding_fusing(node=queries[:, idx], prompt=prompt)
                vector = F.one_hot(queries[:, idx], num_classes=self.nentity).float()
                idx += 1
            else:
                embedding, vector, idx, v2b_logit = self.embed_query_ns(
                    queries, query_structure[0], idx, query_sequence_embedding, whole_query_structure)
            for i in range(len(query_structure[-1])):
                if query_structure[-1][i] == 'n':
                    vector = 10/self.nentity - vector

                    vector = self.my_norm(vector)
                    embedding = self.vec2emb(vector)
                else:
                    r_embedding = torch.index_select(self.relation_embedding, dim=0, index=queries[:, idx])
                    embedding = KGEcalculate(self.KGEmode, embedding, r_embedding, self.embedding_range)
                    mat_list = queries[:, idx]
                    vector = torch.stack([torch.sparse.mm(self.mat[mat_list[i]].t(), vector[i].unsqueeze(1)).squeeze(1)
                                          for i in range(len(mat_list))])
                    vector = self.my_norm(vector)

                    v2b_logit = self.gamma
                    if not self.args.pre_1p:
                        vector, embedding, v2b_logit = self.enhance(vector, embedding, mat_list)

                idx += 1
        else:
            embedding_list = []
            vector_list = []
            for i in range(len(query_structure)):
                embedding, vector, idx, v2b_logit = self.embed_query_ns(
                    queries, query_structure[i], idx, query_sequence_embedding, whole_query_structure)
                embedding_list.append(embedding)
                vector_list.append(vector)
            vector = self.vec_intersection(vector_list)
            embedding = self.vec2emb(vector)

        return embedding, vector, idx, v2b_logit

    def my_norm(self, vector):
        vector11 = vector.masked_fill(vector < self.thr, 0) / torch.max(self.thr, torch.sum(vector, dim=-1).unsqueeze(-1))
        return vector11

    def vec2emb(self, vector):
        atte = self.FFN_vec2emb(self.entity_embedding)
        atte = vector * atte.squeeze()
        atte = self.my_norm(atte)
        embedding = atte @ self.entity_embedding

        return embedding

    def emb2vec(self, vector, embedding, rels):
        distance = self.entity_embedding.unsqueeze(0) - embedding.unsqueeze(1)
        distance = self.gamma - torch.norm(distance, p=1, dim=-1)

        val, _ = torch.sort(distance, dim=-1)

        ind1 = torch.tensor([i for i in range(len(rels))]).cuda()
        cnt = vector.bool().sum(-1).long()*2
        thr = val[(ind1, cnt)].unsqueeze(-1)
        distance = distance.masked_fill(distance <= thr, -1e20)
        vector_inf = torch.softmax(distance, dim=-1)
        vector = vector + vector_inf

        vector = self.my_norm(vector)

        return vector, vector_inf

    def enhance(self, vector, embedding, mat_list):
        vector, vector_inf = self.emb2vec(vector, embedding, mat_list)
        embedding_new = self.vec2emb(vector)

        embedding_check = self.vec2emb(vector_inf)

        return vector, embedding_new, self.cal_logit_ns(embedding, None, embedding_check, None)

    def vec_intersection(self, vector_list):
        cnt = vector_list[0].bool()
        calc_res = vector_list[0]
        for i in range(1, len(vector_list)):
            cnt = cnt + vector_list[i].bool()
            calc_res = calc_res * vector_list[i]

        vector = self.my_norm(calc_res / len(vector_list))
        return vector

    def transform_union_query(self, queries, query_structure):
        if self.query_name_dict[query_structure] == '2u-DNF':
            queries = queries[:, :-1]
        elif self.query_name_dict[query_structure] == 'up-DNF':
            queries = torch.cat([torch.cat([queries[:, :2], queries[:, 5:6]], dim=1), torch.cat([queries[:, 2:4], queries[:, 5:6]], dim=1)], dim=1)
        queries = torch.reshape(queries, [queries.shape[0]*2, -1])
        return queries

    def transform_union_structure(self, query_structure):
        if self.query_name_dict[query_structure] == '2u-DNF':
            return ('e', ('r',))
        elif self.query_name_dict[query_structure] == 'up-DNF':
            return ('e', ('r', 'r'))

    def cal_logit_beta(self, entity_embedding, query_dist):
        alpha_embedding, beta_embedding = torch.chunk(entity_embedding, 2, dim=-1)
        entity_dist = torch.distributions.beta.Beta(alpha_embedding, beta_embedding)
        logit = self.gamma - torch.norm(torch.distributions.kl.kl_divergence(entity_dist, query_dist), p=1, dim=-1)
        return logit

    def forward_beta(self, positive_sample, negative_sample, subsampling_weight, batch_queries_dict, batch_idxs_dict):
        all_idxs, all_alpha_embeddings, all_beta_embeddings = [], [], []
        all_union_idxs, all_union_alpha_embeddings, all_union_beta_embeddings = [], [], []
        for query_structure in batch_queries_dict:
            if 'u' in self.query_name_dict[query_structure] and 'DNF' in self.query_name_dict[query_structure]:
                alpha_embedding, beta_embedding, _ = \
                    self.embed_query_beta(self.transform_union_query(batch_queries_dict[query_structure],
                                                                     query_structure),
                                          self.transform_union_structure(query_structure),
                                          0, self.query_sequence_embedding[query_structure], query_structure)
                all_union_idxs.extend(batch_idxs_dict[query_structure])
                all_union_alpha_embeddings.append(alpha_embedding)
                all_union_beta_embeddings.append(beta_embedding)
            else:
                alpha_embedding, beta_embedding, _ = self.embed_query_beta(batch_queries_dict[query_structure],
                                                                           query_structure,
                                                                           0, self.query_sequence_embedding[query_structure], query_structure)
                all_idxs.extend(batch_idxs_dict[query_structure])
                all_alpha_embeddings.append(alpha_embedding)
                all_beta_embeddings.append(beta_embedding)

        if len(all_alpha_embeddings) > 0:
            all_alpha_embeddings = torch.cat(all_alpha_embeddings, dim=0).unsqueeze(1)
            all_beta_embeddings = torch.cat(all_beta_embeddings, dim=0).unsqueeze(1)
            all_dists = torch.distributions.beta.Beta(all_alpha_embeddings, all_beta_embeddings)
        if len(all_union_alpha_embeddings) > 0:
            all_union_alpha_embeddings = torch.cat(all_union_alpha_embeddings, dim=0).unsqueeze(1)
            all_union_beta_embeddings = torch.cat(all_union_beta_embeddings, dim=0).unsqueeze(1)
            all_union_alpha_embeddings = all_union_alpha_embeddings.view(all_union_alpha_embeddings.shape[0]//2, 2, 1, -1)
            all_union_beta_embeddings = all_union_beta_embeddings.view(all_union_beta_embeddings.shape[0]//2, 2, 1, -1)
            all_union_dists = torch.distributions.beta.Beta(all_union_alpha_embeddings, all_union_beta_embeddings)

        if type(subsampling_weight) != type(None):
            subsampling_weight = subsampling_weight[all_idxs+all_union_idxs]

        if type(positive_sample) != type(None):
            if len(all_alpha_embeddings) > 0:
                positive_sample_regular = positive_sample[all_idxs]
                positive_embedding = self.entity_regularizer(torch.index_select(
                    self.entity_embedding, dim=0, index=positive_sample_regular).unsqueeze(1))
                positive_logit = self.cal_logit_beta(positive_embedding, all_dists)
            else:
                positive_logit = torch.Tensor([]).cuda()

            if len(all_union_alpha_embeddings) > 0:
                positive_sample_union = positive_sample[all_union_idxs]
                positive_embedding = self.entity_regularizer(torch.index_select(
                    self.entity_embedding, dim=0, index=positive_sample_union).unsqueeze(1).unsqueeze(1))
                positive_union_logit = self.cal_logit_beta(positive_embedding, all_union_dists)
                positive_union_logit = torch.max(positive_union_logit, dim=1)[0]
            else:
                positive_union_logit = torch.Tensor([]).cuda()
            positive_logit = torch.cat([positive_logit, positive_union_logit], dim=0)
        else:
            positive_logit = None

        if type(negative_sample) != type(None):
            if len(all_alpha_embeddings) > 0:
                negative_sample_regular = negative_sample[all_idxs]
                batch_size, negative_size = negative_sample_regular.shape
                negative_embedding = self.entity_regularizer(torch.index_select(
                    self.entity_embedding, dim=0, index=negative_sample_regular.view(-1)).view(batch_size, negative_size, -1))
                negative_logit = self.cal_logit_beta(negative_embedding, all_dists)
            else:
                negative_logit = torch.Tensor([]).cuda()

            if len(all_union_alpha_embeddings) > 0:
                negative_sample_union = negative_sample[all_union_idxs]
                batch_size, negative_size = negative_sample_union.shape
                negative_embedding = self.entity_regularizer(torch.index_select(
                    self.entity_embedding, dim=0, index=negative_sample_union.view(-1)).view(batch_size, 1, negative_size, -1))
                negative_union_logit = self.cal_logit_beta(negative_embedding, all_union_dists)
                negative_union_logit = torch.max(negative_union_logit, dim=1)[0]
            else:
                negative_union_logit = torch.Tensor([]).cuda()
            negative_logit = torch.cat([negative_logit, negative_union_logit], dim=0)
        else:
            negative_logit = None

        return positive_logit, negative_logit, subsampling_weight, all_idxs+all_union_idxs

    def cal_logit_box(self, entity_embedding, query_center_embedding, query_offset_embedding):
        delta = (entity_embedding - query_center_embedding).abs()
        distance_out = F.relu(delta - query_offset_embedding)
        distance_in = torch.min(delta, query_offset_embedding)
        logit = self.gamma - torch.norm(distance_out, p=1, dim=-1) - self.cen * torch.norm(distance_in, p=1, dim=-1)
        return logit

    def forward_box(self, positive_sample, negative_sample, subsampling_weight, batch_queries_dict, batch_idxs_dict):
        all_center_embeddings, all_offset_embeddings, all_idxs = [], [], []
        all_union_center_embeddings, all_union_offset_embeddings, all_union_idxs = [], [], []
        for query_structure in batch_queries_dict:
            if 'u' in self.query_name_dict[query_structure]:
                center_embedding, offset_embedding, _ = \
                    self.embed_query_box(self.transform_union_query(batch_queries_dict[query_structure],
                                                                    query_structure),
                                         self.transform_union_structure(query_structure),
                                         0, self.query_sequence_embedding[query_structure], query_structure)
                all_union_center_embeddings.append(center_embedding)
                all_union_offset_embeddings.append(offset_embedding)
                all_union_idxs.extend(batch_idxs_dict[query_structure])
            else:
                center_embedding, offset_embedding, _ = self.embed_query_box(batch_queries_dict[query_structure],
                                                                             query_structure,
                                                                             0, self.query_sequence_embedding[query_structure], query_structure)
                all_center_embeddings.append(center_embedding)
                all_offset_embeddings.append(offset_embedding)
                all_idxs.extend(batch_idxs_dict[query_structure])

        if len(all_center_embeddings) > 0 and len(all_offset_embeddings) > 0:
            all_center_embeddings = torch.cat(all_center_embeddings, dim=0).unsqueeze(1)
            all_offset_embeddings = torch.cat(all_offset_embeddings, dim=0).unsqueeze(1)
        if len(all_union_center_embeddings) > 0 and len(all_union_offset_embeddings) > 0:
            all_union_center_embeddings = torch.cat(all_union_center_embeddings, dim=0).unsqueeze(1)
            all_union_offset_embeddings = torch.cat(all_union_offset_embeddings, dim=0).unsqueeze(1)
            all_union_center_embeddings = all_union_center_embeddings.view(all_union_center_embeddings.shape[0]//2, 2, 1, -1)
            all_union_offset_embeddings = all_union_offset_embeddings.view(all_union_offset_embeddings.shape[0]//2, 2, 1, -1)

        if type(subsampling_weight) != type(None):
            subsampling_weight = subsampling_weight[all_idxs+all_union_idxs]

        if type(positive_sample) != type(None):
            if len(all_center_embeddings) > 0:
                positive_sample_regular = positive_sample[all_idxs]
                positive_embedding = self.embedding_fusing(node=positive_sample_regular, prompt=self.no_union_prompt).unsqueeze(1)
                positive_logit = self.cal_logit_box(positive_embedding, all_center_embeddings, all_offset_embeddings)
            else:
                positive_logit = torch.Tensor([]).cuda()

            if len(all_union_center_embeddings) > 0:
                positive_sample_union = positive_sample[all_union_idxs]
                positive_embedding = self.embedding_fusing(node=positive_sample_union, prompt=self.union_prompt).unsqueeze(1).unsqueeze(1)
                positive_union_logit = self.cal_logit_box(positive_embedding, all_union_center_embeddings, all_union_offset_embeddings)
                positive_union_logit = torch.max(positive_union_logit, dim=1)[0]
            else:
                positive_union_logit = torch.Tensor([]).cuda()
            positive_logit = torch.cat([positive_logit, positive_union_logit], dim=0)
        else:
            positive_logit = None

        if type(negative_sample) != type(None):
            if len(all_center_embeddings) > 0:
                negative_sample_regular = negative_sample[all_idxs]
                batch_size, negative_size = negative_sample_regular.shape
                negative_embedding = self.embedding_fusing(node=negative_sample_regular.view(-1),
                                                           prompt=self.no_union_prompt).view(batch_size, negative_size, -1)
                negative_logit = self.cal_logit_box(negative_embedding, all_center_embeddings, all_offset_embeddings)
            else:
                negative_logit = torch.Tensor([]).cuda()

            if len(all_union_center_embeddings) > 0:
                negative_sample_union = negative_sample[all_union_idxs]
                batch_size, negative_size = negative_sample_union.shape
                negative_embedding = self.embedding_fusing(node=negative_sample_union.view(-1),
                                                           prompt=self.union_prompt).view(batch_size, 1, negative_size, -1)
                negative_union_logit = self.cal_logit_box(negative_embedding, all_union_center_embeddings, all_union_offset_embeddings)
                negative_union_logit = torch.max(negative_union_logit, dim=1)[0]
            else:
                negative_union_logit = torch.Tensor([]).cuda()
            negative_logit = torch.cat([negative_logit, negative_union_logit], dim=0)
        else:
            negative_logit = None

        return positive_logit, negative_logit, subsampling_weight, all_idxs+all_union_idxs

    def cal_logit_vec(self, entity_embedding, query_embedding):
        distance = entity_embedding - query_embedding
        logit = self.gamma - torch.norm(distance, p=1, dim=-1)
        return logit

    def forward_vec(self, positive_sample, negative_sample, subsampling_weight, batch_queries_dict, batch_idxs_dict):
        all_center_embeddings, all_idxs = [], []
        all_union_center_embeddings, all_union_idxs = [], []

        for query_structure in batch_queries_dict:
            if 'u' in self.query_name_dict[query_structure]:
                center_embedding, _ = self.embed_query_vec(self.transform_union_query(batch_queries_dict[query_structure],
                                                                                      query_structure),
                                                           self.transform_union_structure(query_structure), 0, self.query_sequence_embedding[query_structure], query_structure)
                all_union_center_embeddings.append(center_embedding)
                all_union_idxs.extend(batch_idxs_dict[query_structure])
            else:
                center_embedding, _ = self.embed_query_vec(
                    batch_queries_dict[query_structure], query_structure, 0, self.query_sequence_embedding[query_structure], query_structure)
                all_center_embeddings.append(center_embedding)
                all_idxs.extend(batch_idxs_dict[query_structure])

        if len(all_center_embeddings) > 0:
            all_center_embeddings = torch.cat(all_center_embeddings, dim=0).unsqueeze(1)
        if len(all_union_center_embeddings) > 0:
            all_union_center_embeddings = torch.cat(all_union_center_embeddings, dim=0).unsqueeze(1)
            all_union_center_embeddings = all_union_center_embeddings.view(all_union_center_embeddings.shape[0]//2, 2, 1, -1)

        if type(subsampling_weight) != type(None):
            subsampling_weight = subsampling_weight[all_idxs+all_union_idxs]

        if type(positive_sample) != type(None):
            if len(all_center_embeddings) > 0:
                positive_sample_regular = positive_sample[all_idxs]
                positive_embedding = self.embedding_fusing(node=positive_sample_regular, prompt=self.no_union_prompt).unsqueeze(1)
                positive_logit = self.cal_logit_vec(positive_embedding, all_center_embeddings)
            else:
                positive_logit = torch.Tensor([]).cuda()

            if len(all_union_center_embeddings) > 0:
                positive_sample_union = positive_sample[all_union_idxs]
                positive_embedding = self.embedding_fusing(node=positive_sample_union, prompt=self.union_prompt).unsqueeze(1).unsqueeze(1)
                positive_union_logit = self.cal_logit_vec(positive_embedding, all_union_center_embeddings)
                positive_union_logit = torch.max(positive_union_logit, dim=1)[0]
            else:
                positive_union_logit = torch.Tensor([]).cuda()
            positive_logit = torch.cat([positive_logit, positive_union_logit], dim=0)
        else:
            positive_logit = None

        if type(negative_sample) != type(None):
            if len(all_center_embeddings) > 0:
                negative_sample_regular = negative_sample[all_idxs]
                batch_size, negative_size = negative_sample_regular.shape
                negative_embedding = self.embedding_fusing(node=negative_sample_regular.view(-1),
                                                           prompt=self.no_union_prompt).view(batch_size, negative_size, -1)
                negative_logit = self.cal_logit_vec(negative_embedding, all_center_embeddings)
            else:
                negative_logit = torch.Tensor([]).cuda()

            if len(all_union_center_embeddings) > 0:
                negative_sample_union = negative_sample[all_union_idxs]
                batch_size, negative_size = negative_sample_union.shape
                negative_embedding = self.embedding_fusing(node=negative_sample_union.view(-1),
                                                           prompt=self.union_prompt).view(batch_size, 1, negative_size, -1)
                negative_union_logit = self.cal_logit_vec(negative_embedding, all_union_center_embeddings)
                negative_union_logit = torch.max(negative_union_logit, dim=1)[0]
            else:
                negative_union_logit = torch.Tensor([]).cuda()
            negative_logit = torch.cat([negative_logit, negative_union_logit], dim=0)
        else:
            negative_logit = None

        return positive_logit, negative_logit, subsampling_weight, all_idxs+all_union_idxs

    def cal_logit_ns(self, entity_embedding, entity_vector, query_embedding, query_vector):
        embedding_logit = KGELoss(self.KGEmode, entity_embedding, query_embedding, self.gamma, self.phase_weight, self.modules_weight)

        if entity_vector != None:
            vector_logit = torch.sum(entity_vector * torch.log(torch.max(self.thr, query_vector)), dim=-1)

            return embedding_logit, vector_logit
        else:
            return embedding_logit

    def forward_ns(self, positive_sample, negative_sample, subsampling_weight, batch_queries_dict, batch_idxs_dict):
        all_center_embeddings, all_center_vectors, all_idxs = [], [], []
        all_union_center_embeddings, all_union_center_vectors, all_union_idxs = [], [], []
        all_v2b_logit = []
        for query_structure in batch_queries_dict:
            if 'u' in self.query_name_dict[query_structure]:
                center_embedding, center_vector, _, v2b_logit = self.embed_query_ns(self.transform_union_query(batch_queries_dict[query_structure],
                                                                                                               query_structure),
                                                                                    self.transform_union_structure(query_structure), 0, self.query_sequence_embedding[query_structure], query_structure)
                all_union_center_embeddings.append(center_embedding)
                all_union_center_vectors.append(center_vector)
                all_union_idxs.extend(batch_idxs_dict[query_structure])
                all_v2b_logit.append(v2b_logit)
            else:
                center_embedding, center_vector, _, v2b_logit = self.embed_query_ns(
                    batch_queries_dict[query_structure], query_structure, 0, self.query_sequence_embedding[query_structure], query_structure)
                all_center_embeddings.append(center_embedding)
                all_center_vectors.append(center_vector)
                all_idxs.extend(batch_idxs_dict[query_structure])
                all_v2b_logit.append(v2b_logit)

        if len(all_center_embeddings) > 0:
            all_center_embeddings = torch.cat(all_center_embeddings, dim=0).unsqueeze(1)
            all_center_vectors = torch.cat(all_center_vectors, dim=0).unsqueeze(1)
        if len(all_union_center_embeddings) > 0:
            all_union_center_embeddings = torch.cat(all_union_center_embeddings, dim=0).unsqueeze(1)
            all_union_center_vectors = torch.cat(all_union_center_vectors, dim=0).unsqueeze(1)
            all_union_center_embeddings = all_union_center_embeddings.view(all_union_center_embeddings.shape[0]//2, 2, 1, -1)
            all_union_center_vectors = all_union_center_vectors.view(all_union_center_vectors.shape[0]//2, 2, 1, -1)
            all_union_center_vectors = all_union_center_vectors.sum(dim=1).unsqueeze(1)
            all_union_center_vectors = self.my_norm(all_union_center_vectors)

        if len(all_center_embeddings) > 0 and len(all_union_center_embeddings) > 0:
            vectors = torch.cat((all_center_vectors.squeeze(1), all_union_center_vectors.squeeze(1).squeeze(1)), dim=0)
        elif len(all_center_embeddings) > 0:
            vectors = all_center_vectors.squeeze(1)
        else:
            vectors = all_union_center_vectors.squeeze(1).squeeze(1)

        if type(subsampling_weight) != type(None):
            subsampling_weight = subsampling_weight[all_idxs+all_union_idxs]

        if type(positive_sample) != type(None):
            if len(all_center_embeddings) > 0:
                positive_sample_regular = positive_sample[all_idxs]
                positive_embedding = self.embedding_fusing(node=positive_sample_regular, prompt=self.no_union_prompt).unsqueeze(1)
                positive_vector = F.one_hot(positive_sample_regular, num_classes=self.nentity).unsqueeze(1)
                positive_logit, vector_logit = self.cal_logit_ns(positive_embedding, positive_vector, all_center_embeddings, all_center_vectors)

            else:
                positive_logit = torch.Tensor([]).cuda()
                vector_logit = torch.Tensor([]).cuda()

            if len(all_union_center_embeddings) > 0:
                positive_sample_union = positive_sample[all_union_idxs]
                positive_embedding = self.embedding_fusing(node=positive_sample_union, prompt=self.union_prompt).unsqueeze(1).unsqueeze(1)
                positive_vector = F.one_hot(positive_sample_union, num_classes=self.nentity).unsqueeze(1).unsqueeze(1)
                all_union_center_embeddings = self.vec2emb(all_union_center_vectors.squeeze()).unsqueeze(1).unsqueeze(1)
                positive_union_logit, vector_union_logit = self.cal_logit_ns(
                    positive_embedding, positive_vector, all_union_center_embeddings, all_union_center_vectors)

            else:
                positive_union_logit = torch.Tensor([]).cuda()
                vector_union_logit = torch.Tensor([]).cuda()

            positive_logit = torch.cat([positive_logit, positive_union_logit], dim=0)
            vector_logit = torch.cat([vector_logit, vector_union_logit], dim=1)
        else:
            positive_logit = None
            vector_logit = None

        if type(negative_sample) != type(None):

            if len(all_center_embeddings) > 0:
                negative_sample_regular = negative_sample[all_idxs]
                batch_size, negative_size = negative_sample_regular.shape
                negative_embedding = self.embedding_fusing(node=negative_sample_regular.view(-1),
                                                           prompt=self.no_union_prompt).view(batch_size, negative_size, -1)
                negative_vector = None
                negative_logit = self.cal_logit_ns(negative_embedding, negative_vector, all_center_embeddings, all_center_vectors)
            else:
                negative_logit = torch.Tensor([]).cuda()

            if len(all_union_center_embeddings) > 0:
                negative_sample_union = negative_sample[all_union_idxs]
                batch_size, negative_size = negative_sample_union.shape
                negative_embedding = self.embedding_fusing(node=negative_sample_union.view(-1),
                                                           prompt=self.union_prompt).view(batch_size, 1, negative_size, -1)
                negative_vector = None
                negative_union_logit = self.cal_logit_ns(negative_embedding, negative_vector, all_union_center_embeddings, all_union_center_vectors)
                negative_union_logit = torch.max(negative_union_logit, dim=1)[0]
            else:
                negative_union_logit = torch.Tensor([]).cuda()
            negative_logit = torch.cat([negative_logit, negative_union_logit], dim=0)
        else:
            negative_logit = None

        return vectors, vector_logit, torch.cat(all_v2b_logit, dim=0).unsqueeze(-1), positive_logit, negative_logit, subsampling_weight, all_idxs+all_union_idxs
    
    def cal_logit_cone(self, entity_embedding, query_axis_embedding, query_arg_embedding):
        delta1 = entity_embedding - (query_axis_embedding - query_arg_embedding)
        delta2 = entity_embedding - (query_axis_embedding + query_arg_embedding)

        distance2axis = torch.abs(torch.sin((entity_embedding - query_axis_embedding) / 2))
        distance_base = torch.abs(torch.sin(query_arg_embedding / 2))

        indicator_in = distance2axis < distance_base
        distance_out = torch.min(torch.abs(torch.sin(delta1 / 2)), torch.abs(torch.sin(delta2 / 2)))
        distance_out[indicator_in] = 0.

        distance_in = torch.min(distance2axis, distance_base)

        distance = torch.norm(distance_out, p=1, dim=-1) + self.cen * torch.norm(distance_in, p=1, dim=-1)
        logit = self.gamma - distance * self.modulus

        return logit
    
    def forward_cone(self, positive_sample, negative_sample, subsampling_weight, batch_queries_dict, batch_idxs_dict):
        all_idxs, all_axis_embeddings, all_arg_embeddings = [], [], []
        all_union_idxs, all_union_axis_embeddings, all_union_arg_embeddings = [], [], []
        for query_structure in batch_queries_dict:
            if 'u' in self.query_name_dict[query_structure] and 'DNF' in self.query_name_dict[query_structure]:
                axis_embedding, arg_embedding, _ = \
                    self.embed_query_cone(self.transform_union_query(batch_queries_dict[query_structure], query_structure), self.transform_union_structure(query_structure),  0, self.query_sequence_embedding[query_structure], query_structure)
                all_union_idxs.extend(batch_idxs_dict[query_structure])
                all_union_axis_embeddings.append(axis_embedding)
                all_union_arg_embeddings.append(arg_embedding)
            else:
                axis_embedding, arg_embedding, _ = self.embed_query_cone(batch_queries_dict[query_structure], query_structure,  0, self.query_sequence_embedding[query_structure], query_structure)
                all_idxs.extend(batch_idxs_dict[query_structure])
                all_axis_embeddings.append(axis_embedding)
                all_arg_embeddings.append(arg_embedding)

        if len(all_axis_embeddings) > 0:
            all_axis_embeddings = torch.cat(all_axis_embeddings, dim=0).unsqueeze(1)
            all_arg_embeddings = torch.cat(all_arg_embeddings, dim=0).unsqueeze(1)
        if len(all_union_axis_embeddings) > 0:
            all_union_axis_embeddings = torch.cat(all_union_axis_embeddings, dim=0).unsqueeze(1)
            all_union_arg_embeddings = torch.cat(all_union_arg_embeddings, dim=0).unsqueeze(1)
            all_union_axis_embeddings = all_union_axis_embeddings.view(
                all_union_axis_embeddings.shape[0] // 2, 2, 1, -1)
            all_union_arg_embeddings = all_union_arg_embeddings.view(
                all_union_arg_embeddings.shape[0] // 2, 2, 1, -1)
        if type(subsampling_weight) != type(None):
            subsampling_weight = subsampling_weight[all_idxs + all_union_idxs]

        if type(positive_sample) != type(None):
            if len(all_axis_embeddings) > 0:
                positive_sample_regular = positive_sample[all_idxs]
                positive_embedding = self.embedding_fusing(node=positive_sample_regular, prompt=self.no_union_prompt).unsqueeze(1)


                positive_embedding = self.angle_scale(positive_embedding, self.axis_scale)
                positive_embedding = convert_to_axis(positive_embedding)

                positive_logit = self.cal_logit_cone(positive_embedding, all_axis_embeddings, all_arg_embeddings)
            else:
                positive_logit = torch.Tensor([]).cuda()


            if len(all_union_axis_embeddings) > 0:
                positive_sample_union = positive_sample[all_union_idxs]
                positive_embedding = self.embedding_fusing(node=positive_sample_union, prompt = self.union_prompt).unsqueeze(1).unsqueeze(1)


                positive_embedding = self.angle_scale(positive_embedding, self.axis_scale)
                positive_embedding = convert_to_axis(positive_embedding)

                positive_union_logit = self.cal_logit_cone(positive_embedding, all_union_axis_embeddings, all_union_arg_embeddings)

                positive_union_logit = torch.max(positive_union_logit, dim=1)[0]
            else:
                positive_union_logit = torch.Tensor([]).cuda()
            positive_logit = torch.cat([positive_logit, positive_union_logit], dim=0)
        else:
            positive_logit = None

        if type(negative_sample) != type(None):
            if len(all_axis_embeddings) > 0:
                negative_sample_regular = negative_sample[all_idxs]
                batch_size, negative_size = negative_sample_regular.shape
                negative_embedding = self.embedding_fusing(node=negative_sample_regular.view(-1), prompt=self.no_union_prompt).view(batch_size, negative_size, -1)
                negative_embedding = self.angle_scale(negative_embedding, self.axis_scale)
                negative_embedding = convert_to_axis(negative_embedding)

                negative_logit = self.cal_logit_cone(negative_embedding, all_axis_embeddings, all_arg_embeddings)
            else:
                negative_logit = torch.Tensor([]).cuda()

            if len(all_union_axis_embeddings) > 0:
                negative_sample_union = negative_sample[all_union_idxs]
                batch_size, negative_size = negative_sample_union.shape
                negative_embedding = self.embedding_fusing(node=negative_sample_union.view(-1), prompt=self.union_prompt).view(batch_size, 1, negative_size, -1)
                negative_embedding = self.angle_scale(negative_embedding, self.axis_scale)
                negative_embedding = convert_to_axis(negative_embedding)

                negative_union_logit = self.cal_logit_cone(negative_embedding, all_union_axis_embeddings, all_union_arg_embeddings)
                negative_union_logit = torch.max(negative_union_logit, dim=1)[0]
            else:
                negative_union_logit = torch.Tensor([]).cuda()
            negative_logit = torch.cat([negative_logit, negative_union_logit], dim=0)
        else:
            negative_logit = None

        return positive_logit, negative_logit, subsampling_weight, all_idxs + all_union_idxs

    def train_step(self, model, optimizer, train_iterator, args):
        model.train()
        optimizer.zero_grad()
        positive_sample, negative_sample, subsampling_weight, batch_queries, query_structures = next(train_iterator)

        batch_queries_dict = collections.defaultdict(list)
        batch_idxs_dict = collections.defaultdict(list)
        for i, query in enumerate(batch_queries):
            batch_queries_dict[query_structures[i]].append(query)
            batch_idxs_dict[query_structures[i]].append(i)
        for query_structure in batch_queries_dict:
            if args.cuda:
                batch_queries_dict[query_structure] = torch.LongTensor(batch_queries_dict[query_structure]).cuda()
            else:
                batch_queries_dict[query_structure] = torch.LongTensor(batch_queries_dict[query_structure])
        if args.cuda:
            positive_sample = positive_sample.cuda()
            negative_sample = negative_sample.cuda()
            subsampling_weight = subsampling_weight.cuda()

        self.query_sequence_embedding = dict()
        self.no_union_prompt = []
        self.union_prompt = []
        for query_structure in batch_queries_dict:
            one_query_structure_queries = []
            for oneQuery in batch_queries_dict[query_structure]:
                tmp = self.structure_sequence_id[query_structure].copy()
                for idx1, idx2 in relation_inplace[query_structure].items():
                    tmp[idx2] = int(oneQuery[idx1])
                one_query_structure_queries.append(tmp)

            bert_input = torch.tensor(one_query_structure_queries).cuda()

            bert_output = self.bert(bert_input).last_hidden_state
            self.query_sequence_embedding[query_structure] = bert_output
            if 'u' not in self.query_name_dict[query_structure]:
                self.no_union_prompt.append(bert_output[:, -1])
            else:
                self.union_prompt.append(bert_output[:, -1])

        if len(self.no_union_prompt) > 0:
            self.no_union_prompt = torch.cat(self.no_union_prompt, dim=0)
        if len(self.union_prompt) > 0:
            self.union_prompt = torch.cat(self.union_prompt, dim=0)

        if args.geo == 'ns':
            _, vector_logit, v2b_logit, positive_logit, negative_logit, subsampling_weight, _ = model(
                positive_sample, negative_sample, subsampling_weight, batch_queries_dict, batch_idxs_dict)
        else:
            positive_logit, negative_logit, subsampling_weight, _ = model(
                positive_sample, negative_sample, subsampling_weight, batch_queries_dict, batch_idxs_dict)

        v2b_loss = torch.tensor([0])
        if args.geo == 'ns':
            vector_score = F.logsigmoid(vector_logit).squeeze(dim=1)
            vector_loss = - (subsampling_weight * vector_score).sum()
            vector_loss /= subsampling_weight.sum()
            if args.new_loss:
                v2b_score = F.logsigmoid(v2b_logit).squeeze(dim=1)
                v2b_loss = -(subsampling_weight * v2b_score).sum()
                v2b_loss /= subsampling_weight.sum()
        negative_score = F.logsigmoid(-negative_logit).mean(dim=1)
        positive_score = F.logsigmoid(positive_logit).squeeze(dim=1)
        positive_sample_loss = - (subsampling_weight * positive_score).sum()
        negative_sample_loss = - (subsampling_weight * negative_score).sum()
        positive_sample_loss /= subsampling_weight.sum()
        negative_sample_loss /= subsampling_weight.sum()

        if args.geo == 'ns':
            if args.pre_1p:
                loss = (positive_sample_loss + negative_sample_loss)/2
            elif args.new_loss:
                loss = (vector_loss + positive_sample_loss + negative_sample_loss + v2b_loss)/4
            else:
                loss = (positive_sample_loss + negative_sample_loss + vector_loss)/3
        else:
            loss = (positive_sample_loss + negative_sample_loss)/2
        loss.backward()
        optimizer.step()
        if args.geo == 'ns':
            log = {
                'vector_loss': vector_loss.item(),
                'v2b_loss': v2b_loss.item(),
                'positive_sample_loss': positive_sample_loss.item(),
                'negative_sample_loss': negative_sample_loss.item(),
                'loss': loss.item(),
            }
        else:
            log = {
                'positive_sample_loss': positive_sample_loss.item(),
                'negative_sample_loss': negative_sample_loss.item(),
                'loss': loss.item(),
            }
        return log

    def test_step(self, model, easy_answers, answers, args, test_dataloader):

        step = 0
        total_steps = len(test_dataloader)
        logs = collections.defaultdict(list)

        with torch.no_grad():
            for negative_sample, queries, queries_unflatten, query_structures in tqdm(test_dataloader):
                batch_queries_dict = collections.defaultdict(list)
                batch_idxs_dict = collections.defaultdict(list)
                for i, query in enumerate(queries):
                    batch_queries_dict[query_structures[i]].append(query)
                    batch_idxs_dict[query_structures[i]].append(i)
                for query_structure in batch_queries_dict:
                    if args.cuda:
                        batch_queries_dict[query_structure] = torch.LongTensor(batch_queries_dict[query_structure]).cuda()
                    else:
                        batch_queries_dict[query_structure] = torch.LongTensor(batch_queries_dict[query_structure])
                if args.cuda:
                    negative_sample = negative_sample.cuda()

                self.query_sequence_embedding = dict()
                no_union_prompt = []
                union_prompt = []
                for query_structure in batch_queries_dict:
                    one_query_structure_queries = []
                    for oneQuery in batch_queries_dict[query_structure]:
                        tmp = self.structure_sequence_id[query_structure].copy()
                        for idx1, idx2 in relation_inplace[query_structure].items():
                            tmp[idx2] = int(oneQuery[idx1])
                        one_query_structure_queries.append(tmp)

                    bert_input = torch.tensor(one_query_structure_queries).cuda()

                    bert_output = self.bert(bert_input).last_hidden_state
                    self.query_sequence_embedding[query_structure] = bert_output
                    if 'u' not in self.query_name_dict[query_structure]:
                        no_union_prompt.append(bert_output[:, -1])
                    else:
                        union_prompt.append(bert_output[:, -1])

                if len(no_union_prompt) > 0:
                    self.no_union_prompt = torch.cat(no_union_prompt, dim=0)
                if len(union_prompt) > 0:
                    self.union_prompt = torch.cat(union_prompt, dim=0)

                if args.geo == 'ns':
                    vectors, _, _, _, negative_logit, _, idxs = model(None, negative_sample, None, batch_queries_dict, batch_idxs_dict)
                else:
                    _, negative_logit, _, idxs = model(None, negative_sample, None, batch_queries_dict, batch_idxs_dict)

                queries_unflatten = [queries_unflatten[i] for i in idxs]
                query_structures = [query_structures[i] for i in idxs]

                if args.gridsearch:
                    negative_logit = torch.softmax(negative_logit, dim=-1)
                    negative_logit = args.lam * negative_logit + (1 - args.lam) * vectors
                if args.lambdas:
                    negative_logit = torch.softmax(negative_logit, dim=-1)
                    lams = torch.tensor([args.lams[struct] for struct in query_structures]).cuda()
                    negative_logit = lams * negative_logit + (1 - lams) * vectors

                argsort = torch.argsort(negative_logit, dim=1, descending=True)
                ranking = argsort.clone().to(torch.float)
                if args.geo == 'ns':
                    vector_argsort = torch.argsort(vectors, dim=1, descending=True)
                    vector_ranking = vector_argsort.clone().to(torch.float)
                if len(argsort) == args.test_batch_size:
                    ranking = ranking.scatter_(1, argsort, model.batch_entity_range)
                    if args.geo == 'ns':
                        vector_ranking = vector_ranking.scatter_(1, vector_argsort, model.batch_entity_range)

                else:
                    if args.cuda:
                        ranking = ranking.scatter_(1,
                                                   argsort,
                                                   torch.arange(model.nentity).to(torch.float).repeat(argsort.shape[0], 1).cuda()
                                                   )
                        if args.geo == 'ns':
                            vector_ranking = vector_ranking.scatter_(1,
                                                                     vector_argsort,
                                                                     torch.arange(model.nentity).to(torch.float).repeat(
                                                                         vector_argsort.shape[0], 1).cuda()
                                                                     )
                    else:
                        ranking = ranking.scatter_(1,
                                                   argsort,
                                                   torch.arange(model.nentity).to(torch.float).repeat(argsort.shape[0], 1)
                                                   )
                        if args.geo == 'ns':
                            vector_ranking = vector_ranking.scatter_(1,
                                                                     vector_argsort,
                                                                     torch.arange(model.nentity).to(torch.float).repeat(vector_argsort.shape[0], 1)
                                                                     )
                for idx, (i, query, query_structure) in enumerate(zip(argsort[:, 0], queries_unflatten, query_structures)):
                    answer = answers[query]
                    easy_answer = easy_answers[query]
                    num_answer = len(answer)
                    num_easy = len(easy_answer)
                    cur_ranking = ranking[idx, list(easy_answer) + list(answer)]
                    cur_ranking, indices = torch.sort(cur_ranking)
                    masks = indices >= num_easy
                    if args.cuda:
                        answer_list = torch.arange(num_answer + num_easy).to(torch.float).cuda()
                    else:
                        answer_list = torch.arange(num_answer + num_easy).to(torch.float)
                    cur_ranking = cur_ranking - answer_list + 1
                    cur_ranking = cur_ranking[masks]

                    mrr = torch.mean(1./cur_ranking).item()
                    h1 = torch.mean((cur_ranking <= 1).to(torch.float)).item()
                    h3 = torch.mean((cur_ranking <= 3).to(torch.float)).item()
                    h10 = torch.mean((cur_ranking <= 10).to(torch.float)).item()

                    if args.geo != 'ns':
                        logs[query_structure].append({
                            'MRR': mrr,
                            'HITS1': h1,
                            'HITS3': h3,
                            'HITS10': h10,
                        })

                if args.geo == 'ns':

                    for idx, (i, query, query_structure) in enumerate(zip(vector_argsort[:, 0], queries_unflatten, query_structures)):
                        answer = answers[query]
                        easy_answer = easy_answers[query]
                        num_answer = len(answer)
                        num_easy = len(easy_answer)
                        cur_ranking = vector_ranking[idx, list(easy_answer) + list(answer)]
                        cur_ranking, indices = torch.sort(cur_ranking)
                        masks = indices >= num_easy
                        if args.cuda:
                            answer_list = torch.arange(num_answer + num_easy).to(torch.float).cuda()
                        else:
                            answer_list = torch.arange(num_answer + num_easy).to(torch.float)
                        cur_ranking = cur_ranking - answer_list + 1
                        cur_ranking = cur_ranking[masks]

                        vec_mrr = torch.mean(1./cur_ranking).item()
                        vec_h1 = torch.mean((cur_ranking <= 1).to(torch.float)).item()
                        vec_h3 = torch.mean((cur_ranking <= 3).to(torch.float)).item()
                        vec_h10 = torch.mean((cur_ranking <= 10).to(torch.float)).item()

                        logs[query_structure].append({
                            'MRR': mrr,
                            'HITS1': h1,
                            'HITS3': h3,
                            'HITS10': h10,
                            'vec_MRR': vec_mrr,
                            'vec_HITS1': vec_h1,
                            'vec_HITS3': vec_h3,
                            'vec_HITS10': vec_h10,
                        })

                if step % args.test_log_steps == 0:
                    logging.info('Evaluating the model... (%d/%d)' % (step, total_steps))

                step += 1

        metrics = collections.defaultdict(lambda: collections.defaultdict(int))
        for query_structure in logs:
            for metric in logs[query_structure][0].keys():
                if metric in ['num_hard_answer']:
                    continue
                metrics[query_structure][metric] = sum([log[metric] for log in logs[query_structure]])/len(logs[query_structure])
            metrics[query_structure]['num_queries'] = len(logs[query_structure])

        return metrics