#!/usr/bin/env python36
# -*- coding: utf-8 -*-

import datetime
import math
import time
import numpy as np
import torch
from torch import nn
from torch.nn import Module, Parameter
from parsers import opt
import torch.nn.functional as F
from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from collections import Counter
from parsers import opt
class GNN(Module):
    def __init__(self, hidden_size, step=1):
        super(GNN, self).__init__()
        self.step = step
        self.hidden_size = hidden_size
        if opt.method == 's_useoutput':
            self.input_size = hidden_size
        else:
            self.input_size = hidden_size * 2
        if opt.method =='modif_gru_large':
            self.gate_size = hidden_size
        elif opt.method =='modif_gru_use_torch' or  opt.method =='modif_gru_use_torch_modif' or opt.method=='new1_mixed_posneg_br_one' or opt.method=='GIISR' or  opt.method =='modif_rnn_use_torch_modif' or  opt.method =='modif_lstm_use_torch_modif'or  opt.method =='define_gru2' or  opt.method =='define_gru3' or  opt.method[:-1] =='define_gru':
            self.gate_size = hidden_size
        else:
            self.gate_size = 3 * hidden_size
        if opt.method=="define_gru2"  or  opt.method =='define_gru3' or  opt.method[:-1] =='define_gru' or opt.method=='new1_mixed_posneg_br_one' or opt.method=='GIISR':
            self.w_z = Parameter(torch.Tensor(self.gate_size, self.input_size))
            self.w_z2 = Parameter(torch.Tensor(self.gate_size, self.input_size))
            self.w_r = Parameter(torch.Tensor(self.gate_size, self.input_size))
            self.w_o = Parameter(torch.Tensor(self.gate_size, self.input_size))
            self.w_o2 = Parameter(torch.Tensor(self.gate_size, self.input_size))
            self.u_z = Parameter(torch.Tensor(self.gate_size, self.hidden_size))
            self.u_z2 = Parameter(torch.Tensor(self.gate_size, self.hidden_size))
            self.u_r = Parameter(torch.Tensor(self.gate_size, self.hidden_size))
            self.u_o = Parameter(torch.Tensor(self.gate_size, self.hidden_size))
            self.u_o2 = Parameter(torch.Tensor(self.gate_size, self.hidden_size))
            self.wb_z = Parameter(torch.Tensor(self.gate_size))
            self.wb_z2 = Parameter(torch.Tensor(self.gate_size))
            self.wb_r = Parameter(torch.Tensor(self.gate_size))
            self.wb_o = Parameter(torch.Tensor(self.gate_size))
            self.wb_o2 = Parameter(torch.Tensor(self.gate_size))
            self.ub_z = Parameter(torch.Tensor(self.gate_size))
            self.ub_z2 = Parameter(torch.Tensor(self.gate_size))
            self.ub_r = Parameter(torch.Tensor(self.gate_size))
            self.ub_o = Parameter(torch.Tensor(self.gate_size))
            self.ub_o2 = Parameter(torch.Tensor(self.gate_size))

        self.w_ih = Parameter(torch.Tensor(self.gate_size, self.input_size))
        self.w_hh = Parameter(torch.Tensor(self.gate_size, self.hidden_size))
        self.w_ih_two = Parameter(torch.Tensor(self.gate_size, self.input_size))
        self.w_hh_two = Parameter(torch.Tensor(self.gate_size, self.hidden_size))
        # self.w_ih_ut = Parameter(torch.Tensor(self.gate_size, self.input_size))
        # self.w_a = Parameter(torch.Tensor())
        self.b_ih = Parameter(torch.Tensor(self.gate_size))
        self.b_hh = Parameter(torch.Tensor(self.gate_size))
        self.b_ih_two = Parameter(torch.Tensor(self.gate_size))
        # self.b_ih_ut = Parameter(torch.Tensor(self.gate_size))
        self.b_hh_two = Parameter(torch.Tensor(self.gate_size))
        self.b_iah = Parameter(torch.Tensor(self.hidden_size))
        self.b_oah = Parameter(torch.Tensor(self.hidden_size))
        self.b_iah_two = Parameter(torch.Tensor(self.hidden_size))
        self.b_oah_two = Parameter(torch.Tensor(self.hidden_size))
        self.b_adj = Parameter(torch.Tensor(self.hidden_size))
        self.linear_edge_in = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_edge_out = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_edge_in_two = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_edge_out_two = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_edge_f = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_edge_adj = nn.Linear(self.hidden_size, self.hidden_size, bias=True)

    # def GNNCell(self, A, hidden):
    def GNNCell(self, A, hidden,net):
        #############################test13 hy simlarity##########################################

        if opt.method == 'del_bais':
            input_in = torch.matmul(A[:, :, :A.shape[1]], self.linear_edge_in(hidden))
            input_out = torch.matmul(A[:, :, A.shape[1]: 2 * A.shape[1]], self.linear_edge_out(hidden))
            inputs = torch.cat([input_in, input_out], 2)
            gi = F.linear(inputs, self.w_ih, self.b_ih)
            gh = F.linear(hidden, self.w_hh, self.b_hh)
            i_r, i_i, i_n = gi.chunk(3, 2)
            h_r, h_i, h_n = gh.chunk(3, 2)
            resetgate = torch.sigmoid(i_r + h_r)
            inputgate = torch.sigmoid(i_i + h_i)
            newgate = torch.tanh(i_n + resetgate * h_n)
            hy = newgate + inputgate * (hidden - newgate)
        if opt.method == 'change_activation_function':
            input_in = torch.matmul(A[:, :, :A.shape[1]], self.linear_edge_in(hidden)) + self.b_iah
            input_out = torch.matmul(A[:, :, A.shape[1]: 2 * A.shape[1]], self.linear_edge_out(hidden)) + self.b_oah
            inputs = torch.cat([input_in, input_out], 2)
            gi = F.linear(inputs, self.w_ih, self.b_ih)
            gh = F.linear(hidden, self.w_hh, self.b_hh)
            i_r, i_i, i_n = gi.chunk(3, 2)
            h_r, h_i, h_n = gh.chunk(3, 2)
            resetgate = torch.relu(i_r + h_r)
            inputgate = torch.relu(i_i + h_i)
            newgate = torch.relu(i_n + resetgate * h_n)
            hy = newgate + inputgate * (hidden - newgate)
        if opt.method == 'two_gru':
            input_in = torch.matmul(A[:, :, :A.shape[1]], self.linear_edge_in(hidden)) + self.b_iah
            input_out = torch.matmul(A[:, :, A.shape[1]: 2 * A.shape[1]], self.linear_edge_out(hidden)) + self.b_oah
            inputs = torch.cat([input_in, input_out], 2)

            gi = F.linear(inputs, self.w_ih, self.b_ih)
            gh = F.linear(hidden, self.w_hh, self.b_hh)
            i_r, i_i, i_n = gi.chunk(3, 2)
            h_r, h_i, h_n = gh.chunk(3, 2)
            resetgate = torch.sigmoid(i_r + h_r)
            inputgate = torch.sigmoid(i_i + h_i)
            newgate = torch.tanh(i_n + resetgate * h_n)
            hy = newgate + inputgate * (hidden - newgate)

            gi_two = F.linear(inputs, self.w_ih, self.b_ih)
            gh_two = F.linear(hy, self.w_hh, self.b_hh)
            i_r_two, i_i_two, i_n_two = gi_two.chunk(3, 2)
            h_r_two, h_i_two, h_n_two = gh_two.chunk(3, 2)
            resetgate_two = torch.sigmoid(i_r_two + h_r_two)
            inputgate_two = torch.sigmoid(i_i_two + h_i_two)
            newgate_two = torch.tanh(i_n_two + resetgate_two * h_n_two)
            hy = newgate_two + inputgate_two * (hy - newgate_two)

        if opt.method == 'two_gru_modify':
            input_in = torch.matmul(A[:, :, :A.shape[1]], self.linear_edge_in(hidden)) + self.b_iah
            input_out = torch.matmul(A[:, :, A.shape[1]: 2 * A.shape[1]], self.linear_edge_out(hidden)) + self.b_oah
            inputs = torch.cat([input_in, input_out], 2)

            gi = F.linear(inputs, self.w_ih, self.b_ih)
            gh = F.linear(hidden, self.w_hh, self.b_hh)
            i_r, i_i, i_n = gi.chunk(3, 2)
            h_r, h_i, h_n = gh.chunk(3, 2)
            resetgate = torch.sigmoid(i_r + h_r)
            inputgate = torch.sigmoid(i_i + h_i)
            newgate = torch.tanh(i_n + resetgate * h_n)
            hy = newgate + inputgate * (hidden - newgate)

            gi_two = F.linear(inputs, self.w_ih_two, self.b_ih_two)
            gh_two = F.linear(hy, self.w_hh_two, self.b_hh_two)
            i_r_two, i_i_two, i_n_two = gi_two.chunk(3, 2)
            h_r_two, h_i_two, h_n_two = gh_two.chunk(3, 2)
            resetgate_two = torch.sigmoid(i_r_two + h_r_two)
            inputgate_two = torch.sigmoid(i_i_two + h_i_two)
            newgate_two = torch.tanh(i_n_two + resetgate_two * h_n_two)
            hy = newgate_two + inputgate_two * (hy - newgate_two)

        if opt.method == 'two_gru_residual':
            input_in = torch.matmul(A[:, :, :A.shape[1]], self.linear_edge_in(hidden)) + self.b_iah
            input_out = torch.matmul(A[:, :, A.shape[1]: 2 * A.shape[1]], self.linear_edge_out(hidden)) + self.b_oah
            inputs = torch.cat([input_in, input_out], 2)

            gi = F.linear(inputs, self.w_ih, self.b_ih)
            gh = F.linear(hidden, self.w_hh, self.b_hh)
            i_r, i_i, i_n = gi.chunk(3, 2)
            h_r, h_i, h_n = gh.chunk(3, 2)
            resetgate = torch.sigmoid(i_r + h_r)
            inputgate = torch.sigmoid(i_i + h_i)
            newgate = torch.tanh(i_n + resetgate * h_n)
            hy = newgate + inputgate * (hidden - newgate)

            gi_two = F.linear(inputs, self.w_ih, self.b_ih)
            gh_two = F.linear(hy, self.w_hh, self.b_hh)
            i_r_two, i_i_two, i_n_two = gi_two.chunk(3, 2)
            h_r_two, h_i_two, h_n_two = gh_two.chunk(3, 2)
            resetgate_two = torch.sigmoid(i_r_two + h_r_two)
            inputgate_two = torch.sigmoid(i_i_two + h_i_two)
            newgate_two = torch.tanh(i_n_two + resetgate_two * h_n_two)
            hy = newgate_two + inputgate_two * (hy - newgate_two)+hy

        if opt.method == 'two_gru_residual_modify':
            input_in = torch.matmul(A[:, :, :A.shape[1]], self.linear_edge_in(hidden)) + self.b_iah
            input_out = torch.matmul(A[:, :, A.shape[1]: 2 * A.shape[1]], self.linear_edge_out(hidden)) + self.b_oah
            inputs = torch.cat([input_in, input_out], 2)

            gi = F.linear(inputs, self.w_ih, self.b_ih)
            gh = F.linear(hidden, self.w_hh, self.b_hh)
            i_r, i_i, i_n = gi.chunk(3, 2)
            h_r, h_i, h_n = gh.chunk(3, 2)
            resetgate = torch.sigmoid(i_r + h_r)
            inputgate = torch.sigmoid(i_i + h_i)
            newgate = torch.tanh(i_n + resetgate * h_n)
            hy = newgate + inputgate * (hidden - newgate)

            gi_two = F.linear(inputs, self.w_ih_two, self.b_ih_two)
            gh_two = F.linear(hy, self.w_hh_two, self.b_hh_two)
            i_r_two, i_i_two, i_n_two = gi_two.chunk(3, 2)
            h_r_two, h_i_two, h_n_two = gh_two.chunk(3, 2)
            resetgate_two = torch.sigmoid(i_r_two + h_r_two)
            inputgate_two = torch.sigmoid(i_i_two + h_i_two)
            newgate_two = torch.tanh(i_n_two + resetgate_two * h_n_two)
            hy = newgate_two + inputgate_two * (hy - newgate_two)+hy

        if opt.method == 'two_gru_resi_addA':
            input_in = torch.matmul(A[:, :, :A.shape[1]], self.linear_edge_in(hidden)) + self.b_iah
            input_out = torch.matmul(A[:, :, A.shape[1]: 2 * A.shape[1]], self.linear_edge_out(hidden)) + self.b_oah
            inputs = torch.cat([input_in, input_out], 2)

            gi = F.linear(inputs, self.w_ih, self.b_ih)
            gh = F.linear(hidden, self.w_hh, self.b_hh)
            i_r, i_i, i_n = gi.chunk(3, 2)
            h_r, h_i, h_n = gh.chunk(3, 2)
            resetgate = torch.sigmoid(i_r + h_r)
            inputgate = torch.sigmoid(i_i + h_i)
            newgate = torch.tanh(i_n + resetgate * h_n)
            hy = newgate + inputgate * (hidden - newgate)

            gi_two = F.linear(inputs, self.w_ih_two, self.b_ih_two)
            gh_two = F.linear(hy, self.w_hh_two, self.b_hh_two)
            i_r_two, i_i_two, i_n_two = gi_two.chunk(3, 2)
            h_r_two, h_i_two, h_n_two = gh_two.chunk(3, 2)
            resetgate_two = torch.sigmoid(i_r_two + h_r_two)
            inputgate_two = torch.sigmoid(i_i_two + h_i_two)
            newgate_two = torch.tanh(i_n_two + resetgate_two * h_n_two)
            hy = newgate_two + inputgate_two * (hy - newgate_two)+hy

            hy_flatten = hy.reshape(hy.shape[0], hy.shape[1] * hy.shape[2]).cuda()
            hy_dis = cdist(hy_flatten.cpu().detach(), hy_flatten.cpu().detach(), metric='euclidean')
            hy_dis = MinMaxScaler().fit(hy_dis).transform(hy_dis).T
            hy_dis = torch.tensor(hy_dis, dtype=torch.float32)
            hy_dis = hy_dis.pow(2)
            idx_k = np.argsort(hy_dis)[:, :opt.k]
            idx_len = idx_k.shape[0]
            hy_k = hy_dis[np.arange(0, idx_len, 1).reshape(idx_len, 1), idx_k]
            hy_i_k = hy_k[:, -1]
            hy_j_k = hy_k[idx_k, -1]
            mole = -hy_k
            deno = hy_j_k * hy_i_k.reshape(hy_i_k.shape[0], 1)
            hy_ij = torch.exp(mole / deno)
            A_ij = torch.zeros(hy_ij.shape[0], hy_ij.shape[0])
            A_ij[np.arange(0, idx_len, 1).reshape(idx_len, 1), idx_k] = hy_ij
            hy_matmul = torch.matmul(A_ij.cuda(), hy_flatten)
            hy = hy_matmul.reshape(hy.shape[0], hy.shape[1], hy.shape[2])

        if opt.method == 'two_gru_define_resi_addA':
            input_in = torch.matmul(A[:, :, :A.shape[1]], self.linear_edge_in(hidden)) + self.b_iah
            input_out = torch.matmul(A[:, :, A.shape[1]: 2 * A.shape[1]], self.linear_edge_out(hidden)) + self.b_oah
            inputs = torch.cat([input_in, input_out], 2)

            gi = F.linear(inputs, self.w_ih, self.b_ih)
            gh = F.linear(hidden, self.w_hh, self.b_hh)
            i_r, i_i, i_n = gi.chunk(3, 2)
            h_r, h_i, h_n = gh.chunk(3, 2)
            resetgate = torch.sigmoid(i_r + h_r)
            inputgate = torch.sigmoid(i_i + h_i)
            newgate = torch.tanh(i_n + h_n)
            hy = newgate + inputgate * (hidden - newgate)

            gi_two = F.linear(inputs, self.w_ih_two, self.b_ih_two)
            gh_two = F.linear(hy, self.w_hh_two, self.b_hh_two)
            i_r_two, i_i_two, i_n_two = gi_two.chunk(3, 2)
            h_r_two, h_i_two, h_n_two = gh_two.chunk(3, 2)
            resetgate_two = torch.sigmoid(i_r_two + h_r_two)
            inputgate_two = torch.sigmoid(i_i_two + h_i_two)
            newgate_two = torch.tanh(i_n_two + h_n_two)
            hy = newgate_two + inputgate_two * (hy - newgate_two)+hy

            hy_flatten = hy.reshape(hy.shape[0], hy.shape[1] * hy.shape[2]).cuda()
            hy_dis = cdist(hy_flatten.cpu().detach(), hy_flatten.cpu().detach(), metric='euclidean')
            hy_dis = MinMaxScaler().fit(hy_dis).transform(hy_dis).T
            hy_dis = torch.tensor(hy_dis, dtype=torch.float32)
            hy_dis = hy_dis.pow(2)
            idx_k = np.argsort(hy_dis)[:, :opt.k]
            idx_len = idx_k.shape[0]
            hy_k = hy_dis[np.arange(0, idx_len, 1).reshape(idx_len, 1), idx_k]
            hy_i_k = hy_k[:, -1]
            hy_j_k = hy_k[idx_k, -1]
            mole = -hy_k
            deno = hy_j_k * hy_i_k.reshape(hy_i_k.shape[0], 1)
            hy_ij = torch.exp(mole / deno)
            A_ij = torch.zeros(hy_ij.shape[0], hy_ij.shape[0])
            A_ij[np.arange(0, idx_len, 1).reshape(idx_len, 1), idx_k] = hy_ij
            hy_matmul = torch.matmul(A_ij.cuda(), hy_flatten)
            hy = hy_matmul.reshape(hy.shape[0], hy.shape[1], hy.shape[2])
        if opt.method == 'two_gru_define_resi_addA_mixed':
            input_in = torch.matmul(A[:, :, :A.shape[1]], self.linear_edge_in(hidden)) + self.b_iah
            input_out = torch.matmul(A[:, :, A.shape[1]: 2 * A.shape[1]], self.linear_edge_out(hidden)) + self.b_oah
            inputs = torch.cat([input_in, input_out], 2)

            gi = F.linear(inputs, self.w_ih, self.b_ih)
            gh = F.linear(hidden, self.w_hh, self.b_hh)
            i_r, i_i, i_n = gi.chunk(3, 2)
            h_r, h_i, h_n = gh.chunk(3, 2)
            resetgate = torch.sigmoid(i_r + h_r)
            inputgate = torch.sigmoid(i_i + h_i)
            newgate = torch.tanh(i_n + h_n)
            hy = newgate + inputgate * (hidden - newgate)

            gi_two = F.linear(inputs, self.w_ih_two, self.b_ih_two)
            gh_two = F.linear(hy, self.w_hh_two, self.b_hh_two)
            i_r_two, i_i_two, i_n_two = gi_two.chunk(3, 2)
            h_r_two, h_i_two, h_n_two = gh_two.chunk(3, 2)
            resetgate_two = torch.sigmoid(i_r_two + h_r_two)
            inputgate_two = torch.sigmoid(i_i_two + h_i_two)
            newgate_two = torch.tanh(i_n_two + h_n_two)
            hy = newgate_two + inputgate_two * (hy - newgate_two)+hy

            hy_flatten = hy.reshape(hy.shape[0], hy.shape[1] * hy.shape[2]).cuda()
            hy_dis = cdist(hy_flatten.cpu().detach(), hy_flatten.cpu().detach(), metric='euclidean')
            hy_dis = MinMaxScaler().fit(hy_dis).transform(hy_dis).T
            hy_dis = torch.tensor(hy_dis, dtype=torch.float32)
            hy_dis = hy_dis.pow(2)
            #obtain positive k
            idx_k_pos = np.argsort(hy_dis)[:, :opt.k]
            #obtatin negative k
            idx_k_neg = np.argsort(hy_dis)[:, -opt.k:]

            #obatin positive A matrix
            idx_len_pos = idx_k_pos.shape[0]
            hy_k_pos = hy_dis[np.arange(0, idx_len_pos, 1).reshape(idx_len_pos, 1), idx_k_pos]
            hy_i_k_pos = hy_k_pos[:, -1]
            hy_j_k_pos = hy_k_pos[idx_k_pos, -1]
            mole_pos = -hy_k_pos
            deno_pos = hy_j_k_pos * hy_i_k_pos.reshape(hy_i_k_pos.shape[0], 1)
            hy_ij_pos = torch.exp(mole_pos / deno_pos)
            A_ij_pos = torch.zeros(hy_ij_pos.shape[0], hy_ij_pos.shape[0])
            A_ij_pos[np.arange(0, idx_len_pos, 1).reshape(idx_len_pos, 1), idx_k_pos] = hy_ij_pos

            #obatin negative Amatrix
            idx_len_neg = idx_k_neg.shape[0]
            hy_k_neg = hy_dis[np.arange(0, idx_len_neg, 1).reshape(idx_len_neg, 1), idx_k_neg]
            hy_i_k_neg = hy_k_neg[:, -1]
            hy_j_k_neg = hy_k_neg[idx_k_neg, -1]
            mole_neg = -hy_k_neg
            deno_neg = hy_j_k_neg * hy_i_k_neg.reshape(hy_i_k_neg.shape[0], 1)
            hy_ij_neg = torch.exp(mole_neg / deno_neg)
            A_ij_neg = torch.zeros(hy_ij_neg.shape[0], hy_ij_neg.shape[0])
            A_ij_neg[np.arange(0, idx_len_neg, 1).reshape(idx_len_neg, 1), idx_k_neg] = hy_ij_neg

            hy_matmul_pos = torch.matmul(A_ij_pos.cuda(), hy_flatten)
            hy_matmul_neg = torch.matmul(A_ij_neg.cuda(), hy_flatten)
            hy_matmul = hy_matmul_pos*hy_matmul_neg
            hy = hy_matmul.reshape(hy.shape[0], hy.shape[1], hy.shape[2])
        if opt.method == 'two_gru_define_resi_addA_mixed2':
            input_in = torch.matmul(A[:, :, :A.shape[1]], self.linear_edge_in(hidden)) + self.b_iah
            input_out = torch.matmul(A[:, :, A.shape[1]: 2 * A.shape[1]], self.linear_edge_out(hidden)) + self.b_oah
            inputs = torch.cat([input_in, input_out], 2)

            gi = F.linear(inputs, self.w_ih, self.b_ih)
            gh = F.linear(hidden, self.w_hh, self.b_hh)
            i_r, i_i, i_n = gi.chunk(3, 2)
            h_r, h_i, h_n = gh.chunk(3, 2)
            resetgate = torch.sigmoid(i_r + h_r)
            inputgate = torch.sigmoid(i_i + h_i)
            newgate = torch.tanh(i_n + h_n)
            hy = newgate + inputgate * (hidden - newgate)

            gi_two = F.linear(inputs, self.w_ih_two, self.b_ih_two)
            gh_two = F.linear(hy, self.w_hh_two, self.b_hh_two)
            i_r_two, i_i_two, i_n_two = gi_two.chunk(3, 2)
            h_r_two, h_i_two, h_n_two = gh_two.chunk(3, 2)
            resetgate_two = torch.sigmoid(i_r_two + h_r_two)
            inputgate_two = torch.sigmoid(i_i_two + h_i_two)
            newgate_two = torch.tanh(i_n_two + h_n_two)
            hy = newgate_two + inputgate_two * (hy - newgate_two)+hy

            hy_flatten = hy.reshape(hy.shape[0], hy.shape[1] * hy.shape[2]).cuda()
            hy_dis = cdist(hy_flatten.cpu().detach(), hy_flatten.cpu().detach(), metric='euclidean')
            hy_dis = MinMaxScaler().fit(hy_dis).transform(hy_dis).T
            hy_dis = torch.tensor(hy_dis, dtype=torch.float32)
            hy_dis = hy_dis.pow(2)
            #obtain positive k
            idx_k_pos = np.argsort(hy_dis)[:, :opt.k]
            #obtain positive k and  k of k
            idx_kk_pos = np.argsort(hy_dis)[:, :opt.k]
            idx_kk_pos_kk = idx_kk_pos[idx_kk_pos[:]][:, :, 1]
            idx_kk_pos = np.concatenate((tuple(idx_kk_pos.tolist()), tuple(idx_kk_pos_kk.tolist())), axis=1)
            idx_kk_pos_sort_index =  np.argsort(hy_dis[np.arange(0, hy_dis.shape[0],1).reshape(hy_dis.shape[0], 1), idx_kk_pos])
            idx_kk_pos = idx_kk_pos[np.arange(0,idx_kk_pos.shape[0],1).reshape(idx_kk_pos.shape[0],1),idx_kk_pos_sort_index]
            idx_k_pos = idx_kk_pos

            #obatin positive A matrix
            idx_len_pos = idx_k_pos.shape[0]
            hy_k_pos = hy_dis[np.arange(0, idx_len_pos, 1).reshape(idx_len_pos, 1), idx_k_pos]
            hy_i_k_pos = hy_k_pos[:, -1]
            hy_j_k_pos = hy_k_pos[idx_k_pos, -1]
            mole_pos = -hy_k_pos
            deno_pos = hy_j_k_pos * hy_i_k_pos.reshape(hy_i_k_pos.shape[0], 1)
            hy_ij_pos = torch.exp(mole_pos / deno_pos)
            A_ij_pos = torch.zeros(hy_ij_pos.shape[0], hy_ij_pos.shape[0])
            A_ij_pos[np.arange(0, idx_len_pos, 1).reshape(idx_len_pos, 1), idx_k_pos] = hy_ij_pos

            hy_matmul_pos = torch.matmul(A_ij_pos.cuda(), hy_flatten)
            hy = hy_matmul_pos.reshape(hy.shape[0], hy.shape[1], hy.shape[2])

        if opt.method == 'two_gru_define_resi_addA_mixed2comparek':
            input_in = torch.matmul(A[:, :, :A.shape[1]], self.linear_edge_in(hidden)) + self.b_iah
            input_out = torch.matmul(A[:, :, A.shape[1]: 2 * A.shape[1]], self.linear_edge_out(hidden)) + self.b_oah
            inputs = torch.cat([input_in, input_out], 2)

            gi = F.linear(inputs, self.w_ih, self.b_ih)
            gh = F.linear(hidden, self.w_hh, self.b_hh)
            i_r, i_i, i_n = gi.chunk(3, 2)
            h_r, h_i, h_n = gh.chunk(3, 2)
            resetgate = torch.sigmoid(i_r + h_r)
            inputgate = torch.sigmoid(i_i + h_i)
            newgate = torch.tanh(i_n + h_n)
            hy = newgate + inputgate * (hidden - newgate)

            gi_two = F.linear(inputs, self.w_ih_two, self.b_ih_two)
            gh_two = F.linear(hy, self.w_hh_two, self.b_hh_two)
            i_r_two, i_i_two, i_n_two = gi_two.chunk(3, 2)
            h_r_two, h_i_two, h_n_two = gh_two.chunk(3, 2)
            resetgate_two = torch.sigmoid(i_r_two + h_r_two)
            inputgate_two = torch.sigmoid(i_i_two + h_i_two)
            newgate_two = torch.tanh(i_n_two + h_n_two)
            hy = newgate_two + inputgate_two * (hy - newgate_two)+hy

            hy_flatten = hy.reshape(hy.shape[0], hy.shape[1] * hy.shape[2]).cuda()
            hy_dis = cdist(hy_flatten.cpu().detach(), hy_flatten.cpu().detach(), metric='euclidean')
            hy_dis = MinMaxScaler().fit(hy_dis).transform(hy_dis).T
            hy_dis = torch.tensor(hy_dis, dtype=torch.float32)
            hy_dis = hy_dis.pow(2)
            #obtain positive k
            idx_k_pos = np.argsort(hy_dis)[:, :opt.k]

            #obtain positive k and  k of k
            idx_kk_pos = np.argsort(hy_dis)[:, :opt.k]
            idx_kk_pos_kk = idx_kk_pos[idx_kk_pos[:]][:, :, 1]
            idx_kk_pos = np.concatenate((tuple(idx_kk_pos.tolist()), tuple(idx_kk_pos_kk.tolist())), axis=1)
            idx_kk_pos_sort_index =  np.argsort(hy_dis[np.arange(0, hy_dis.shape[0],1).reshape(hy_dis.shape[0], 1), idx_kk_pos])
            idx_kk_pos = idx_kk_pos[np.arange(0,idx_kk_pos.shape[0],1).reshape(idx_kk_pos.shape[0],1),idx_kk_pos_sort_index]
            # idx_k_pos = idx_kk_pos

            #obatin positive A matrix
            idx_len_pos = idx_k_pos.shape[0]
            hy_k_pos = hy_dis[np.arange(0, idx_len_pos, 1).reshape(idx_len_pos, 1), idx_k_pos]
            hy_i_k_pos = hy_k_pos[:, -1]
            hy_j_k_pos = hy_k_pos[idx_k_pos, -1]
            mole_pos = -hy_k_pos
            deno_pos = hy_j_k_pos * hy_i_k_pos.reshape(hy_i_k_pos.shape[0], 1)
            hy_ij_pos = torch.exp(mole_pos / deno_pos)
            A_ij_pos = torch.zeros(hy_ij_pos.shape[0], hy_ij_pos.shape[0])
            A_ij_pos[np.arange(0, idx_len_pos, 1).reshape(idx_len_pos, 1), idx_k_pos] = hy_ij_pos

            hy_matmul_pos = torch.matmul(A_ij_pos.cuda(), hy_flatten)
            hy = hy_matmul_pos.reshape(hy.shape[0], hy.shape[1], hy.shape[2])

        if opt.method == 'two_gru_define_resi_addA_mixed_three':
            input_in = torch.matmul(A[:, :, :A.shape[1]], self.linear_edge_in(hidden)) + self.b_iah
            input_out = torch.matmul(A[:, :, A.shape[1]: 2 * A.shape[1]], self.linear_edge_out(hidden)) + self.b_oah
            inputs = torch.cat([input_in, input_out], 2)

            gi = F.linear(inputs, self.w_ih, self.b_ih)
            gh = F.linear(hidden, self.w_hh, self.b_hh)
            i_r, i_i, i_n = gi.chunk(3, 2)
            h_r, h_i, h_n = gh.chunk(3, 2)
            resetgate = torch.sigmoid(i_r + h_r)
            inputgate = torch.sigmoid(i_i + h_i)
            newgate = torch.tanh(i_n + h_n)
            hy = newgate + inputgate * (hidden - newgate)

            gi_two = F.linear(inputs, self.w_ih_two, self.b_ih_two)
            gh_two = F.linear(hy, self.w_hh_two, self.b_hh_two)
            i_r_two, i_i_two, i_n_two = gi_two.chunk(3, 2)
            h_r_two, h_i_two, h_n_two = gh_two.chunk(3, 2)
            resetgate_two = torch.sigmoid(i_r_two + h_r_two)
            inputgate_two = torch.sigmoid(i_i_two + h_i_two)
            newgate_two = torch.tanh(i_n_two + h_n_two)
            hy = newgate_two + inputgate_two * (hy - newgate_two)+hy

            hy_flatten = hy.reshape(hy.shape[0], hy.shape[1] * hy.shape[2]).cuda()
            hy_dis = cdist(hy_flatten.cpu().detach(), hy_flatten.cpu().detach(), metric='euclidean')
            hy_dis = MinMaxScaler().fit(hy_dis).transform(hy_dis).T
            hy_dis = torch.tensor(hy_dis, dtype=torch.float32)
            hy_dis = hy_dis.pow(2)
            #obtain positive k
            idx_k_pos = np.argsort(hy_dis)[:, :opt.k]
            #obtatin negative k
            idx_k_neg = np.argsort(hy_dis)[:, -opt.k:]

            #obatin positive A matrix
            idx_len_pos = idx_k_pos.shape[0]
            hy_k_pos = hy_dis[np.arange(0, idx_len_pos, 1).reshape(idx_len_pos, 1), idx_k_pos]
            hy_i_k_pos = hy_k_pos[:, -1]
            hy_j_k_pos = hy_k_pos[idx_k_pos, -1]
            mole_pos = -hy_k_pos
            deno_pos = hy_j_k_pos * hy_i_k_pos.reshape(hy_i_k_pos.shape[0], 1)
            hy_ij_pos = torch.exp(mole_pos / deno_pos)
            A_ij_pos = torch.zeros(hy_ij_pos.shape[0], hy_ij_pos.shape[0])
            A_ij_pos[np.arange(0, idx_len_pos, 1).reshape(idx_len_pos, 1), idx_k_pos] = hy_ij_pos

            #obatin negative Amatrix
            idx_len_neg = idx_k_neg.shape[0]
            hy_k_neg = hy_dis[np.arange(0, idx_len_neg, 1).reshape(idx_len_neg, 1), idx_k_neg]
            hy_i_k_neg = hy_k_neg[:, -1]
            hy_j_k_neg = hy_k_neg[idx_k_neg, -1]
            mole_neg = -hy_k_neg
            deno_neg = hy_j_k_neg * hy_i_k_neg.reshape(hy_i_k_neg.shape[0], 1)
            hy_ij_neg = torch.exp(mole_neg / deno_neg)
            A_ij_neg = torch.zeros(hy_ij_neg.shape[0], hy_ij_neg.shape[0])
            A_ij_neg[np.arange(0, idx_len_neg, 1).reshape(idx_len_neg, 1), idx_k_neg] = hy_ij_neg

            hy_matmul_pos = torch.matmul(A_ij_pos.cuda(), hy_flatten)
            hy_matmul_neg = torch.matmul(A_ij_neg.cuda(), hy_flatten)
            hy_matmul = hy_matmul_pos*hy_matmul_neg

            hy_pos = hy_matmul_pos.reshape(hy.shape[0], hy.shape[1], hy.shape[2])
            hy_neg = hy_matmul_neg.reshape(hy.shape[0], hy.shape[1], hy.shape[2])

            hy = hy_pos

        if opt.method == 'mixed_posneg_br_one':
            input_in = torch.matmul(A[:, :, :A.shape[1]], self.linear_edge_in(hidden)) + self.b_iah
            input_out = torch.matmul(A[:, :, A.shape[1]: 2 * A.shape[1]], self.linear_edge_out(hidden)) + self.b_oah
            inputs = torch.cat([input_in, input_out], 2)

            gi = F.linear(inputs, self.w_ih, self.b_ih)
            gh = F.linear(hidden, self.w_hh, self.b_hh)
            i_r, i_i, i_n = gi.chunk(3, 2)
            h_r, h_i, h_n = gh.chunk(3, 2)
            resetgate = torch.sigmoid(i_r + h_r)
            inputgate = torch.sigmoid(i_i + h_i)
            newgate = torch.tanh(i_n + resetgate * h_n)
            hy = newgate + inputgate * (hidden - newgate)

            gi_two = F.linear(inputs, self.w_ih_two, self.b_ih_two)
            gh_two = F.linear(hy, self.w_hh_two, self.b_hh_two)
            i_r_two, i_i_two, i_n_two = gi_two.chunk(3, 2)
            h_r_two, h_i_two, h_n_two = gh_two.chunk(3, 2)
            resetgate_two = torch.sigmoid(i_r_two + h_r_two)
            inputgate_two = torch.sigmoid(i_i_two + h_i_two)
            newgate_two = torch.tanh(i_n_two + resetgate_two * h_n_two)
            hy = newgate_two + inputgate_two * (hy - newgate_two)+hy

            hy_flatten = hy.reshape(hy.shape[0], hy.shape[1] * hy.shape[2]).cuda()
            hy_dis = cdist(hy_flatten.cpu().detach(), hy_flatten.cpu().detach(), metric='euclidean')
            hy_dis = MinMaxScaler().fit(hy_dis).transform(hy_dis).T
            hy_dis = torch.tensor(hy_dis, dtype=torch.float32)
            hy_dis = hy_dis.pow(2)
            #obtain positive k
            idx_k_pos = np.argsort(hy_dis)[:, :opt.k]
            #obtatin negative k
            idx_k_neg = np.argsort(hy_dis)[:, -opt.k:]

            #obatin positive A matrix
            idx_len_pos = idx_k_pos.shape[0]
            hy_k_pos = hy_dis[np.arange(0, idx_len_pos, 1).reshape(idx_len_pos, 1), idx_k_pos]
            hy_i_k_pos = hy_k_pos[:, -1]
            hy_j_k_pos = hy_k_pos[idx_k_pos, -1]
            mole_pos = -hy_k_pos
            deno_pos = hy_j_k_pos * hy_i_k_pos.reshape(hy_i_k_pos.shape[0], 1)
            hy_ij_pos = torch.exp(mole_pos / deno_pos)
            A_ij_pos = torch.zeros(hy_ij_pos.shape[0], hy_ij_pos.shape[0])
            A_ij_pos[np.arange(0, idx_len_pos, 1).reshape(idx_len_pos, 1), idx_k_pos] = hy_ij_pos

            #obatin negative Amatrix
            # idx_len_neg = idx_k_neg.shape[0]
            # hy_k_neg = hy_dis[np.arange(0, idx_len_neg, 1).reshape(idx_len_neg, 1), idx_k_neg]
            # hy_i_k_neg = hy_k_neg[:, -1]
            # hy_j_k_neg = hy_k_neg[idx_k_neg, -1]
            # mole_neg = -hy_k_neg
            # deno_neg = hy_j_k_neg * hy_i_k_neg.reshape(hy_i_k_neg.shape[0], 1)
            # hy_ij_neg = torch.exp(mole_neg / deno_neg)
            # A_ij_neg = torch.zeros(hy_ij_neg.shape[0], hy_ij_neg.shape[0])
            # A_ij_neg[np.arange(0, idx_len_neg, 1).reshape(idx_len_neg, 1), idx_k_neg] = hy_ij_neg

            hy_matmul_pos = torch.matmul(A_ij_pos.cuda(), hy_flatten)
            # hy_matmul_neg = torch.matmul(A_ij_neg.cuda(), hy_flatten)
            # hy_matmul = hy_matmul_pos*hy_matmul_neg

            hy_pos = hy_matmul_pos.reshape(hy.shape[0], hy.shape[1], hy.shape[2])
            # hy_neg = hy_matmul_neg.reshape(hy.shape[0], hy.shape[1], hy.shape[2])
            hy_neg = np.array([1])
            hy = hy_pos
        if opt.method == 'new1_mixed_posneg_br_one':
            input_in = torch.matmul(A[:, :, :A.shape[1]], self.linear_edge_in(hidden)) + self.b_iah
            input_out = torch.matmul(A[:, :, A.shape[1]: 2 * A.shape[1]], self.linear_edge_out(hidden)) + self.b_oah
            inputs = torch.cat([input_in, input_out], 2)
            w_za = F.linear(inputs, self.w_z, self.wb_z)
            w_oa = F.linear(inputs, self.w_o, self.wb_o)
            u_zv = F.linear(hidden, self.u_z, self.ub_z)
            u_ov = F.linear(hidden, self.u_o, self.ub_o)
            prelu = torch.nn.PReLU().cuda()
            zs = prelu(w_za + u_zv)
            vtb = prelu(w_oa +u_ov)
            hy = (1-zs)*hidden + zs*vtb

            hy_flatten = hy.reshape(hy.shape[0], hy.shape[1] * hy.shape[2]).cuda()
            hy_dis = cdist(hy_flatten.cpu().detach(), hy_flatten.cpu().detach(), metric='euclidean')
            hy_dis = MinMaxScaler().fit(hy_dis).transform(hy_dis).T
            hy_dis = torch.tensor(hy_dis, dtype=torch.float32)
            hy_dis = hy_dis.pow(2)
            # obtain positive k
            idx_k_pos = np.argsort(hy_dis)[:, :opt.k]
            # obtatin negative k
            idx_k_neg = np.argsort(hy_dis)[:, -opt.k:]

            # obatin positive A matrix
            idx_len_pos = idx_k_pos.shape[0]
            hy_k_pos = hy_dis[np.arange(0, idx_len_pos, 1).reshape(idx_len_pos, 1), idx_k_pos]
            hy_i_k_pos = hy_k_pos[:, -1]
            hy_j_k_pos = hy_k_pos[idx_k_pos, -1]
            mole_pos = -hy_k_pos
            deno_pos = hy_j_k_pos * hy_i_k_pos.reshape(hy_i_k_pos.shape[0], 1)
            hy_ij_pos = torch.exp(mole_pos / deno_pos)
            A_ij_pos = torch.zeros(hy_ij_pos.shape[0], hy_ij_pos.shape[0])
            A_ij_pos[np.arange(0, idx_len_pos, 1).reshape(idx_len_pos, 1), idx_k_pos] = hy_ij_pos

            hy_matmul_pos = torch.matmul(A_ij_pos.cuda(), hy_flatten)
            hy_pos = hy_matmul_pos.reshape(hy.shape[0], hy.shape[1], hy.shape[2])

            hy_neg = np.array([1])
            hy = hy_pos
        if opt.method == 'GIISR':
            input_in = torch.matmul(A[:, :, :A.shape[1]], self.linear_edge_in(hidden)) + self.b_iah
            input_out = torch.matmul(A[:, :, A.shape[1]: 2 * A.shape[1]], self.linear_edge_out(hidden)) + self.b_oah
            inputs = torch.cat([input_in, input_out], 2)
            w_za = F.linear(inputs, self.w_z, self.wb_z)
            w_oa = F.linear(inputs, self.w_o, self.wb_o)
            u_zv = F.linear(hidden, self.u_z, self.ub_z)
            u_ov = F.linear(hidden, self.u_o, self.ub_o)
            prelu = torch.nn.PReLU().cuda()
            zs = prelu(w_za + u_zv)
            vtb = prelu(w_oa +u_ov)
            hy = (1-zs)*hidden + zs*vtb

            w_za2 = F.linear(inputs, self.w_z2, self.wb_z2)
            w_oa2 = F.linear(inputs, self.w_o2, self.wb_o2)
            u_zv2 = F.linear(hy, self.u_z2, self.ub_z2)
            u_ov2 = F.linear(hy, self.u_o2, self.ub_o2)
            prelu = torch.nn.PReLU().cuda()
            zs2 = prelu(w_za2 + u_zv2)
            vtb2 = prelu(w_oa2 +u_ov2)
            hy = (1 - zs2) * hy + zs2 * vtb2 + hy

            hy_flatten = hy.reshape(hy.shape[0], hy.shape[1] * hy.shape[2]).cuda()
            hy_dis = cdist(hy_flatten.cpu().detach(), hy_flatten.cpu().detach(), metric='euclidean')
            hy_dis = MinMaxScaler().fit(hy_dis).transform(hy_dis).T
            hy_dis = torch.tensor(hy_dis, dtype=torch.float32)
            hy_dis = hy_dis.pow(2)
            # obtain positive k
            idx_k_pos = np.argsort(hy_dis)[:, :opt.k]
            # obtatin negative k
            idx_k_neg = np.argsort(hy_dis)[:, -opt.k:]

            # obatin positive A matrix
            idx_len_pos = idx_k_pos.shape[0]
            hy_k_pos = hy_dis[np.arange(0, idx_len_pos, 1).reshape(idx_len_pos, 1), idx_k_pos]
            hy_i_k_pos = hy_k_pos[:, -1]
            hy_j_k_pos = hy_k_pos[idx_k_pos, -1]
            mole_pos = -hy_k_pos
            deno_pos = hy_j_k_pos * hy_i_k_pos.reshape(hy_i_k_pos.shape[0], 1)
            hy_ij_pos = torch.exp(mole_pos / deno_pos)
            A_ij_pos = torch.zeros(hy_ij_pos.shape[0], hy_ij_pos.shape[0])
            A_ij_pos[np.arange(0, idx_len_pos, 1).reshape(idx_len_pos, 1), idx_k_pos] = hy_ij_pos

            hy_matmul_pos = torch.matmul(A_ij_pos.cuda(), hy_flatten)
            hy_pos = hy_matmul_pos.reshape(hy.shape[0], hy.shape[1], hy.shape[2])

            hy_neg = np.array([1])
            hy = hy_pos
        if opt.method == 'mixed_posneg_br_one_no_square':
            input_in = torch.matmul(A[:, :, :A.shape[1]], self.linear_edge_in(hidden)) + self.b_iah
            input_out = torch.matmul(A[:, :, A.shape[1]: 2 * A.shape[1]], self.linear_edge_out(hidden)) + self.b_oah
            inputs = torch.cat([input_in, input_out], 2)

            gi = F.linear(inputs, self.w_ih, self.b_ih)
            gh = F.linear(hidden, self.w_hh, self.b_hh)
            i_r, i_i, i_n = gi.chunk(3, 2)
            h_r, h_i, h_n = gh.chunk(3, 2)
            resetgate = torch.sigmoid(i_r + h_r)
            inputgate = torch.sigmoid(i_i + h_i)
            newgate = torch.tanh(i_n + resetgate * h_n)
            hy = newgate + inputgate * (hidden - newgate)

            gi_two = F.linear(inputs, self.w_ih_two, self.b_ih_two)
            gh_two = F.linear(hy, self.w_hh_two, self.b_hh_two)
            i_r_two, i_i_two, i_n_two = gi_two.chunk(3, 2)
            h_r_two, h_i_two, h_n_two = gh_two.chunk(3, 2)
            resetgate_two = torch.sigmoid(i_r_two + h_r_two)
            inputgate_two = torch.sigmoid(i_i_two + h_i_two)
            newgate_two = torch.tanh(i_n_two + resetgate_two * h_n_two)
            hy = newgate_two + inputgate_two * (hy - newgate_two)+hy

            hy_flatten = hy.reshape(hy.shape[0], hy.shape[1] * hy.shape[2]).cuda()
            hy_dis = cdist(hy_flatten.cpu().detach(), hy_flatten.cpu().detach(), metric='euclidean')
            hy_dis = MinMaxScaler().fit(hy_dis).transform(hy_dis).T
            hy_dis = torch.tensor(hy_dis, dtype=torch.float32)
            # hy_dis = hy_dis.pow(2)
            #obtain positive k
            idx_k_pos = np.argsort(hy_dis)[:, :opt.k]
            #obtatin negative k
            idx_k_neg = np.argsort(hy_dis)[:, -opt.k:]

            #obatin positive A matrix
            idx_len_pos = idx_k_pos.shape[0]
            hy_k_pos = hy_dis[np.arange(0, idx_len_pos, 1).reshape(idx_len_pos, 1), idx_k_pos]
            hy_i_k_pos = hy_k_pos[:, -1]
            hy_j_k_pos = hy_k_pos[idx_k_pos, -1]
            mole_pos = -hy_k_pos
            deno_pos = hy_j_k_pos * hy_i_k_pos.reshape(hy_i_k_pos.shape[0], 1)
            hy_ij_pos = torch.exp(mole_pos / deno_pos)
            A_ij_pos = torch.zeros(hy_ij_pos.shape[0], hy_ij_pos.shape[0])
            A_ij_pos[np.arange(0, idx_len_pos, 1).reshape(idx_len_pos, 1), idx_k_pos] = hy_ij_pos

            #obatin negative Amatrix
            # idx_len_neg = idx_k_neg.shape[0]
            # hy_k_neg = hy_dis[np.arange(0, idx_len_neg, 1).reshape(idx_len_neg, 1), idx_k_neg]
            # hy_i_k_neg = hy_k_neg[:, -1]
            # hy_j_k_neg = hy_k_neg[idx_k_neg, -1]
            # mole_neg = -hy_k_neg
            # deno_neg = hy_j_k_neg * hy_i_k_neg.reshape(hy_i_k_neg.shape[0], 1)
            # hy_ij_neg = torch.exp(mole_neg / deno_neg)
            # A_ij_neg = torch.zeros(hy_ij_neg.shape[0], hy_ij_neg.shape[0])
            # A_ij_neg[np.arange(0, idx_len_neg, 1).reshape(idx_len_neg, 1), idx_k_neg] = hy_ij_neg

            hy_matmul_pos = torch.matmul(A_ij_pos.cuda(), hy_flatten)
            # hy_matmul_neg = torch.matmul(A_ij_neg.cuda(), hy_flatten)
            # hy_matmul = hy_matmul_pos*hy_matmul_neg

            hy_pos = hy_matmul_pos.reshape(hy.shape[0], hy.shape[1], hy.shape[2])
            # hy_neg = hy_matmul_neg.reshape(hy.shape[0], hy.shape[1], hy.shape[2])
            hy_neg = np.array([1])
            hy = hy_pos
        if opt.method == 'mixed_posneg_br_two_no_square':
            input_in = torch.matmul(A[:, :, :A.shape[1]], self.linear_edge_in(hidden)) + self.b_iah
            input_out = torch.matmul(A[:, :, A.shape[1]: 2 * A.shape[1]], self.linear_edge_out(hidden)) + self.b_oah
            inputs = torch.cat([input_in, input_out], 2)

            gi = F.linear(inputs, self.w_ih, self.b_ih)
            gh = F.linear(hidden, self.w_hh, self.b_hh)
            i_r, i_i, i_n = gi.chunk(3, 2)
            h_r, h_i, h_n = gh.chunk(3, 2)
            resetgate = torch.sigmoid(i_r + h_r)
            inputgate = torch.sigmoid(i_i + h_i)
            newgate = torch.tanh(i_n + resetgate * h_n)
            hy = newgate + inputgate * (hidden - newgate)

            gi_two = F.linear(inputs, self.w_ih_two, self.b_ih_two)
            gh_two = F.linear(hy, self.w_hh_two, self.b_hh_two)
            i_r_two, i_i_two, i_n_two = gi_two.chunk(3, 2)
            h_r_two, h_i_two, h_n_two = gh_two.chunk(3, 2)
            resetgate_two = torch.sigmoid(i_r_two + h_r_two)
            inputgate_two = torch.sigmoid(i_i_two + h_i_two)
            newgate_two = torch.tanh(i_n_two + resetgate_two * h_n_two)
            hy = newgate_two + inputgate_two * (hy - newgate_two)+hy

            hy_flatten = hy.reshape(hy.shape[0], hy.shape[1] * hy.shape[2]).cuda()
            hy_dis = cdist(hy_flatten.cpu().detach(), hy_flatten.cpu().detach(), metric='euclidean')
            hy_dis = MinMaxScaler().fit(hy_dis).transform(hy_dis).T
            hy_dis = torch.tensor(hy_dis, dtype=torch.float32)
            # hy_dis = hy_dis.pow(2)
            #obtain positive k
            idx_k_pos = np.argsort(hy_dis)[:, :opt.k]
            #obtain posisive k->negative k
            temp = np.argsort(hy_dis)
            temp = np.array(temp.cpu())
            idx_knk_pos = np.argsort(hy_dis)[:, :opt.k]
            idx_knk_pos_kk =  np.argsort(hy_dis)[idx_knk_pos[:]][:, :, -1]
            #obtatin negative k
            idx_k_neg = np.argsort(hy_dis)[:, -opt.k:]

            idx_knk_pos = np.concatenate((tuple(idx_knk_pos.tolist()), tuple(idx_knk_pos_kk.tolist())), axis=1)
            idx_knk_pos_sort_index = np.argsort(
                hy_dis[np.arange(0, hy_dis.shape[0], 1).reshape(hy_dis.shape[0], 1), idx_knk_pos])
            idx_knk_pos = idx_knk_pos[
                np.arange(0, idx_knk_pos.shape[0], 1).reshape(idx_knk_pos.shape[0], 1), idx_knk_pos_sort_index]

            idx_k_pos = idx_knk_pos

            #obtatin negative k
            idx_k_neg = np.argsort(hy_dis)[:, -opt.k:]

            #obatin positive A matrix
            idx_len_pos = idx_k_pos.shape[0]
            hy_k_pos = hy_dis[np.arange(0, idx_len_pos, 1).reshape(idx_len_pos, 1), idx_k_pos]
            hy_i_k_pos = hy_k_pos[:, -1]
            hy_j_k_pos = hy_k_pos[idx_k_pos, -1]
            mole_pos = -hy_k_pos
            deno_pos = hy_j_k_pos * hy_i_k_pos.reshape(hy_i_k_pos.shape[0], 1)
            hy_ij_pos = torch.exp(mole_pos / deno_pos)
            A_ij_pos = torch.zeros(hy_ij_pos.shape[0], hy_ij_pos.shape[0])
            A_ij_pos[np.arange(0, idx_len_pos, 1).reshape(idx_len_pos, 1), idx_k_pos] = hy_ij_pos

            #obatin negative Amatrix
            idx_len_neg = idx_k_neg.shape[0]
            hy_k_neg = hy_dis[np.arange(0, idx_len_neg, 1).reshape(idx_len_neg, 1), idx_k_neg]
            hy_i_k_neg = hy_k_neg[:, -1]
            hy_j_k_neg = hy_k_neg[idx_k_neg, -1]
            mole_neg = -hy_k_neg
            deno_neg = hy_j_k_neg * hy_i_k_neg.reshape(hy_i_k_neg.shape[0], 1)
            hy_ij_neg = torch.exp(mole_neg / deno_neg)
            A_ij_neg = torch.zeros(hy_ij_neg.shape[0], hy_ij_neg.shape[0])
            A_ij_neg[np.arange(0, idx_len_neg, 1).reshape(idx_len_neg, 1), idx_k_neg] = hy_ij_neg

            hy_matmul_pos = torch.matmul(A_ij_pos.cuda(), hy_flatten)
            hy_matmul_neg = torch.matmul(A_ij_neg.cuda(), hy_flatten)
            # hy_matmul = hy_matmul_pos*hy_matmul_neg

            hy_pos = hy_matmul_pos.reshape(hy.shape[0], hy.shape[1], hy.shape[2])
            hy_neg = hy_matmul_neg.reshape(hy.shape[0], hy.shape[1], hy.shape[2])
            # hy_neg = np.array([1])
            hy = hy_pos
        if opt.method == 'mixed_posneg_br_one_euclidean':
            input_in = torch.matmul(A[:, :, :A.shape[1]], self.linear_edge_in(hidden)) + self.b_iah
            input_out = torch.matmul(A[:, :, A.shape[1]: 2 * A.shape[1]], self.linear_edge_out(hidden)) + self.b_oah
            inputs = torch.cat([input_in, input_out], 2)

            gi = F.linear(inputs, self.w_ih, self.b_ih)
            gh = F.linear(hidden, self.w_hh, self.b_hh)
            i_r, i_i, i_n = gi.chunk(3, 2)
            h_r, h_i, h_n = gh.chunk(3, 2)
            resetgate = torch.sigmoid(i_r + h_r)
            inputgate = torch.sigmoid(i_i + h_i)
            newgate = torch.tanh(i_n + resetgate * h_n)
            hy = newgate + inputgate * (hidden - newgate)

            gi_two = F.linear(inputs, self.w_ih_two, self.b_ih_two)
            gh_two = F.linear(hy, self.w_hh_two, self.b_hh_two)
            i_r_two, i_i_two, i_n_two = gi_two.chunk(3, 2)
            h_r_two, h_i_two, h_n_two = gh_two.chunk(3, 2)
            resetgate_two = torch.sigmoid(i_r_two + h_r_two)
            inputgate_two = torch.sigmoid(i_i_two + h_i_two)
            newgate_two = torch.tanh(i_n_two + resetgate_two * h_n_two)
            hy = newgate_two + inputgate_two * (hy - newgate_two)+hy

            hy_flatten = hy.reshape(hy.shape[0], hy.shape[1] * hy.shape[2]).cuda()
            if opt.distance == 'euclidean':
                hy_dis = cdist(hy_flatten.cpu().detach(), hy_flatten.cpu().detach(), metric='euclidean')
            elif opt.distance == 'cosine':
                hy_dis = cdist(hy_flatten.cpu().detach(), hy_flatten.cpu().detach(), metric='cosine')

            hy_dis = MinMaxScaler().fit(hy_dis).transform(hy_dis).T
            hy_dis = torch.tensor(hy_dis, dtype=torch.float32)
            #obtain positive k
            idx_k_pos = np.argsort(hy_dis)[:, :opt.k]
            idx_len_pos = idx_k_pos.shape[0]
            hy_k_pos = hy_dis[np.arange(0, idx_len_pos, 1).reshape(idx_len_pos, 1), idx_k_pos]

            #obatin positive A matrix
            A_ij_pos = torch.zeros(idx_len_pos,idx_len_pos)
            A_ij_pos[np.arange(0, idx_len_pos, 1).reshape(idx_len_pos, 1), idx_k_pos] = hy_k_pos

            #obatin negative Amatrix
            # idx_len_neg = idx_k_neg.shape[0]
            # hy_k_neg = hy_dis[np.arange(0, idx_len_neg, 1).reshape(idx_len_neg, 1), idx_k_neg]
            # hy_i_k_neg = hy_k_neg[:, -1]
            # hy_j_k_neg = hy_k_neg[idx_k_neg, -1]
            # mole_neg = -hy_k_neg
            # deno_neg = hy_j_k_neg * hy_i_k_neg.reshape(hy_i_k_neg.shape[0], 1)
            # hy_ij_neg = torch.exp(mole_neg / deno_neg)
            # A_ij_neg = torch.zeros(hy_ij_neg.shape[0], hy_ij_neg.shape[0])
            # A_ij_neg[np.arange(0, idx_len_neg, 1).reshape(idx_len_neg, 1), idx_k_neg] = hy_ij_neg

            hy_matmul_pos = torch.matmul(A_ij_pos.cuda(), hy_flatten)
            # hy_matmul_neg = torch.matmul(A_ij_neg.cuda(), hy_flatten)
            # hy_matmul = hy_matmul_pos*hy_matmul_neg

            hy_pos = hy_matmul_pos.reshape(hy.shape[0], hy.shape[1], hy.shape[2])
            # hy_neg = hy_matmul_neg.reshape(hy.shape[0], hy.shape[1], hy.shape[2])
            hy_neg = np.array([1])
            hy = hy_pos

        if opt.method == 'mixed_posneg_br_two':
            input_in = torch.matmul(A[:, :, :A.shape[1]], self.linear_edge_in(hidden)) + self.b_iah
            input_out = torch.matmul(A[:, :, A.shape[1]: 2 * A.shape[1]], self.linear_edge_out(hidden)) + self.b_oah
            inputs = torch.cat([input_in, input_out], 2)

            gi = F.linear(inputs, self.w_ih, self.b_ih)
            gh = F.linear(hidden, self.w_hh, self.b_hh)
            i_r, i_i, i_n = gi.chunk(3, 2)
            h_r, h_i, h_n = gh.chunk(3, 2)
            resetgate = torch.sigmoid(i_r + h_r)
            inputgate = torch.sigmoid(i_i + h_i)
            newgate = torch.tanh(i_n + resetgate * h_n)
            hy = newgate + inputgate * (hidden - newgate)

            gi_two = F.linear(inputs, self.w_ih_two, self.b_ih_two)
            gh_two = F.linear(hy, self.w_hh_two, self.b_hh_two)
            i_r_two, i_i_two, i_n_two = gi_two.chunk(3, 2)
            h_r_two, h_i_two, h_n_two = gh_two.chunk(3, 2)
            resetgate_two = torch.sigmoid(i_r_two + h_r_two)
            inputgate_two = torch.sigmoid(i_i_two + h_i_two)
            newgate_two = torch.tanh(i_n_two + resetgate_two * h_n_two)
            hy = newgate_two + inputgate_two * (hy - newgate_two) + hy

            hy_flatten = hy.reshape(hy.shape[0], hy.shape[1] * hy.shape[2]).cuda()
            hy_dis = cdist(hy_flatten.cpu().detach(), hy_flatten.cpu().detach(), metric='euclidean')
            hy_dis = MinMaxScaler().fit(hy_dis).transform(hy_dis).T
            hy_dis = torch.tensor(hy_dis, dtype=torch.float32)
            hy_dis = hy_dis.pow(2)
            # obtain positive k
            idx_k_pos = np.argsort(hy_dis)[:, :opt.k]
            # obtain positive k and  k of k
            idx_kk_pos = np.argsort(hy_dis)[:, :opt.k]
            idx_kk_pos_kk = idx_kk_pos[idx_kk_pos[:]][:, :, 1]
            idx_kk_pos = np.concatenate((tuple(idx_kk_pos.tolist()), tuple(idx_kk_pos_kk.tolist())), axis=1)
            temp_idx_kk_pos = np.array(idx_kk_pos)
            temp_hy_dis = np.array(hy_dis)
            idx_kk_pos_sort_index = np.argsort(
                hy_dis[np.arange(0, hy_dis.shape[0], 1).reshape(hy_dis.shape[0], 1), idx_kk_pos])
            idx_kk_pos = idx_kk_pos[
                np.arange(0, idx_kk_pos.shape[0], 1).reshape(idx_kk_pos.shape[0], 1), idx_kk_pos_sort_index]
            idx_k_pos = idx_kk_pos
            # obtatin negative k
            idx_k_neg = np.argsort(hy_dis)[:, -opt.k:]

            # obatin positive A matrix
            idx_len_pos = idx_k_pos.shape[0]
            hy_k_pos = hy_dis[np.arange(0, idx_len_pos, 1).reshape(idx_len_pos, 1), idx_k_pos]
            hy_i_k_pos = hy_k_pos[:, -1]
            hy_j_k_pos = hy_k_pos[idx_k_pos, -1]
            mole_pos = -hy_k_pos
            deno_pos = hy_j_k_pos * hy_i_k_pos.reshape(hy_i_k_pos.shape[0], 1)
            hy_ij_pos = torch.exp(mole_pos / deno_pos)
            A_ij_pos = torch.zeros(hy_ij_pos.shape[0], hy_ij_pos.shape[0])
            A_ij_pos[np.arange(0, idx_len_pos, 1).reshape(idx_len_pos, 1), idx_k_pos] = hy_ij_pos

            # obatin negative Amatrix
            idx_len_neg = idx_k_neg.shape[0]
            hy_k_neg = hy_dis[np.arange(0, idx_len_neg, 1).reshape(idx_len_neg, 1), idx_k_neg]
            hy_i_k_neg = hy_k_neg[:, -1]
            hy_j_k_neg = hy_k_neg[idx_k_neg, -1]
            mole_neg = -hy_k_neg
            deno_neg = hy_j_k_neg * hy_i_k_neg.reshape(hy_i_k_neg.shape[0], 1)
            hy_ij_neg = torch.exp(mole_neg / deno_neg)
            A_ij_neg = torch.zeros(hy_ij_neg.shape[0], hy_ij_neg.shape[0])
            A_ij_neg[np.arange(0, idx_len_neg, 1).reshape(idx_len_neg, 1), idx_k_neg] = hy_ij_neg

            hy_matmul_pos = torch.matmul(A_ij_pos.cuda(), hy_flatten)
            hy_matmul_neg = torch.matmul(A_ij_neg.cuda(), hy_flatten)
            # hy_matmul = hy_matmul_pos*hy_matmul_neg

            hy_pos = hy_matmul_pos.reshape(hy.shape[0], hy.shape[1], hy.shape[2])
            hy_neg = hy_matmul_neg.reshape(hy.shape[0], hy.shape[1], hy.shape[2])
            # hy_neg = None
            hy = hy_pos
        if opt.method == 'mixed_posneg_br_three':
            input_in = torch.matmul(A[:, :, :A.shape[1]], self.linear_edge_in(hidden)) + self.b_iah
            input_out = torch.matmul(A[:, :, A.shape[1]: 2 * A.shape[1]], self.linear_edge_out(hidden)) + self.b_oah
            inputs = torch.cat([input_in, input_out], 2)

            gi = F.linear(inputs, self.w_ih, self.b_ih)
            gh = F.linear(hidden, self.w_hh, self.b_hh)
            i_r, i_i, i_n = gi.chunk(3, 2)
            h_r, h_i, h_n = gh.chunk(3, 2)
            resetgate = torch.sigmoid(i_r + h_r)
            inputgate = torch.sigmoid(i_i + h_i)
            newgate = torch.tanh(i_n + resetgate * h_n)
            hy = newgate + inputgate * (hidden - newgate)

            gi_two = F.linear(inputs, self.w_ih_two, self.b_ih_two)
            gh_two = F.linear(hy, self.w_hh_two, self.b_hh_two)
            i_r_two, i_i_two, i_n_two = gi_two.chunk(3, 2)
            h_r_two, h_i_two, h_n_two = gh_two.chunk(3, 2)
            resetgate_two = torch.sigmoid(i_r_two + h_r_two)
            inputgate_two = torch.sigmoid(i_i_two + h_i_two)
            newgate_two = torch.tanh(i_n_two + resetgate_two * h_n_two)
            hy = newgate_two + inputgate_two * (hy - newgate_two) + hy

            hy_flatten = hy.reshape(hy.shape[0], hy.shape[1] * hy.shape[2]).cuda()
            hy_dis = cdist(hy_flatten.cpu().detach(), hy_flatten.cpu().detach(), metric='euclidean')
            hy_dis = MinMaxScaler().fit(hy_dis).transform(hy_dis).T
            hy_dis = torch.tensor(hy_dis, dtype=torch.float32)
            hy_dis = hy_dis.pow(2)
            # obtain positive k
            idx_k_pos = np.argsort(hy_dis)[:, :opt.k]
            # obtain positive k and  k of k
            idx_kk_pos = np.argsort(hy_dis)[:, :opt.k]
            idx_kk_pos_kk = idx_kk_pos[idx_kk_pos[:]][:, :, 1]
            idx_kk_pos = np.concatenate((tuple(idx_kk_pos.tolist()), tuple(idx_kk_pos_kk.tolist())), axis=1)
            idx_kk_pos_norepeat = np.array([list(np.unique(idx_kk_pos[i])) for i in range(len(idx_kk_pos))])
            repat_list = []
            no_repat_list=[]
            for i in range(len(idx_kk_pos)):
                repat_list.append([k for k, v in Counter(idx_kk_pos[i]).items() if v >= 2])
            for i in range(len(idx_kk_pos)):
                no_repat_list.append([k for k, v in Counter(idx_kk_pos[i]).items() if v <=1])
            temp_idx_kk_pos = np.array(idx_kk_pos)
            temp_hy_dis = np.array(hy_dis)
            idx_kk_pos_sort_index = np.argsort(
                hy_dis[np.arange(0, hy_dis.shape[0], 1).reshape(hy_dis.shape[0], 1), idx_kk_pos])
            idx_kk_pos = idx_kk_pos[
                np.arange(0, idx_kk_pos.shape[0], 1).reshape(idx_kk_pos.shape[0], 1), idx_kk_pos_sort_index]
            idx_k_pos = idx_kk_pos
            # obtatin negative k
            idx_k_neg = np.argsort(hy_dis)[:, -opt.k:]

            # obatin positive A matrix
            idx_len_pos = idx_k_pos.shape[0]
            hy_k_pos = hy_dis[np.arange(0, idx_len_pos, 1).reshape(idx_len_pos, 1), idx_k_pos]
            hy_i_k_pos = hy_k_pos[:, -1]
            hy_j_k_pos = hy_k_pos[idx_k_pos, -1]
            mole_pos = -hy_k_pos
            deno_pos = hy_j_k_pos * hy_i_k_pos.reshape(hy_i_k_pos.shape[0], 1)
            hy_ij_pos = torch.exp(mole_pos / deno_pos)
            A_ij_pos = torch.zeros(hy_ij_pos.shape[0], hy_ij_pos.shape[0])
            A_ij_pos[np.arange(0, idx_len_pos, 1).reshape(idx_len_pos, 1), idx_k_pos] = hy_ij_pos
            temp_A_ij_pos = np.array(A_ij_pos.cpu().detach())
            for i in range(idx_len_pos):
                A_ij_pos[i, no_repat_list[i]]=0
                A_ij_pos[i, i] = 1
            temp_B_ij_pos = np.array(A_ij_pos.cpu().detach())
            # obatin negative Amatrix
            idx_len_neg = idx_k_neg.shape[0]
            hy_k_neg = hy_dis[np.arange(0, idx_len_neg, 1).reshape(idx_len_neg, 1), idx_k_neg]
            hy_i_k_neg = hy_k_neg[:, -1]
            hy_j_k_neg = hy_k_neg[idx_k_neg, -1]
            mole_neg = -hy_k_neg
            deno_neg = hy_j_k_neg * hy_i_k_neg.reshape(hy_i_k_neg.shape[0], 1)
            hy_ij_neg = torch.exp(mole_neg / deno_neg)
            A_ij_neg = torch.zeros(hy_ij_neg.shape[0], hy_ij_neg.shape[0])
            A_ij_neg[np.arange(0, idx_len_neg, 1).reshape(idx_len_neg, 1), idx_k_neg] = hy_ij_neg

            hy_matmul_pos = torch.matmul(A_ij_pos.cuda(), hy_flatten)
            hy_matmul_neg = torch.matmul(A_ij_neg.cuda(), hy_flatten)
            # hy_matmul = hy_matmul_pos*hy_matmul_neg

            hy_pos = hy_matmul_pos.reshape(hy.shape[0], hy.shape[1], hy.shape[2])
            hy_neg = hy_matmul_neg.reshape(hy.shape[0], hy.shape[1], hy.shape[2])
            # hy_neg = None
            hy = hy_pos

        if opt.method == 'two_gru_resmod_changea':
            input_in = torch.matmul(A[:, :, :A.shape[1]], self.linear_edge_in(hidden)) + self.b_iah
            input_out = torch.matmul(A[:, :, A.shape[1]: 2 * A.shape[1]], self.linear_edge_out(hidden)) + self.b_oah
            inputs = torch.cat([input_in, input_out], 2)

            gi = F.linear(inputs, self.w_ih, self.b_ih)
            gh = F.linear(hidden, self.w_hh, self.b_hh)
            i_r, i_i, i_n = gi.chunk(3, 2)
            h_r, h_i, h_n = gh.chunk(3, 2)
            resetgate = torch.relu(i_r + h_r)
            inputgate = torch.relu(i_i + h_i)
            newgate = torch.relu(i_n + resetgate * h_n)
            hy = newgate + inputgate * (hidden - newgate)

            gi_two = F.linear(inputs, self.w_ih_two, self.b_ih_two)
            gh_two = F.linear(hy, self.w_hh_two, self.b_hh_two)
            i_r_two, i_i_two, i_n_two = gi_two.chunk(3, 2)
            h_r_two, h_i_two, h_n_two = gh_two.chunk(3, 2)
            resetgate_two = torch.relu(i_r_two + h_r_two)
            inputgate_two = torch.relu(i_i_two + h_i_two)
            newgate_two = torch.relu(i_n_two + resetgate_two * h_n_two)
            hy = newgate_two + inputgate_two * (hy - newgate_two)+hy


        if opt.method == 'before_two_gcn':
            input_in = torch.matmul(A[:, :, :A.shape[1]], self.linear_edge_in(hidden)) + self.b_iah
            input_in = torch.relu(input_in)
            input_in = torch.matmul(A[:, :, :A.shape[1]], self.linear_edge_in_two(input_in)) + self.b_iah_two
            input_in = torch.relu(input_in)

            input_out = torch.matmul(A[:, :, A.shape[1]: 2 * A.shape[1]], self.linear_edge_out(hidden)) + self.b_oah
            input_out = torch.relu(input_out)
            input_out = torch.matmul(A[:, :, A.shape[1]: 2 * A.shape[1]], self.linear_edge_out_two(input_out)) + self.b_oah_two
            input_out = torch.relu(input_out)
            inputs = torch.cat([input_in, input_out], 2)
            gi = F.linear(inputs, self.w_ih, self.b_ih)
            gh = F.linear(hidden, self.w_hh, self.b_hh)
            i_r, i_i, i_n = gi.chunk(3, 2)
            h_r, h_i, h_n = gh.chunk(3, 2)
            resetgate = torch.sigmoid(i_r + h_r)
            inputgate = torch.sigmoid(i_i + h_i)
            newgate = torch.tanh(i_n + resetgate * h_n)
            hy = newgate + inputgate * (hidden - newgate)
            # hy = (1 - inputgate) * hidden + inputgate * newgate
        if opt.method == 'add_adjacency_wb_reul':
            input_in = torch.matmul(A[:, :, :A.shape[1]], self.linear_edge_in(hidden)) + self.b_iah
            input_out = torch.matmul(A[:, :, A.shape[1]: 2 * A.shape[1]], self.linear_edge_out(hidden)) + self.b_oah
            inputs = torch.cat([input_in, input_out], 2)
            gi = F.linear(inputs, self.w_ih, self.b_ih)
            gh = F.linear(hidden, self.w_hh, self.b_hh)
            i_r, i_i, i_n = gi.chunk(3, 2)
            h_r, h_i, h_n = gh.chunk(3, 2)
            resetgate = torch.sigmoid(i_r + h_r)
            inputgate = torch.sigmoid(i_i + h_i)
            newgate = torch.tanh(i_n + resetgate * h_n)
            hy = newgate + inputgate * (hidden - newgate)
            hy_flatten = hy.reshape(hy.shape[0], hy.shape[1] * hy.shape[2]).cuda()
            hy_dis = cdist(hy_flatten.cpu().detach(), hy_flatten.cpu().detach(), metric='euclidean')
            hy_dis = MinMaxScaler().fit(hy_dis).transform(hy_dis).T
            hy_dis = torch.tensor(hy_dis, dtype=torch.float32)
            hy_dis = hy_dis.pow(2)
            idx_k = np.argsort(hy_dis)[:, :30]
            idx_len = idx_k.shape[0]
            hy_k = hy_dis[np.arange(0, idx_len, 1).reshape(idx_len, 1), idx_k]
            hy_i_k = hy_k[:, -1]
            hy_j_k = hy_k[idx_k, -1]
            mole = -hy_k
            deno = hy_j_k * hy_i_k.reshape(hy_i_k.shape[0], 1)
            hy_ij = torch.exp(mole / deno)
            A_ij = torch.zeros(hy_ij.shape[0], hy_ij.shape[0])
            A_ij[np.arange(0, idx_len, 1).reshape(idx_len, 1), idx_k] = hy_ij
            hy_matmul = torch.matmul(A_ij.cuda(), hy_flatten)
            hy_linnear=torch.nn.Linear(hy_matmul.shape[1],hy_matmul.shape[1]).cuda()
            with torch.no_grad():
                hy_matmul = hy_linnear(hy_matmul.cuda())
            hy_matmul=torch.relu(hy_matmul)
            hy = hy_matmul.reshape(hy.shape[0], hy.shape[1], hy.shape[2])
        if opt.method == 'add_adjacency_wb':
            input_in = torch.matmul(A[:, :, :A.shape[1]], self.linear_edge_in(hidden)) + self.b_iah
            input_out = torch.matmul(A[:, :, A.shape[1]: 2 * A.shape[1]], self.linear_edge_out(hidden)) + self.b_oah
            inputs = torch.cat([input_in, input_out], 2)
            gi = F.linear(inputs, self.w_ih, self.b_ih)
            gh = F.linear(hidden, self.w_hh, self.b_hh)
            i_r, i_i, i_n = gi.chunk(3, 2)
            h_r, h_i, h_n = gh.chunk(3, 2)
            resetgate = torch.sigmoid(i_r + h_r)
            inputgate = torch.sigmoid(i_i + h_i)
            newgate = torch.tanh(i_n + resetgate * h_n)
            hy = newgate + inputgate * (hidden - newgate)
            hy_flatten = hy.reshape(hy.shape[0], hy.shape[1] * hy.shape[2]).cuda()
            hy_dis = cdist(hy_flatten.cpu().detach(), hy_flatten.cpu().detach(), metric='euclidean')
            hy_dis = MinMaxScaler().fit(hy_dis).transform(hy_dis).T
            hy_dis = torch.tensor(hy_dis, dtype=torch.float32)
            hy_dis = hy_dis.pow(2)
            idx_k = np.argsort(hy_dis)[:, :30]
            idx_len = idx_k.shape[0]
            hy_k = hy_dis[np.arange(0, idx_len, 1).reshape(idx_len, 1), idx_k]
            hy_i_k = hy_k[:, -1]
            hy_j_k = hy_k[idx_k, -1]
            mole = -hy_k
            deno = hy_j_k * hy_i_k.reshape(hy_i_k.shape[0], 1)
            hy_ij = torch.exp(mole / deno)
            A_ij = torch.zeros(hy_ij.shape[0], hy_ij.shape[0])
            A_ij[np.arange(0, idx_len, 1).reshape(idx_len, 1), idx_k] = hy_ij
            hy_matmul = torch.matmul(A_ij.cuda(), hy_flatten)
            hy_linnear=torch.nn.Linear(hy_matmul.shape[1],hy_matmul.shape[1]).cuda()
            with torch.no_grad():
                hy_matmul = hy_linnear(hy_matmul.cuda())
            # hy_matmul=torch.relu(hy_matmul)
            hy = hy_matmul.reshape(hy.shape[0], hy.shape[1], hy.shape[2])
        if opt.method == 'add_adjacency_wb_a':
            #update wb
            input_in = torch.matmul(A[:, :, :A.shape[1]], self.linear_edge_in(hidden)) + self.b_iah
            input_out = torch.matmul(A[:, :, A.shape[1]: 2 * A.shape[1]], self.linear_edge_out(hidden)) + self.b_oah
            inputs = torch.cat([input_in, input_out], 2)
            gi = F.linear(inputs, self.w_ih, self.b_ih)
            gh = F.linear(hidden, self.w_hh, self.b_hh)
            i_r, i_i, i_n = gi.chunk(3, 2)
            h_r, h_i, h_n = gh.chunk(3, 2)
            resetgate = torch.sigmoid(i_r + h_r)
            inputgate = torch.sigmoid(i_i + h_i)
            newgate = torch.tanh(i_n + resetgate * h_n)
            hy = newgate + inputgate * (hidden - newgate)
            hy_flatten = hy.reshape(hy.shape[0], hy.shape[1] * hy.shape[2]).cuda()
            hy_dis = cdist(hy_flatten.cpu().detach(), hy_flatten.cpu().detach(), metric='euclidean')
            hy_dis = MinMaxScaler().fit(hy_dis).transform(hy_dis).T
            hy_dis = torch.tensor(hy_dis, dtype=torch.float32)
            hy_dis = hy_dis.pow(2)
            idx_k = np.argsort(hy_dis)[:, :opt.k]
            idx_len = idx_k.shape[0]
            hy_k = hy_dis[np.arange(0, idx_len, 1).reshape(idx_len, 1), idx_k]
            hy_i_k = hy_k[:, -1]
            hy_j_k = hy_k[idx_k, -1]
            mole = -hy_k
            deno = hy_j_k * hy_i_k.reshape(hy_i_k.shape[0], 1)
            hy_ij = torch.exp(mole / deno)
            A_ij = torch.zeros(hy_ij.shape[0], hy_ij.shape[0])
            A_ij[np.arange(0, idx_len, 1).reshape(idx_len, 1), idx_k] = hy_ij
            hy_matmul = torch.matmul(A_ij.cuda(), hy_flatten)
            hy = hy_matmul.reshape(hy.shape[0], hy.shape[1], hy.shape[2])
            hy = self.linear_edge_adj(hy) + self.b_adj
        if opt.method=='add_adjacencyAeye':
            input_in = torch.matmul(A[:, :, :A.shape[1]]+torch.eye(A.shape[1]).cuda(), self.linear_edge_in(hidden)) + self.b_iah
            input_out = torch.matmul(A[:, :, A.shape[1]: 2 * A.shape[1]]+torch.eye(A.shape[1]).cuda(), self.linear_edge_out(hidden)) + self.b_oah
            inputs = torch.cat([input_in, input_out], 2)
            gi = F.linear(inputs, self.w_ih, self.b_ih)
            gh = F.linear(hidden, self.w_hh, self.b_hh)
            i_r, i_i, i_n = gi.chunk(3, 2)
            h_r, h_i, h_n = gh.chunk(3, 2)
            resetgate = torch.sigmoid(i_r + h_r)
            inputgate = torch.sigmoid(i_i + h_i)
            newgate = torch.tanh(i_n + resetgate * h_n)
            hy = newgate + inputgate * (hidden - newgate)
            hy_flatten = hy.reshape(hy.shape[0], hy.shape[1]*hy.shape[2]).cuda()
            hy_dis = cdist(hy_flatten.cpu().detach(), hy_flatten.cpu().detach(), metric='euclidean')
            hy_dis = MinMaxScaler().fit(hy_dis).transform(hy_dis).T
            hy_dis = torch.tensor(hy_dis, dtype=torch.float32)
            hy_dis = hy_dis.pow(2)
            idx_k = np.argsort(hy_dis)[:,:opt.k]
            idx_len = idx_k.shape[0]
            hy_k = hy_dis[np.arange(0,idx_len,1).reshape(idx_len,1),idx_k]
            hy_i_k = hy_k[:,-1]
            hy_j_k = hy_k[idx_k,-1]
            mole = -hy_k
            deno = hy_j_k*hy_i_k.reshape(hy_i_k.shape[0],1)
            hy_ij = torch.exp(mole/deno)
            A_ij = torch.zeros(hy_ij.shape[0],hy_ij.shape[0])
            A_ij[np.arange(0,idx_len,1).reshape(idx_len,1),idx_k]=hy_ij
            hy_matmul = torch.matmul(A_ij.cuda(),hy_flatten)
            hy = hy_matmul.reshape(hy.shape[0],hy.shape[1],hy.shape[2])
        if opt.method=='s_useinput':
            input_in = torch.matmul(A[:, :, :A.shape[1]], self.linear_edge_in(hidden)) + self.b_iah
            input_out = torch.matmul(A[:, :, A.shape[1]: 2 * A.shape[1]], self.linear_edge_out(hidden)) + self.b_oah
            inputs = input_in
            gi = F.linear(inputs, self.w_ih, self.b_ih)
            gh = F.linear(hidden, self.w_hh, self.b_hh)
            i_r, i_i, i_n = gi.chunk(3, 2)
            h_r, h_i, h_n = gh.chunk(3, 2)
            resetgate = torch.sigmoid(i_r + h_r)
            inputgate = torch.sigmoid(i_i + h_i)
            newgate = torch.tanh(i_n + resetgate * h_n)
            hy = newgate + inputgate * (hidden - newgate)

        if opt.method=='s_useoutput':
            input_in = torch.matmul(A[:, :, :A.shape[1]], self.linear_edge_in(hidden)) + self.b_iah
            input_out = torch.matmul(A[:, :, A.shape[1]: 2 * A.shape[1]], self.linear_edge_out(hidden)) + self.b_oah
            inputs = input_out
            gi = F.linear(inputs, self.w_ih, self.b_ih)
            gh = F.linear(hidden, self.w_hh, self.b_hh)
            i_r, i_i, i_n = gi.chunk(3, 2)
            h_r, h_i, h_n = gh.chunk(3, 2)
            resetgate = torch.sigmoid(i_r + h_r)
            inputgate = torch.sigmoid(i_i + h_i)
            newgate = torch.tanh(i_n + resetgate * h_n)
            hy = newgate + inputgate * (hidden - newgate)

        if opt.method == 's_in_out_exchange':
            input_in = torch.matmul(A[:, :, :A.shape[1]], self.linear_edge_in(hidden)) + self.b_iah
            input_out = torch.matmul(A[:, :, A.shape[1]: 2 * A.shape[1]], self.linear_edge_out(hidden)) + self.b_oah
            inputs = torch.cat([input_out, input_in], 2)
            gi = F.linear(inputs, self.w_ih, self.b_ih)
            gh = F.linear(hidden, self.w_hh, self.b_hh)
            i_r, i_i, i_n = gi.chunk(3, 2)
            h_r, h_i, h_n = gh.chunk(3, 2)
            resetgate = torch.sigmoid(i_r + h_r)
            inputgate = torch.sigmoid(i_i + h_i)
            newgate = torch.tanh(i_n + resetgate * h_n)
            hy = newgate + inputgate * (hidden - newgate)

        if opt.method=='add_adjacency':
            input_in = torch.matmul(A[:, :, :A.shape[1]], self.linear_edge_in(hidden)) + self.b_iah
            input_out = torch.matmul(A[:, :, A.shape[1]: 2 * A.shape[1]], self.linear_edge_out(hidden)) + self.b_oah
            inputs = torch.cat([input_in, input_out], 2)
            gi = F.linear(inputs, self.w_ih, self.b_ih)
            gh = F.linear(hidden, self.w_hh, self.b_hh)
            i_r, i_i, i_n = gi.chunk(3, 2)
            h_r, h_i, h_n = gh.chunk(3, 2)
            resetgate = torch.sigmoid(i_r + h_r)
            inputgate = torch.sigmoid(i_i + h_i)
            newgate = torch.tanh(i_n + resetgate * h_n)
            hy = newgate + inputgate * (hidden - newgate)
            hy_flatten = hy.reshape(hy.shape[0], hy.shape[1]*hy.shape[2]).cuda()
            hy_dis = cdist(hy_flatten.cpu().detach(), hy_flatten.cpu().detach(), metric='euclidean')
            hy_dis = MinMaxScaler().fit(hy_dis).transform(hy_dis).T
            hy_dis = torch.tensor(hy_dis, dtype=torch.float32)
            hy_dis = hy_dis.pow(2)
            idx_k = np.argsort(hy_dis)[:,:opt.k]
            idx_len = idx_k.shape[0]
            hy_k = hy_dis[np.arange(0,idx_len,1).reshape(idx_len,1),idx_k]
            hy_i_k = hy_k[:,-1]
            hy_j_k = hy_k[idx_k,-1]
            mole = -hy_k
            deno = hy_j_k*hy_i_k.reshape(hy_i_k.shape[0],1)
            hy_ij = torch.exp(mole/deno)
            A_ij = torch.zeros(hy_ij.shape[0],hy_ij.shape[0])
            A_ij[np.arange(0,idx_len,1).reshape(idx_len,1),idx_k]=hy_ij
            hy_matmul = torch.matmul(A_ij.cuda(),hy_flatten)
            hy = hy_matmul.reshape(hy.shape[0],hy.shape[1],hy.shape[2])

        if opt.method=='add_adjacency_method2':
            input_in = torch.matmul(A[:, :, :A.shape[1]], self.linear_edge_in(hidden)) + self.b_iah
            input_out = torch.matmul(A[:, :, A.shape[1]: 2 * A.shape[1]], self.linear_edge_out(hidden)) + self.b_oah
            inputs = torch.cat([input_in, input_out], 2)
            gi = F.linear(inputs, self.w_ih, self.b_ih)
            gh = F.linear(hidden, self.w_hh, self.b_hh)
            i_r, i_i, i_n = gi.chunk(3, 2)
            h_r, h_i, h_n = gh.chunk(3, 2)
            resetgate = torch.sigmoid(i_r + h_r)
            inputgate = torch.sigmoid(i_i + h_i)
            newgate = torch.tanh(i_n + resetgate * h_n)
            hy = newgate + inputgate * (hidden - newgate)
            hy_flatten = hy.reshape(hy.shape[0], hy.shape[1]*hy.shape[2]).cuda()
            hy_dis = cdist(hy_flatten.cpu().detach(), hy_flatten.cpu().detach(), metric='euclidean')
            hy_dis = MinMaxScaler().fit(hy_dis).transform(hy_dis).T
            hy_dis = torch.tensor(hy_dis, dtype=torch.float32)
            hy_dis = hy_dis.pow(2)
            idx_k = np.argsort(hy_dis)[:,:opt.k]
            idx_len = idx_k.shape[0]
            hy_k = hy_dis[np.arange(0,idx_len,1).reshape(idx_len,1),idx_k]
            hy_i_k = hy_k[:,-1]
            hy_j_k = hy_k[idx_k,-1]
            mole = -hy_k
            deno = hy_j_k*hy_i_k.reshape(hy_i_k.shape[0],1)
            hy_ij = torch.exp(mole/deno)
            A_ij = torch.zeros(hy_ij.shape[0],hy_ij.shape[0])
            A_ij[np.arange(0,idx_len,1).reshape(idx_len,1),idx_k]=hy_ij
            rowsum=A_ij.sum(1)
            D=torch.pow(rowsum, -0.5)
            D = torch.diag(D)
            A_ij = torch.matmul(torch.matmul(D, A_ij), D)
            hy_matmul = torch.matmul(A_ij.cuda(),hy_flatten)
            hy = hy_matmul.reshape(hy.shape[0],hy.shape[1],hy.shape[2])

        if opt.method=='add_adjacency_method3':
            input_in = torch.matmul(A[:, :, :A.shape[1]], self.linear_edge_in(hidden)) + self.b_iah
            input_out = torch.matmul(A[:, :, A.shape[1]: 2 * A.shape[1]], self.linear_edge_out(hidden)) + self.b_oah
            inputs = torch.cat([input_in, input_out], 2)
            gi = F.linear(inputs, self.w_ih, self.b_ih)
            gh = F.linear(hidden, self.w_hh, self.b_hh)
            i_r, i_i, i_n = gi.chunk(3, 2)
            h_r, h_i, h_n = gh.chunk(3, 2)
            resetgate = torch.sigmoid(i_r + h_r)
            inputgate = torch.sigmoid(i_i + h_i)
            newgate = torch.tanh(i_n + resetgate * h_n)
            hy = newgate + inputgate * (hidden - newgate)
            hy_flatten = hy.reshape(hy.shape[0], hy.shape[1]*hy.shape[2]).cuda()
            hy_dis = cdist(hy_flatten.cpu().detach(), hy_flatten.cpu().detach(), metric='euclidean')
            hy_dis = MinMaxScaler().fit(hy_dis).transform(hy_dis).T
            hy_dis = torch.tensor(hy_dis, dtype=torch.float32)
            hy_dis = hy_dis.pow(2)
            idx_k = np.argsort(hy_dis)[:,:opt.k]
            idx_len = idx_k.shape[0]
            hy_k = hy_dis[np.arange(0,idx_len,1).reshape(idx_len,1),idx_k]
            hy_i_k = hy_k[:,-1]
            hy_j_k = hy_k[idx_k,-1]
            mole = -hy_k
            deno = hy_j_k*hy_i_k.reshape(hy_i_k.shape[0],1)
            hy_ij = torch.exp(mole/deno)
            A_ij = torch.zeros(hy_ij.shape[0],hy_ij.shape[0])
            A_ij[np.arange(0,idx_len,1).reshape(idx_len,1),idx_k]=hy_ij
            A_ij = A_ij + torch.eye(A_ij.shape[0])
            rowsum=A_ij.sum(1)
            D=torch.pow(rowsum, -0.5)
            D = torch.diag(D)
            A_ij = torch.matmul(torch.matmul(D, A_ij), D)
            hy_matmul = torch.matmul(A_ij.cuda(),hy_flatten)
            hy = hy_matmul.reshape(hy.shape[0],hy.shape[1],hy.shape[2])

        #############################test12_start Similarity from cons##########################################
        # A_flatten=A.reshape(A.shape[0],A.shape[1]*A.shape[2]).cuda()
        # A_flatten_temp = np.array(A_flatten.cpu())
        # cos_A = cosine_similarity(A_flatten.cpu(), A_flatten.cpu())
        # cos_A = torch.tensor(cos_A, dtype=torch.float32)
        # A=torch.matmul(cos_A.cuda(),A_flatten).reshape(A.shape[0],A.shape[1],A.shape[2])
        # input_in = torch.matmul(A[:, :, :A.shape[1]], self.linear_edge_in(hidden)) + self.b_iah
        # input_out = torch.matmul(A[:, :, A.shape[1]: 2 * A.shape[1]], self.linear_edge_out(hidden)) + self.b_oah

        #############################test10_start(input diag ,output diag)##########################################
        # A_input_in = A[:, :, :A.shape[1]]
        # s_input_in = A[:,:,:A.shape[1]].reshape(A.shape[0],A.shape[1]*A.shape[1]).cuda()
        # dis_input = cdist(s_input_in.cpu(), s_input_in.cpu(), metric='euclidean')
        # dis_input = MinMaxScaler().fit(dis_input).transform(dis_input).T
        # dis_input = torch.tensor(dis_input,dtype=torch.float32)
        # dis_diag = torch.eye(dis_input.shape[0], dis_input.shape[1])
        # dis_input = dis_input+dis_diag
        # A_input_in = torch.matmul(dis_input.cuda(), s_input_in).reshape(A_input_in.shape[0],
        #                                                                          A_input_in.shape[1],
        #                                                                          A_input_in.shape[2])
        # input_in = torch.matmul(A_input_in, self.linear_edge_in(hidden)) + self.b_iah
        #
        # A_input_out = A[:, :, A.shape[1]: 2 * A.shape[1]]
        # s_input_out = A[:, :, A.shape[1]: 2 * A.shape[1]].reshape(A.shape[0], A.shape[1] * A.shape[1]).cuda()
        # dis_output = cdist(s_input_out.cpu(), s_input_out.cpu(), metric='euclidean')
        # dis_output = MinMaxScaler().fit(dis_output).transform(dis_output).T
        # dis_output = torch.tensor(dis_output, dtype=torch.float32)
        # dis_output = dis_output+dis_diag
        # A_input_out = torch.matmul(dis_output.cuda(), s_input_out).reshape(A_input_out.shape[0],
        #                                                                 A_input_out.shape[1],
        #                                                                 A_input_out.shape[2])
        # input_out = torch.matmul(A_input_out, self.linear_edge_out(hidden)) + self.b_oah
        #############################test9_start(input diat)##########################################
        # A_input_in = A[:, :, :A.shape[1]]
        # s_input_in = A[:,:,:A.shape[1]].reshape(A.shape[0],A.shape[1]*A.shape[1]).cuda()
        # dis_input = cdist(s_input_in.cpu(), s_input_in.cpu(), metric='euclidean')
        # dis_input = MinMaxScaler().fit(dis_input).transform(dis_input).T
        # dis_input = torch.tensor(dis_input,dtype=torch.float32)
        # dis_diag = torch.eye(dis_input.shape[0], dis_input.shape[1])
        # dis_input = dis_input+dis_diag
        # A_input_in = torch.matmul(dis_input.cuda(), s_input_in).reshape(A_input_in.shape[0],
        #                                                                          A_input_in.shape[1],
        #                                                                          A_input_in.shape[2])
        # input_in = torch.matmul(A_input_in, self.linear_edge_in(hidden)) + self.b_iah
        #
        # A_input_out = A[:, :, A.shape[1]: 2 * A.shape[1]]
        # s_input_out = A[:, :, A.shape[1]: 2 * A.shape[1]].reshape(A.shape[0], A.shape[1] * A.shape[1]).cuda()
        # dis_output = cdist(s_input_out.cpu(), s_input_out.cpu(), metric='euclidean')
        # dis_output = MinMaxScaler().fit(dis_output).transform(dis_output).T
        # dis_output = torch.tensor(dis_output, dtype=torch.float32)
        # A_input_out = torch.matmul(dis_output.cuda(), s_input_out).reshape(A_input_out.shape[0],
        #                                                                 A_input_out.shape[1],
        #                                                                 A_input_out.shape[2])
        # input_out = torch.matmul(A_input_out, self.linear_edge_out(hidden)) + self.b_oah
        #############################test8_start##########################################
        # A_input_in = A[:, :, :A.shape[1]]
        # s_input_in = A[:,:,:A.shape[1]].reshape(A.shape[0],A.shape[1]*A.shape[1]).cuda()
        # dis_input = cdist(s_input_in.cpu(), s_input_in.cpu(), metric='euclidean')
        # dis_input = MinMaxScaler().fit(dis_input).transform(dis_input).T
        # dis_input = torch.tensor(dis_input,dtype=torch.float32)
        # A_input_in = torch.matmul(dis_input.cuda(), s_input_in).reshape(A_input_in.shape[0],
        #                                                                          A_input_in.shape[1],
        #                                                                          A_input_in.shape[2])
        # input_in = torch.matmul(A_input_in, self.linear_edge_in(hidden)) + self.b_iah
        #
        # A_input_out = A[:, :, A.shape[1]: 2 * A.shape[1]]
        # s_input_out = A[:, :, A.shape[1]: 2 * A.shape[1]].reshape(A.shape[0], A.shape[1] * A.shape[1]).cuda()
        # dis_output = cdist(s_input_out.cpu(), s_input_out.cpu(), metric='euclidean')
        # dis_output = MinMaxScaler().fit(dis_output).transform(dis_output).T
        # dis_output = torch.tensor(dis_output, dtype=torch.float32)
        # A_input_out = torch.matmul(dis_output.cuda(), s_input_out).reshape(A_input_out.shape[0],
        #                                                                 A_input_out.shape[1],
        #                                                                 A_input_out.shape[2])
        # input_out = torch.matmul(A_input_out, self.linear_edge_out(hidden)) + self.b_oah
        #############################test7_start##########################################
        # A_input_in = A[:, :, :A.shape[1]]
        # s_input_in = A[:,:,:A.shape[1]].reshape(A.shape[0],A.shape[1]*A.shape[1]).cuda()
        # dis_input = cdist(s_input_in.cpu(), s_input_in.cpu(), metric='euclidean')
        # dis_input = MinMaxScaler().fit(dis_input).transform(dis_input).T
        # dis_input = torch.tensor(dis_input,dtype=torch.float32)
        # A_input_in = torch.matmul(dis_input.cuda(), s_input_in).reshape(A_input_in.shape[0],
        #                                                                          A_input_in.shape[1],
        #                                                                          A_input_in.shape[2])
        # input_in = torch.matmul(A_input_in, self.linear_edge_in(hidden)) + self.b_iah
        #
        # A_input_out = A[:, :, A.shape[1]: 2 * A.shape[1]]
        # s_input_out = A[:, :, A.shape[1]: 2 * A.shape[1]].reshape(A.shape[0], A.shape[1] * A.shape[1]).cuda()
        # dis_output = cdist(s_input_out.cpu(), s_input_out.cpu(), metric='euclidean')
        #
        # dis_output = torch.tensor(dis_output, dtype=torch.float32)
        # A_input_out = torch.matmul(dis_output.cuda(), s_input_out).reshape(A_input_out.shape[0],
        #                                                                 A_input_out.shape[1],
        #                                                                 A_input_out.shape[2])
        # input_out = torch.matmul(A_input_out, self.linear_edge_out(hidden)) + self.b_oah

        #############################test6_start##########################################
        # A_input_in = A[:, :, :A.shape[1]]
        # s_input_in = A[:,:,:A.shape[1]].reshape(A.shape[0],A.shape[1]*A.shape[1]).cuda()
        # dis_input = cdist(s_input_in.cpu(), s_input_in.cpu(), metric='euclidean')
        # dis_input = torch.tensor(dis_input,dtype=torch.float32)
        # A_input_in = torch.matmul(dis_input.cuda(), s_input_in).reshape(A_input_in.shape[0],
        #                                                                          A_input_in.shape[1],
        #                                                                          A_input_in.shape[2])
        # input_in = torch.matmul(A_input_in, self.linear_edge_in(hidden)) + self.b_iah
        #
        # A_input_out = A[:, :, A.shape[1]: 2 * A.shape[1]]
        # s_input_out = A[:, :, A.shape[1]: 2 * A.shape[1]].reshape(A.shape[0], A.shape[1] * A.shape[1]).cuda()
        # dis_output = cdist(s_input_out.cpu(), s_input_out.cpu(), metric='euclidean')
        # dis_output = torch.tensor(dis_output, dtype=torch.float32)
        # A_input_out = torch.matmul(dis_output.cuda(), s_input_out).reshape(A_input_out.shape[0],
        #                                                                 A_input_out.shape[1],
        #                                                                 A_input_out.shape[2])
        # input_out = torch.matmul(A_input_out, self.linear_edge_out(hidden)) + self.b_oah

        #############################test5_start##########################################
        # A_input_in = A[:, :, :A.shape[1]]
        # s_input_in = A[:,:,:A.shape[1]].reshape(A.shape[0],A.shape[1]*A.shape[1]).cuda()
        # dis = cdist(s_input_in.cpu(), s_input_in.cpu(), metric='euclidean')
        # dis = torch.tensor(dis,dtype=torch.float32)
        # A_input_in = torch.matmul(dis.cuda(), s_input_in).reshape(A_input_in.shape[0],
        #                                                                          A_input_in.shape[1],
        #                                                                          A_input_in.shape[2])
        # input_in = torch.matmul(A_input_in, self.linear_edge_in(hidden)) + self.b_iah
        # print("Test time start")
        #############################test4_start##########################################
        # A_input_in = A[:, :, :A.shape[1]]
        # s_input_in = A[:,:,:A.shape[1]].reshape(A.shape[0],A.shape[1]*A.shape[1]).cuda()
        # s_input_in_test = np.array(s_input_in.cpu())
        # s_input_in_one = torch.ones(1,s_input_in.shape[1]).cuda()
        # s_input_in_if = s_input_in.sum(axis=1).cuda()
        # s_input_in[s_input_in_if[:]==0.] = s_input_in_one
        # module_len_all = torch.sqrt(torch.sum(s_input_in ** 2, axis=1))
        # matrix_deno = torch.matmul(module_len_all.reshape(module_len_all.shape[0],1), module_len_all.reshape(1,module_len_all.shape[0]))
        # matrix_mole = torch.matmul(s_input_in,s_input_in.T)
        # matrix_deno_result = matrix_mole/matrix_deno
        # A_input_in = torch.matmul(matrix_deno_result.cuda(),s_input_in).reshape(A_input_in.shape[0],A_input_in.shape[1],A_input_in.shape[2])
        # input_in = torch.matmul(A_input_in, self.linear_edge_in(hidden)) + self.b_iah
        # print("Test time start")
        #############################test3_start##########################################
        # A_input_in = A[:, :, :A.shape[1]]
        # s_input_in = A[:, :, :A.shape[1]].reshape(A.shape[0], A.shape[1] * A.shape[1]).cuda()
        # module_len_all=torch.sqrt(torch.sum(s_input_in**2,axis=1))
        # mul_all=torch.matmul(s_input_in,s_input_in.T)
        # result = torch.eye(len(module_len_all),len(module_len_all))
        # for i in range(0,len(module_len_all)):
        #     for j in range(0,len(module_len_all)):
        #         mole=mul_all[i,j]
        #         deno = module_len_all[i] * module_len_all[j]
        #         if deno==0:
        #             if module_len_all[i]==0 and module_len_all[j]==0:
        #                 result[i,j]=1
        #             else:
        #                 result[i, j] = 0
        #         else:
        #             result[i,j] = mole/deno
        # A_input_in = torch.matmul(result.cuda(),s_input_in).reshape(A_input_in.shape[0],A_input_in.shape[1],A_input_in.shape[2])
        # input_in = torch.matmul(A_input_in, self.linear_edge_in(hidden)) + self.b_iah
        # print("Test time start")
        #############################test2_start##########################################
        # input_in = torch.matmul(A[:, :, :A.shape[1]], self.linear_edge_in(hidden)) + self.b_iah
        # s_input_in = A[:,:,:A.shape[1]].reshape(A.shape[0],A.shape[1]*A.shape[1]).cuda()
        # s_input_in_one = torch.ones(1,s_input_in.shape[1]).cuda()
        # s_input_in_if = s_input_in.sum(axis=1).cuda()
        # s_input_in[s_input_in_if[:]==0.] = s_input_in_one
        # module_len_all = torch.sqrt(torch.sum(s_input_in ** 2, axis=1))
        # matrix_mole = torch.matmul(module_len_all.reshape(module_len_all.shape[0],1), module_len_all.reshape(1,module_len_all.shape[0]))
        # matrix_deno = torch.matmul(s_input_in,s_input_in.T)
        # matrix_deno_result = matrix_mole/matrix_deno
        # input_in=torch.matmul(input_in, matrix_deno_result.cuda())
        #############################test1_start##########################################
        # input_in = torch.matmul(A[:, :, :A.shape[1]], self.linear_edge_in(hidden)) + self.b_iah
        # s_input_in = A[:, :, :A.shape[1]].reshape(A.shape[0], A.shape[1] * A.shape[1]).cuda()
        # module_len_all=torch.sqrt(torch.sum(s_input_in**2,axis=1))
        # mul_all=torch.matmul(s_input_in,s_input_in.T)
        # result = torch.eye(len(module_len_all),len(module_len_all))
        # for i in range(0,len(module_len_all)):
        #     for j in range(0,len(module_len_all)):
        #         mole=mul_all[i,j]
        #         deno = module_len_all[i] * module_len_all[j]
        #         if deno==0:
        #             if module_len_all[i]==0 and module_len_all[j]==0:
        #                 result[i,j]=1
        #             else:
        #                 result[i, j] = 0
        #         else:
        #             result[i,j] = mole/deno
        # input_in=torch.matmul(input_in, result.cuda())
        # print("Test time end")
        #############################test1_end##########################################
        # input_out = torch.matmul(A[:, :, A.shape[1]: 2 * A.shape[1]], self.linear_edge_out(hidden)) + self.b_oah
        # input_in = torch.matmul(A[:, :, :A.shape[1]], self.linear_edge_in(hidden)) + self.b_iah
        # input_out = torch.matmul(A[:, :, A.shape[1]: 2 * A.shape[1]], self.linear_edge_out(hidden)) + self.b_oah
        # inputs = torch.cat([input_in, input_out], 2)
        # gi = F.linear(inputs, self.w_ih, self.b_ih)
        # gh = F.linear(hidden, self.w_hh, self.b_hh)
        # i_r, i_i, i_n = gi.chunk(3, 2)
        # h_r, h_i, h_n = gh.chunk(3, 2)
        # resetgate = torch.sigmoid(i_r + h_r)
        # inputgate = torch.sigmoid(i_i + h_i)
        # newgate = torch.tanh(i_n + resetgate * h_n)
        # hy = newgate + inputgate * (hidden - newgate)
        # hy = (1-inputgate)*hidden + inputgate*newgate
        if opt.method == 'activateA':
            input_in = torch.matmul(A[:, :, :A.shape[1]], self.linear_edge_in(hidden)) + self.b_iah
            input_out = torch.matmul(A[:, :, A.shape[1]: 2 * A.shape[1]], self.linear_edge_out(hidden)) + self.b_oah
            inputs = torch.cat([input_in, input_out], 2)
            inputs = torch.relu(inputs)
            gi = F.linear(inputs, self.w_ih, self.b_ih)
            gh = F.linear(hidden, self.w_hh, self.b_hh)
            i_r, i_i, i_n = gi.chunk(3, 2)
            h_r, h_i, h_n = gh.chunk(3, 2)
            resetgate = torch.sigmoid(i_r + h_r)
            inputgate = torch.sigmoid(i_i + h_i)
            newgate = torch.tanh(i_n + resetgate * h_n)
            hy = newgate + inputgate * (hidden - newgate)
            # hy = (1 - inputgate) * hidden + inputgate * newgate
        if opt.method == 'original':
            input_in = torch.matmul(A[:, :, :A.shape[1]], self.linear_edge_in(hidden)) + self.b_iah
            input_out = torch.matmul(A[:, :, A.shape[1]: 2 * A.shape[1]], self.linear_edge_out(hidden)) + self.b_oah
            inputs = torch.cat([input_in, input_out], 2)
            gi = F.linear(inputs, self.w_ih, self.b_ih)
            gh = F.linear(hidden, self.w_hh, self.b_hh)
            i_r, i_i, i_n = gi.chunk(3, 2)
            h_r, h_i, h_n = gh.chunk(3, 2)
            resetgate = torch.sigmoid(i_r + h_r)
            inputgate = torch.sigmoid(i_i + h_i)
            newgate = torch.tanh(i_n + resetgate * h_n)
            hy = newgate + inputgate * (hidden - newgate)
            # hy = (1 - inputgate) * hidden + inputgate * newgate
            hy_pos = hy
            hy_neg = np.array([1])
        if opt.method=='modif_gru_large':
            input_in = torch.matmul(A[:, :, :A.shape[1]], self.linear_edge_in(hidden)) + self.b_iah
            input_out = torch.matmul(A[:, :, A.shape[1]: 2 * A.shape[1]], self.linear_edge_out(hidden)) + self.b_oah
            inputs = torch.cat([input_in, input_out], 2)
            gi = F.linear(inputs, self.w_ih, self.b_ih)
            hy = torch.relu(gi)
        if opt.method=='modif_gru_use_torch':
            input_in = torch.matmul(A[:, :, :A.shape[1]], self.linear_edge_in(hidden)) + self.b_iah
            input_out = torch.matmul(A[:, :, A.shape[1]: 2 * A.shape[1]], self.linear_edge_out(hidden)) + self.b_oah
            inputs = torch.cat([input_in, input_out], 2)
            #input_zize, output_size(input_size is 200,hy is 100)
            gru = torch.nn.GRU(200,100, 1,batch_first=True).cuda()

            h_0 = F.linear(inputs, self.w_ih, self.b_ih)
            h_0 = h_0[:,0]
            h_0 = h_0.reshape(1,inputs.shape[0],100)
            h_0 = h_0.contiguous()
            output,hn = gru(inputs.cuda(), h_0.cuda())
            hy = output

        if opt.method=='modif_gru_use_torch_modif':
            input_in = torch.matmul(A[:, :, :A.shape[1]], self.linear_edge_in(hidden)) + self.b_iah
            input_out = torch.matmul(A[:, :, A.shape[1]: 2 * A.shape[1]], self.linear_edge_out(hidden)) + self.b_oah
            inputs = torch.cat([input_in, input_out], 2)
            #input_zize, output_size(input_size is 200,hy is 100)
            h_0 = F.linear(hidden, self.w_hh, self.b_hh)
            # h_0 = F.linear(inputs, self.w_ih, self.b_ih)
            h_0 = h_0[:,0]
            h_0_0 = torch.cat([h_0, h_0],0)
            h_0 = h_0_0.reshape(2,h_0.shape[0],h_0.shape[1])
            h_0 = h_0.contiguous()
            output,hn = net(inputs.cuda(), h_0.cuda())
            params = [(para[0],para[1].shape) for para in list(net.named_parameters())]
            hy = output
            hy_neg = np.array([1])
            hy = hy_pos
        if opt.method=='modif_rnn_use_torch_modif':
            input_in = torch.matmul(A[:, :, :A.shape[1]], self.linear_edge_in(hidden)) + self.b_iah
            input_out = torch.matmul(A[:, :, A.shape[1]: 2 * A.shape[1]], self.linear_edge_out(hidden)) + self.b_oah
            inputs = torch.cat([input_in, input_out], 2)
            #input_zize, output_size(input_size is 200,hy is 100)
            h_0 = F.linear(hidden, self.w_hh, self.b_hh)
            # h_0 = F.linear(inputs, self.w_ih, self.b_ih)
            h_0 = h_0[:,0]
            h_0_0 = torch.cat([h_0, h_0],0)
            h_0 = h_0_0.reshape(2,h_0.shape[0],h_0.shape[1])
            h_0 = h_0.contiguous()
            output,hn = net(inputs.cuda(), h_0.cuda())

            params = [(para[0],para[1].shape) for para in list(net.named_parameters())]
            hy = output
            hy_neg = np.array([1])
            hy_pos = hy

        if opt.method=='modif_lstm_use_torch_modif':
            input_in = torch.matmul(A[:, :, :A.shape[1]], self.linear_edge_in(hidden)) + self.b_iah
            input_out = torch.matmul(A[:, :, A.shape[1]: 2 * A.shape[1]], self.linear_edge_out(hidden)) + self.b_oah
            inputs = torch.cat([input_in, input_out], 2)
            #input_zize, output_size(input_size is 200,hy is 100)
            h_0 = F.linear(hidden, self.w_hh, self.b_hh)
            # h_0 = F.linear(inputs, self.w_ih, self.b_ih)
            h_0 = h_0[:,0]
            h_0_0 = torch.cat([h_0, h_0],0)
            h_0 = h_0_0.reshape(2,h_0.shape[0],h_0.shape[1])
            h_0 = h_0.contiguous()
            output,hn = net(inputs.cuda(), (h_0.cuda(),h_0.cuda()))
            params = [(para[0],para[1].shape) for para in list(net.named_parameters())]
            hy = output
            hy_neg = np.array([1])
            hy_pos = hy

        if opt.method == 'define_gru':
            input_in = torch.matmul(A[:, :, :A.shape[1]], self.linear_edge_in(hidden)) + self.b_iah
            input_out = torch.matmul(A[:, :, A.shape[1]: 2 * A.shape[1]], self.linear_edge_out(hidden)) + self.b_oah
            inputs = torch.cat([input_in, input_out], 2)
            gi = F.linear(inputs, self.w_ih, self.b_ih)
            gh = F.linear(hidden, self.w_hh, self.b_hh)
            i_r, i_i, i_n = gi.chunk(3, 2)
            h_r, h_i, h_n = gh.chunk(3, 2)
            resetgate = torch.sigmoid(i_r + h_r)
            inputgate = torch.sigmoid(i_i + h_i)
            newgate = torch.tanh(i_n + resetgate * h_n)
            hy = newgate
            # hy = newgate + inputgate * (hidden - newgate)
        if opt.method == 'define_gru2':
            input_in = torch.matmul(A[:, :, :A.shape[1]], self.linear_edge_in(hidden)) + self.b_iah
            input_out = torch.matmul(A[:, :, A.shape[1]: 2 * A.shape[1]], self.linear_edge_out(hidden)) + self.b_oah
            inputs = torch.cat([input_in, input_out], 2)
            w_za = F.linear(inputs, self.w_z, self.wb_z)
            w_ra = F.linear(inputs, self.w_r, self.wb_r)
            w_oa = F.linear(inputs, self.w_o, self.wb_o)
            u_zv = F.linear(hidden, self.u_z, self.ub_z)
            u_rv = F.linear(hidden, self.u_r, self.ub_r)
            zs = torch.sigmoid(w_za + u_zv)
            rs = torch.sigmoid(w_ra + u_rv)
            u_orv = F.linear(rs*hidden, self.u_o, self.ub_o)
            vtb = torch.tanh(w_oa + u_orv)
            hy = (1-zs)*hidden + zs*vtb

        if opt.method == 'define_gru3':
            input_in = torch.matmul(A[:, :, :A.shape[1]], self.linear_edge_in(hidden)) + self.b_iah
            input_out = torch.matmul(A[:, :, A.shape[1]: 2 * A.shape[1]], self.linear_edge_out(hidden)) + self.b_oah
            inputs = torch.cat([input_in, input_out], 2)
            w_za = F.linear(inputs, self.w_z, self.wb_z)
            w_ra = F.linear(inputs, self.w_r, self.wb_r)
            w_oa = F.linear(inputs, self.w_o, self.wb_o)
            u_zv = F.linear(hidden, self.u_z, self.ub_z)
            u_rv = F.linear(hidden, self.u_r, self.ub_r)
            u_ov = F.linear(hidden, self.u_o, self.ub_o)

            zs = torch.sigmoid(w_za + u_zv)
            rs = torch.sigmoid(w_ra + u_rv)
            vtb = torch.tanh(w_oa +rs*u_ov)
            hy = (1-zs)*hidden + zs*vtb

        if opt.method == 'define_gru4':

            input_in = torch.matmul(A[:, :, :A.shape[1]], self.linear_edge_in(hidden)) + self.b_iah
            input_out = torch.matmul(A[:, :, A.shape[1]: 2 * A.shape[1]], self.linear_edge_out(hidden)) + self.b_oah
            inputs = torch.cat([input_in, input_out], 2)
            w_za = F.linear(inputs, self.w_z, self.wb_z)
            w_ra = F.linear(inputs, self.w_r, self.wb_r)
            w_oa = F.linear(inputs, self.w_o, self.wb_o)
            u_zv = F.linear(hidden, self.u_z, self.ub_z)
            u_rv = F.linear(hidden, self.u_r, self.ub_r)
            u_ov = F.linear(hidden, self.u_o, self.ub_o)

            zs = torch.sigmoid(w_za + u_zv)
            rs = torch.sigmoid(w_ra + u_rv)
            vtb = torch.tanh(w_oa +u_ov)
            hy = (1-zs)*hidden + zs*vtb
        if opt.method == 'define_gru5':
            input_in = torch.matmul(A[:, :, :A.shape[1]], self.linear_edge_in(hidden)) + self.b_iah
            input_out = torch.matmul(A[:, :, A.shape[1]: 2 * A.shape[1]], self.linear_edge_out(hidden)) + self.b_oah
            inputs = torch.cat([input_in, input_out], 2)
            w_za = F.linear(inputs, self.w_z, self.wb_z)
            w_oa = F.linear(inputs, self.w_o, self.wb_o)
            u_zv = F.linear(hidden, self.u_z, self.ub_z)
            u_ov = F.linear(hidden, self.u_o, self.ub_o)
            zs = torch.sigmoid(w_za + u_zv)
            vtb = torch.relu(w_oa +u_ov)
            hy = (1-zs)*hidden + zs*vtb
        if opt.method == 'define_gru6':
            input_in = torch.matmul(A[:, :, :A.shape[1]], self.linear_edge_in(hidden)) + self.b_iah
            input_out = torch.matmul(A[:, :, A.shape[1]: 2 * A.shape[1]], self.linear_edge_out(hidden)) + self.b_oah
            inputs = torch.cat([input_in, input_out], 2)
            w_za = F.linear(inputs, self.w_z, self.wb_z)
            w_oa = F.linear(inputs, self.w_o, self.wb_o)
            u_zv = F.linear(hidden, self.u_z, self.ub_z)
            u_ov = F.linear(hidden, self.u_o, self.ub_o)
            zs = torch.relu(w_za + u_zv)
            vtb = torch.relu(w_oa +u_ov)
            hy = (1-zs)*hidden + zs*vtb
        if opt.method == 'define_gru7':
            input_in = torch.matmul(A[:, :, :A.shape[1]], self.linear_edge_in(hidden)) + self.b_iah
            input_out = torch.matmul(A[:, :, A.shape[1]: 2 * A.shape[1]], self.linear_edge_out(hidden)) + self.b_oah
            inputs = torch.cat([input_in, input_out], 2)
            w_za = F.linear(inputs, self.w_z, self.wb_z)
            w_oa = F.linear(inputs, self.w_o, self.wb_o)
            u_zv = F.linear(hidden, self.u_z, self.ub_z)
            u_ov = F.linear(hidden, self.u_o, self.ub_o)
            prelu = torch.nn.PReLU().cuda()
            zs = prelu(w_za + u_zv)
            vtb = prelu(w_oa +u_ov)
            # zs = torch.nn.PReLU(w_za + u_zv)
            # vtb = torch.nn.PReLU(w_oa +u_ov)
            hy = (1-zs)*hidden + zs*vtb
            hy_pos = hy
            hy_neg = np.array([1])
        if opt.method == 'define_gru8':
            input_in = torch.matmul(A[:, :, :A.shape[1]], self.linear_edge_in(hidden)) + self.b_iah
            input_out = torch.matmul(A[:, :, A.shape[1]: 2 * A.shape[1]], self.linear_edge_out(hidden)) + self.b_oah
            inputs = torch.cat([input_in, input_out], 2)
            w_za = F.linear(inputs, self.w_z, self.wb_z)
            w_oa = F.linear(inputs, self.w_o, self.wb_o)
            u_zv = F.linear(hidden, self.u_z, self.ub_z)
            u_ov = F.linear(hidden, self.u_o, self.ub_o)
            prelu = torch.nn.PReLU().cuda()
            zs = prelu(w_za + u_zv)
            vtb = prelu(w_oa +u_ov)
            hy = zs*hidden + zs*vtb
        if opt.method == 'define_gru9':
            input_in = torch.matmul(A[:, :, :A.shape[1]], self.linear_edge_in(hidden)) + self.b_iah
            input_out = torch.matmul(A[:, :, A.shape[1]: 2 * A.shape[1]], self.linear_edge_out(hidden)) + self.b_oah
            inputs = torch.cat([input_in, input_out], 2)
            w_za = F.linear(inputs, self.w_z, self.wb_z)
            w_oa = F.linear(inputs, self.w_o, self.wb_o)
            u_zv = F.linear(hidden, self.u_z, self.ub_z)
            u_ov = F.linear(hidden, self.u_o, self.ub_o)
            prelu = torch.nn.PReLU().cuda()
            zs = prelu(w_za + u_zv)
            vtb = prelu(w_oa +u_ov)
            # zs = torch.nn.PReLU(w_za + u_zv)
            # vtb = torch.nn.PReLU(w_oa +u_ov)
            hy = (1-zs)*hidden + zs*vtb
            #the second layer
            w_za2 = F.linear(inputs, self.w_z2, self.wb_z2)
            w_oa2 = F.linear(inputs, self.w_o2, self.wb_o2)
            u_zv2 = F.linear(hy, self.u_z2, self.ub_z2)
            u_ov2 = F.linear(hy, self.u_o2, self.ub_o2)
            prelu = torch.nn.PReLU().cuda()
            zs2 = prelu(w_za2 + u_zv2)
            vtb2 = prelu(w_oa2 +u_ov2)
            hy = (1 - zs2) * hy + zs2 * vtb2 + hy
            hy_pos = hy
            hy_neg = np.array([1])


        # return hy
        return hy_pos, hy_neg
    # def forward(self, A, hidden):
    def forward(self, A, hidden,gru):
        for i in range(self.step):
            # hidden = self.GNNCell(A, hidden)
            hidden_pos, hidden_neg = self.GNNCell(A, hidden,gru)
            # hidden_pos = self.GNNCell(A, hidden, gru)
        return hidden_pos,hidden_neg
        # return hidden_pos

class SessionGraph(Module):
    def __init__(self, opt, n_node):
        super(SessionGraph, self).__init__()
        self.hidden_size = opt.hiddenSize
        self.n_node = n_node
        self.batch_size = opt.batchSize
        self.nonhybrid = opt.nonhybrid
        self.embedding = nn.Embedding(self.n_node, self.hidden_size)
        self.gnn = GNN(self.hidden_size, step=opt.step)
        self.linear_one = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_two = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_three = nn.Linear(self.hidden_size, 1, bias=False)
        if opt.method_net_last_n1 == 'last_n1_modify2'or  opt.method_net_last_n1 == 'last_n1_modify3' :
            self.linear_transform = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        else:
            self.linear_transform = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=True)
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=opt.lr, weight_decay=opt.l2)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=opt.lr_dc_step, gamma=opt.lr_dc)
        self.net = None
        #input dim is 200, outpudim is 100, the number of layer is 2
        if opt.method=='modif_gru_use_torch_modif':
            self.net = torch.nn.GRU(200, 100, 2, batch_first=True)
        if opt.method=='modif_rnn_use_torch_modif':
            self.net = torch.nn.RNN(200, 100, 2, batch_first=True)
        if opt.method == 'modif_lstm_use_torch_modif':
            self.net = torch.nn.LSTM(200, 100, 2, batch_first=True)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def compute_scores(self, hidden, mask):
        mask_cpu = np.array(mask.cpu())
        sum_torch = torch.sum(mask, 1)
        arange_torch = torch.arange(mask.shape[0]).long(), torch.sum(mask, 1) - 1
        #extracte last click seesion item
        ht = hidden[torch.arange(mask.shape[0]).long(), torch.sum(mask, 1) - 1]  # batch_size x latent_size
        #last items embedding
        q1 = self.linear_one(ht).view(ht.shape[0], 1, ht.shape[1])  # batch_size x 1 x latent_size
        #all items embedding
        q2 = self.linear_two(hidden)  # batch_size x seq_length x latent_size

        alpha = self.linear_three(torch.sigmoid(q1 + q2))
        mask_view = mask.view(mask.shape[0], -1, 1)
        a = torch.sum(alpha * hidden * mask.view(mask.shape[0], -1, 1).float(), 1)

        if opt.method_net_last_n1=='last_n1_modify':
            if not self.nonhybrid:
                a = a+ht
        elif opt.method_net_last_n1 == 'last_n1_modify2':
            if not self.nonhybrid:
                a = self.linear_transform(a+ht)
        elif opt.method_net_last_n1 == 'last_n1_modify3':
            if not self.nonhybrid:
                a = self.linear_transform(a+ht)
                a = torch.sigmoid(a)
        else:
            if not self.nonhybrid:
                a = self.linear_transform(torch.cat([a, ht], 1))
        if opt.method_net_last=='modify':
            b = self.embedding.weight[:]  # n_nodes x latent_size
        else:
            b = self.embedding.weight[1:]  # n_nodes x latent_size
        scores = torch.matmul(a, b.transpose(1, 0))
        return scores

    def mix_scores(self,score_pos, score_neg):
        score = score_pos + score_neg
        return score

    def forward(self, inputs, A):
        inputs_cpu = np.array(inputs.cpu())
        hidden = self.embedding(inputs)
        # hidden = self.gnn(A, hidden)
        # hidden_pos  = self.gnn(A, hidden, self.gru)
        # return hidden_pos
        hidden_pos, hidden_neg = self.gnn(A, hidden,self.net)
        return hidden_pos,hidden_neg

def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.cuda()
    else:
        return variable


def trans_to_cpu(variable):
    if torch.cuda.is_available():
        return variable.cpu()
    else:
        return variable


def forward(model, i, data):
    alias_inputs, A, items, mask, targets = data.get_slice(i)
    alias_inputs = trans_to_cuda(torch.Tensor(alias_inputs).long())
    items = trans_to_cuda(torch.Tensor(items).long())
    A = trans_to_cuda(torch.Tensor(A).float())
    mask = trans_to_cuda(torch.Tensor(mask).long())
    #hidden for gcn return, we get hidden_pos and hidden_neg
    hidden_pos,hidden_neg = model(items, A)
    # hidden_pos = model(items, A)
    #get positive scores
    get_pos = lambda i: hidden_pos[i][alias_inputs[i]]
    seq_hidden_pos = torch.stack([get_pos(i) for i in torch.arange(len(alias_inputs)).long()])
    scores_pos = model.compute_scores(seq_hidden_pos, mask)
    #get negative scores
    scores_neg = None
    if hidden_neg.shape[0]!=1:
        get_neg = lambda i: hidden_neg[i][alias_inputs[i]]
        seq_hidden_neg = torch.stack([get_neg(i) for i in torch.arange(len(alias_inputs)).long()])
        scores_neg = model.compute_scores(seq_hidden_neg, mask)

    return targets,scores_pos,scores_neg

#to slice for train data
def train_test(model, train_data, test_data):
    model.scheduler.step()
    print('start training: ', datetime.datetime.now())
    model.train()
    total_loss = 0.0
    slices = train_data.generate_batch(model.batch_size)
    for i, j in zip(slices, np.arange(len(slices))):
        model.optimizer.zero_grad()
        # if opt.method_addneg_mix == 'addneg_mix':
        targets, scores_pos, scores_neg = forward(model, i, train_data)

        targets = trans_to_cuda(torch.Tensor(targets).long())
        loss_pos = model.loss_function(scores_pos, targets - 1)
        loss_pos.backward()
        model.optimizer.step()
        total_loss += loss_pos
        if j % int(len(slices) / 5 + 1) == 0:
            print('[%d/%d] Loss: %.4f' % (j, len(slices), loss_pos.item()))
            file_path_one = "../logs_loss/" + opt.dataset + 'batch' + str(opt.batchSize) + opt.method + str(
                opt.k) + opt.distance +"all"+ ".txt"
            with open(file_path_one, "a") as f:
                f.write(str(loss_pos.item()))
                f.write(',')

    file_path = "../logs_loss/"+ opt.dataset + 'batch' + str(opt.batchSize) + opt.method + str(opt.k)+ opt.distance+ ".txt"
    with open(file_path,"a") as f:
        f.write(str(total_loss.item()/len(slices)))
        f.write(',')
    print('\tLoss:\t%.3f' % total_loss)

    print('start predicting: ', datetime.datetime.now())
    model.eval()
    hit, mrr = [], []
    slices = test_data.generate_batch(model.batch_size)
    for i in slices:
        targets, scores_pos, scores_neg = forward(model, i, test_data)
        sub_scores = scores_pos.topk(20)[1]
        sub_scores = trans_to_cpu(sub_scores).detach().numpy()
        for score, target, mask in zip(sub_scores, targets, test_data.mask):
            hit.append(np.isin(target - 1, score))
            if len(np.where(score == target - 1)[0]) == 0:
                mrr.append(0)
            else:
                mrr.append(1 / (np.where(score == target - 1)[0][0] + 1))
    hit = np.mean(hit) * 100
    mrr = np.mean(mrr) * 100
    return hit, mrr
