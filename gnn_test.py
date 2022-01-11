import logging
import math
import multiprocessing
import os
import threading
from os import walk
import sys

import networkx
import networkx as nx
from scipy import sparse
import scipy
import numpy as np
import pandas as pd
from sklearn import preprocessing
import torch as th
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
import torch_sparse as th_sp



def gen_graph(graph_type):
    if graph_type == '4_node_line':
        nx_graph = nx.Graph()
        nx_graph.add_edge('A', 'B')
        nx_graph.add_edge('B', 'C')
        nx_graph.add_edge('C', 'D')
        return nx_graph


def gen_graph_sig(sig_type, l_nodes, sig_dim=3):
    if sig_type == 'rand_sig':
        np_sig = np.random.randn(len(l_nodes), sig_dim)
        return np_sig
    elif sig_type == '4_node_line_man':
        np_sig = np.asarray([[1, 1, 1], [2, 3, 4], [5, 2, 5], [6, 6, 1]])
        return np_sig


class GCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(GCNConv, self).__init__(aggr='add')
        self.lin = th.nn.Linear(in_channels, out_channels)

    # def propagate(self, edge_index, g_sig):
    #     print(edge_index)
    #     return g_sig

    # def aggregate(self, inputs, index):
    #     print(inputs, index)
    #     return inputs

    # def message(self, x_j, my_param):
    #     print(x_j)
    #     return x_j

    def update(self, inputs):
        print(inputs)
        return inputs

    def forward(self, g_sig, edge_index):
        # >>> if 'propagate' is not overloaded, then the param 'x' is required without changing its name.
        # >>> and all custom params in 'message, aggregate, update' need to be specified when calling 'propagate'
        return self.propagate(edge_index, x=g_sig, my_param='my_param_val')


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    my_graph = gen_graph(graph_type='4_node_line')
    sp_A = sparse.coo_matrix(nx.linalg.adjacency_matrix(my_graph))
    th_sp_A = th.sparse.FloatTensor(th.LongTensor(np.vstack((sp_A.row, sp_A.col))),
                                    th.FloatTensor(sp_A.data), th.Size(sp_A.shape))
    th_sp_A = th_sp.tensor.from_scipy(sp_A)
    edge_idx = np.asarray([sp_A.row, sp_A.col])
    t_edge_idx = th.LongTensor(edge_idx)
    in_g_sig = gen_graph_sig(sig_type='4_node_line_man', l_nodes=list(my_graph.nodes()))
    t_in_g_sig = th.from_numpy(in_g_sig).type(th.float32)
    my_gcn = GCNConv(3, 3)
    out_g_sig = my_gcn(t_in_g_sig, th_sp_A)
    print(out_g_sig)