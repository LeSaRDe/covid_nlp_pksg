import time
import logging

import numpy as np
import numpy.linalg
import networkx as nx
import matplotlib.pyplot as plt


def get_nb_cell_val(nx_digraph, node_from, node_to, node_suc):
    if node_from == node_suc:
        return 0.0
    nb_cell_val = nx_digraph.edges[(node_to, node_suc)]['weight']
    return nb_cell_val


def nx_digraph_to_nb_matrix(nx_digraph):
    timer_start = time.time()
    logging.debug('[nx_digraph_to_nb_matrix] Starts...')
    if not isinstance(nx_digraph, nx.DiGraph):
        raise Exception('[nx_digraph_to_nb_matrix] nx_digraph is not nx.DiGraph!')
    cnt_edges = len(nx_digraph.edges())
    if cnt_edges <= 0:
        return None

    mat_nb = np.zeros((cnt_edges, cnt_edges))
    d_edges = dict()
    for edge in nx_digraph.edges(data=True):
        node_from = edge[0]
        node_to = edge[1]
        if (node_from, node_to) not in d_edges:
            d_edges[(node_from, node_to)] = len(d_edges)
        for node_suc in nx_digraph.successors(node_to):
            if (node_to, node_suc) not in d_edges:
                d_edges[(node_to, node_suc)] = len(d_edges)
            nb_cell_val = get_nb_cell_val(nx_digraph, node_from, node_to, node_suc)
            mat_nb[d_edges[(node_from, node_to)], d_edges[(node_to, node_suc)]] = nb_cell_val
    for row_id in range(mat_nb.shape[0]):
        row_sum = np.sum(mat_nb[row_id])
        if row_sum > 0:
            mat_nb[row_id] = mat_nb[row_id] / row_sum
    if not np.isfinite(mat_nb).all():
        raise Exception('[nx_digraph_to_nb_matrix] mat_nb is not valid: %s' % str(mat_nb))

    logging.debug('[nx_digraph_to_nb_matrix] All done in %s secs.' % str(time.time() - timer_start))
    return mat_nb, d_edges


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    test_digraph = nx.stochastic_block_model([50, 200], [[0.1, 0.8], [0.1, 0.1]], directed=True)

    cnt_core_to_core = 0
    cnt_core_to_shell = 0
    cnt_shell_to_core = 0
    cnt_shell_to_shell = 0
    s_cores = test_digraph.graph['partition'][0]
    s_shells = test_digraph.graph['partition'][1]
    for edge in test_digraph.edges(data=True):
        node_from = edge[0]
        node_to = edge[1]
        if node_from in s_cores:
            if node_to in s_cores:
                cnt_core_to_core += 1
            else:
                cnt_core_to_shell += 1
        else:
            if node_to in s_cores:
                cnt_shell_to_core += 1
            else:
                cnt_shell_to_shell += 1

    cnt_edges = len(test_digraph.edges())
    print('partition: %s' % str(test_digraph.graph['partition']))
    print('cnt_edges = %s' % cnt_edges)
    print('cnt_core_to_core = %s, %s' % (cnt_core_to_core, cnt_core_to_core / cnt_edges))
    print('cnt_core_to_shell = %s, %s' % (cnt_core_to_shell, cnt_core_to_shell / cnt_edges))
    print('cnt_shell_to_core = %s, %s' % (cnt_shell_to_core, cnt_shell_to_core / cnt_edges))
    print('cnt_shell_to_shell = %s, %s' % (cnt_shell_to_shell, cnt_shell_to_shell / cnt_edges))

    test_digraph = nx.stochastic_graph(test_digraph)

    mat_nb, d_edges = nx_digraph_to_nb_matrix(test_digraph)
    l_eigenvals = np.linalg.eigvals(mat_nb)
    l_eigenval_pairs = []
    for eigenval in l_eigenvals:
        l_eigenval_pairs.append((np.real(eigenval), np.imag(eigenval)))
    plt.scatter([item[0] for item in l_eigenval_pairs], [item[1] for item in l_eigenval_pairs], c='b')
    plt.show()
    print()