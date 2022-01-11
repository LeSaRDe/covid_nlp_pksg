import json
import logging
import math
import multiprocessing
import sys
import time
from os import walk

import networkx as nx
import pandas as pd
import numpy as np
from scipy.spatial.distance import cosine
import scipy.sparse as sp
from scipy.sparse import csr_matrix
from sklearn.cluster import KMeans

import global_settings


def merge_phrase_cluster(d_cluster_1, d_cluster_2, d_pe, cid_prefix, new_id_start):
    '''
    'd_cluster_1' and 'd_cluster_2' are unmergeable.
    Returns an unmergeble cluster and new_id_start for new potential cluster id.
    '''
    timer_start = time.time()
    l_rm_1 = []
    l_rm_2 = []
    rm_add_cnt_1 = 0
    rm_add_cnt_2 = 0
    d_new_cluster = dict()
    for cid_1 in d_cluster_1:
        embed_1 = d_cluster_1[cid_1][1]
        for cid_2 in d_cluster_2:
            embed_2 = d_cluster_2[cid_2][1]
            sim = 1.0 - cosine(embed_1, embed_2)
            if sim >= global_settings.g_phrase_sim_threshold:
                new_cid = cid_prefix + str(new_id_start)
                new_id_start += 1
                l_pid_for_new_cluster = d_cluster_1[cid_1][0] + d_cluster_2[cid_2][0]
                embed_for_new_cluster = np.mean([d_pe[pid] for pid in l_pid_for_new_cluster], axis=0)
                d_new_cluster[new_cid] = (l_pid_for_new_cluster, embed_for_new_cluster)
                l_rm_1.append(cid_1)
                l_rm_2.append(cid_2)
                rm_add_cnt_1 += 1
                if rm_add_cnt_1 % 1000 == 0 and rm_add_cnt_1 >= 1000:
                    l_rm_1 = list(set(l_rm_1))
                rm_add_cnt_2 += 1
                if rm_add_cnt_2 % 1000 == 0 and rm_add_cnt_2 >= 1000:
                    l_rm_2 = list(set(l_rm_2))

    if len(d_new_cluster) <= 0:
        for cid in d_cluster_2:
            if cid in d_cluster_1:
                raise Exception('[merge_phrase_cluster] Something wrong happened: %s, %s'
                                % (d_cluster_2[cid], cid_prefix))
            d_cluster_1[cid] = d_cluster_2[cid]
        logging.debug('[merge_phrase_cluster] cid_prefix = %s, new_id_start = %s, in %s secs'
                      % (cid_prefix, new_id_start, time.time() - timer_start))
        return d_cluster_1, new_id_start
    else:
        for cid in l_rm_1:
            del d_cluster_1[cid]
        for cid in l_rm_2:
            del d_cluster_2[cid]
        for cid in d_cluster_2:
            if cid in d_cluster_1:
                raise Exception('[merge_phrase_cluster] Something wrong happened: %s, %s'
                                % (d_cluster_2[cid], cid_prefix))
            d_cluster_1[cid] = d_cluster_2[cid]
        d_new_cluster, new_id_start = hierarchical_phrase_clustering(d_new_cluster, d_pe, cid_prefix, new_id_start)
        return merge_phrase_cluster(d_cluster_1, d_new_cluster, d_pe, cid_prefix, new_id_start)


def hierarchical_phrase_clustering(d_cluster, d_pe, cid_prefix, new_id_start):
    '''
    Returns an unmergeble cluster and new_id_start for new potential cluster id.
    '''
    if len(d_cluster) <= 0:
        return None
    timer_start = time.time()

    l_rm = []
    rm_add_cnt = 0
    d_new_cluster = dict()
    l_cluster_id = list(d_cluster.keys())
    for i in range(len(l_cluster_id) - 1):
        cid_i = l_cluster_id[i]
        embed_i = d_cluster[cid_i][1]
        for j in range(i + 1, len(l_cluster_id)):
            cid_j = l_cluster_id[j]
            embed_j = d_cluster[cid_j][1]
            sim = 1.0 - cosine(embed_i, embed_j)
            if sim >= global_settings.g_phrase_sim_threshold:
                new_cid = cid_prefix + str(new_id_start)
                new_id_start += 1
                l_pid_for_new_cluster = d_cluster[cid_i][0] + d_cluster[cid_j][0]
                embed_for_new_cluster = np.mean([d_pe[pid] for pid in l_pid_for_new_cluster], axis=0)
                d_new_cluster[new_cid] = (l_pid_for_new_cluster, embed_for_new_cluster)
                l_rm += [cid_i, cid_j]
                rm_add_cnt += 2
                if rm_add_cnt % 1000 == 0 and rm_add_cnt >= 1000:
                    l_rm = list(set(l_rm))
                    rm_add_cnt = 0
    if len(d_new_cluster) <= 0:
        logging.debug('[hierarchical_phrase_clustering] cid_prefix = %s, new_id_start = %s, in %s secs'
                      % (cid_prefix, new_id_start, time.time() - timer_start))
        return d_cluster, new_id_start
    else:
        d_new_cluster, new_id_start = hierarchical_phrase_clustering(d_new_cluster, d_pe, cid_prefix, new_id_start)
        if d_new_cluster is None or len(d_new_cluster) <= 0:
            raise Exception('[hierarchical_phrase_clustering] Something wrong happened: %s, %s, %s'
                            % (d_cluster, cid_prefix, new_id_start))
        for cid in l_rm:
            del d_cluster[cid]
        return merge_phrase_cluster(d_cluster, d_new_cluster, d_pe, cid_prefix, new_id_start)


def phrase_clustering_single_proc(task_id):
    logging.debug('[phrase_clustering_single_proc] Proc %s: Starts.' % str(task_id))
    timer_start = time.time()

    df_phrase_embed = pd.read_pickle(global_settings.g_phrase_cluster_task_file_fmt.format(task_id))
    logging.debug('[phrase_clustering_single_proc] Proc %s: Load in %s phrase embed recs.'
                  % (task_id, len(df_phrase_embed)))

    d_pe = dict()
    for pid, phrase_embed_rec in df_phrase_embed.iterrows():
        phrase_embed = np.asarray(phrase_embed_rec['phrase_embed'], dtype=np.float32)
        d_pe[pid] = phrase_embed

    d_cluster = dict()
    cid_prefix = task_id + '#'
    for idx, pid in enumerate(d_pe):
        cid = cid_prefix + str(idx)
        d_cluster[cid] = ([pid], d_pe[pid])
    new_id_start = len(d_pe)

    d_ret, _ = hierarchical_phrase_clustering(d_cluster, d_pe, cid_prefix, new_id_start)
    l_pc_rec = []
    l_pid_to_cid = []
    for cid in d_ret:
        l_pid = d_ret[cid][0]
        c_embed = d_ret[cid][1]
        l_pc_rec.append((cid, l_pid, c_embed))
        for pid in l_pid:
            l_pid_to_cid.append((pid, cid))
    df_pc = pd.DataFrame(l_pc_rec, columns=['pc_id', 'l_pid', 'pc_embed'])
    pd.to_pickle(df_pc, global_settings.g_phrase_cluster_int_file_fmt.format(task_id))
    df_pid_to_cid = pd.DataFrame(l_pid_to_cid, columns=['phrase_id', 'pc_id'])
    pd.to_pickle(df_pid_to_cid, global_settings.g_phrase_cluster_pid_to_cid_int_file_fmt.format(task_id))
    logging.debug('[phrase_clustering_single_proc] Proc %s: All done with %s phrase clusters '
                  'and %s pid to cid mappings in %s sec.'
                  % (task_id, len(df_pc), len(df_pid_to_cid), time.time() - timer_start))


def phrase_clustering_multiproc(num_proc, job_id):
    logging.debug('[phrase_clustering_multiproc] Starts.')
    timer_start = time.time()

    l_task_ids = [str(job_id) + '#' + str(idx) for idx in range(int(num_proc))]
    l_proc = []
    for task_id in l_task_ids:
        p = multiprocessing.Process(target=phrase_clustering_single_proc,
                                    args=(task_id,),
                                    name='Proc ' + str(task_id))
        p.start()
        l_proc.append(p)

    while len(l_proc) > 0:
        for p in l_proc:
            if p.is_alive():
                p.join(1)
            else:
                l_proc.remove(p)
                logging.debug('[phrase_clustering_multiproc] %s is finished.' % p.name)
    logging.debug('[phrase_clustering_multiproc] All done in %s secs.' % str(time.time() - timer_start))


def gen_phrase_clustering_tasks(ds_name, num_tasks, job_id):
    logging.debug('[gen_phrase_clustering_tasks] Starts.')
    timer_start = time.time()

    df_phrase_embed = pd.read_pickle(global_settings.g_phrase_embed_file_fmt.format(ds_name))
    num_phrase_embed_rec = len(df_phrase_embed)
    logging.debug('[gen_phrase_clustering_tasks] Load in %s recs.' % str(num_phrase_embed_rec))

    num_tasks = int(num_tasks)
    batch_size = math.ceil(num_phrase_embed_rec / num_tasks)
    l_tasks = []
    for i in range(0, num_phrase_embed_rec, batch_size):
        if i + batch_size < num_phrase_embed_rec:
            l_tasks.append(df_phrase_embed.iloc[i:i + batch_size])
        else:
            l_tasks.append(df_phrase_embed.iloc[i:])
    logging.debug('[gen_phrase_clustering_tasks] Need to generate %s tasks.' % str(len(l_tasks)))

    for idx, df_task in enumerate(l_tasks):
        df_task = df_task[['phrase_embed']]
        pd.to_pickle(df_task, global_settings.g_phrase_cluster_task_file_fmt.format(str(job_id) + '#' + str(idx)))
    logging.debug('[gen_phrase_clustering_tasks] All done with %s tasks in %s secs.'
                  % (len(l_tasks), time.time() - timer_start))


def gen_phrase_sim_job(ds_name, num_jobs):
    logging.debug('[gen_phrase_sim_job] Starts.')

    df_phrase_embed = pd.read_pickle(global_settings.g_phrase_embed_file_fmt.format(ds_name))
    logging.debug('[gen_phrase_sim_job] Load in %s phrase embeds.' % str(len(df_phrase_embed)))
    d_row_id_to_phrase_id = dict()
    for row_id, phrase_id in enumerate(df_phrase_embed.index):
        d_row_id_to_phrase_id[row_id] = phrase_id
    with open(global_settings.g_phrase_sim_row_id_to_phrase_id_file_fmt.format(ds_name), 'w+') as out_fd:
        json.dump(d_row_id_to_phrase_id, out_fd)
        out_fd.close()
    logging.debug('[gen_phrase_sim_job] Row id to phrase id done with %s rows.' % str(len(d_row_id_to_phrase_id)))

    num_phrase_embed = len(df_phrase_embed)
    batch_size = 10000
    l_iterations = []
    for i in range(0, num_phrase_embed, batch_size):
        if i + batch_size < num_phrase_embed:
            l_iterations.append((i, i + batch_size))
        else:
            l_iterations.append((i, num_phrase_embed))
    logging.debug('[gen_phrase_sim_job] %s iterations for computing sims.' % str(len(l_iterations)))

    num_iteration = len(l_iterations)
    batch_size = math.ceil(num_iteration / int(num_jobs))
    l_jobs = []
    for i in range(0, num_iteration, batch_size):
        if i + batch_size < num_iteration:
            l_jobs.append(l_iterations[i:i + batch_size])
        else:
            l_jobs.append(l_iterations[i:num_iteration])

    for job_id, job in enumerate(l_jobs):
        job_str = '\n'.join(str(item[0]) + '|' + str(item[1]) for item in job)
        with open(global_settings.g_phrase_sim_task_file_fmt.format(str(job_id)), 'w+') as out_fd:
            out_fd.write(job_str)
            out_fd.close()
    logging.debug('[gen_phrase_sim_job] All done with %s jobs.' % str(len(l_jobs)))


# def phrase_sim_single_job(ds_name, job_id):
#     logging.debug('[phrase_sim_single_job] Job %s: Starts.' % str(job_id))
#     timer_start = time.time()
#
#     l_task = []
#     with open(global_settings.g_phrase_sim_task_file_fmt.format(str(job_id)), 'r') as in_fd:
#         for ln in in_fd:
#             l_fields = ln.split('|')
#             start = int(l_fields[0].strip())
#             end = int(l_fields[1].strip())
#             l_task.append((start, end))
#         in_fd.close()
#     logging.debug('[phrase_sim_single_job] Job %s: Read in %s tasks %s.' % (job_id, len(l_task), l_task))
#
#     df_phrase_embed = pd.read_pickle(global_settings.g_phrase_embed_file_fmt.format(ds_name))
#     logging.debug('[phrase_sim_single_job] Job %s: Load in %s phrase embeds.' % (job_id, len(df_phrase_embed)))
#
#     with open(global_settings.g_phrase_sim_row_id_to_phrase_id_file_fmt.format(ds_name), 'r') as in_fd:
#         d_row_id_to_phrase_id = json.load(in_fd)
#         in_fd.close()
#     logging.debug('[phrase_sim_single_job] Job %s: Load in row id to phrase id mapping with %s rows.'
#                   % (job_id, len(d_row_id_to_phrase_id)))
#     if len(d_row_id_to_phrase_id) != len(df_phrase_embed):
#         raise Exception('[phrase_sim_single_job] Job %s: d_row_id_to_phrase_id mismatches with df_phrase_embed!'
#                         % str(job_id))
#
#     l_ready = []
#     phrase_id_pair_cnt = 0
#     np_phrase_embed = np.stack(df_phrase_embed['phrase_embed'].to_list())
#     for task_int in l_task:
#         l_sim_pair = []
#         start = task_int[0]
#         end = task_int[1]
#         np_phrase_sim = np.matmul(np_phrase_embed[start:end], np.transpose(np_phrase_embed))
#         if not np.isfinite(np_phrase_sim).all():
#             raise Exception('[phrase_sim_single_job] Job %s: Invalid np_phrase_sim!' % str(job_id))
#         logging.debug('[phrase_sim_single_job] Job %s: np_phrase_sim done for [%s, %s] in %s secs.'
#                       % (job_id, start, end, time.time() - timer_start))
#         tup_indices = np.nonzero(np_phrase_sim >= global_settings.g_phrase_sim_threshold)
#         for i in range(len(tup_indices[0])):
#             row_id_1 = start + tup_indices[0][i]
#             row_id_2 = tup_indices[1][i]
#             if row_id_2 <= row_id_1:
#                 continue
#             phrase_id_1 = d_row_id_to_phrase_id[str(row_id_1)]
#             phrase_id_2 = d_row_id_to_phrase_id[str(row_id_2)]
#             l_sim_pair.append([phrase_id_1, phrase_id_2])
#         l_ready.append((start, end, l_sim_pair))
#         phrase_id_pair_cnt += len(l_sim_pair)
#         logging.debug('[phrase_sim_single_job] Job %s: [%s, %s] done with %s phrase id pairs, %s in total, in %s secs.'
#                       % (job_id, start, end, len(l_sim_pair), phrase_id_pair_cnt, time.time() - timer_start))
#
#     df_ready = pd.DataFrame(l_ready, columns=['start', 'end', 'l_phrase_id_pair'])
#     pd.to_pickle(df_ready, global_settings.g_phrase_sim_int_file_fmt.format(str(job_id)))
#     logging.debug('[phrase_sim_single_job] Job %s: All done with %s phrase id pairs in %s secs.'
#                   % (job_id, phrase_id_pair_cnt, time.time() - timer_start))


# def build_phrase_sim_graph_single_proc(task_id):
#     logging.debug('[build_phrase_sim_graph_single_proc] Proc %s: Starts.' % str(task_id))
#     timer_start = time.time()
#
#     df_phrase_sim = pd.read_pickle(global_settings.g_phrase_sim_int_file_fmt.format(str(task_id)))
#     logging.debug('[build_phrase_sim_graph_single_proc] Proc %s: Load in %s phrase sim recs.'
#                   % (task_id, len(df_phrase_sim)))
#
#     phrase_sim_graph = nx.Graph()
#     for _, phrase_sim_rec in df_phrase_sim.iterrows():
#         l_phrase_id_pair = phrase_sim_rec['l_phrase_id_pair']
#         for phrase_id_pair in l_phrase_id_pair:
#             phrase_id_1 = phrase_id_pair[0]
#             phrase_id_2 = phrase_id_pair[1]
#             phrase_sim_graph.add_edge(phrase_id_1, phrase_id_2)
#             edge_cnt = len(phrase_sim_graph.edges())
#             if edge_cnt % 10000 == 0 and edge_cnt >= 10000:
#                 logging.debug('[build_phrase_sim_graph_single_proc] Proc %s: Add %s edges and %s nodes in %s secs.'
#                               % (task_id, edge_cnt, len(phrase_sim_graph.nodes()), time.time() - timer_start))
#     logging.debug('[build_phrase_sim_graph_single_proc] Proc %s: Add %s edges and %s nodes in %s secs.'
#                   % (task_id, len(phrase_sim_graph.edges()), len(phrase_sim_graph.nodes()), time.time() - timer_start))
#
#     with open(global_settings.g_phrase_sim_graph_int_file_fmt.format(str(task_id)), 'w+') as out_fd:
#         phrase_sim_graph_json = nx.adjacency_data(phrase_sim_graph)
#         json.dump(phrase_sim_graph_json, out_fd)
#         out_fd.close()
#     logging.debug('[build_phrase_sim_graph_single_proc] Proc %s: All done in %s secs.'
#                   % (task_id, time.time() - timer_start))
#
#
# def build_phrase_sim_graph_multiproc(num_tasks):
#     logging.debug('[build_phrase_sim_graph_multiproc] Starts.')
#     timer_start = time.time()
#
#     l_task_ids = [str(idx) for idx in range(int(num_tasks))]
#     l_proc = []
#     for task_id in l_task_ids:
#         p = multiprocessing.Process(target=build_phrase_sim_graph_single_proc,
#                                     args=(task_id,),
#                                     name='Proc ' + str(task_id))
#         p.start()
#         l_proc.append(p)
#
#     while len(l_proc) > 0:
#         for p in l_proc:
#             if p.is_alive():
#                 p.join(1)
#             else:
#                 l_proc.remove(p)
#                 logging.debug('[build_phrase_sim_graph_multiproc] %s is finished.' % p.name)
#     logging.debug('[build_phrase_sim_graph_multiproc] All done in %s secs.' % str(time.time() - timer_start))


def phrase_sim_graph_adj_single_job(ds_name, job_id):
    logging.debug('[phrase_sim_graph_adj_single_job] Job %s: Starts.' % str(job_id))
    timer_start = time.time()

    l_task = []
    with open(global_settings.g_phrase_sim_task_file_fmt.format(str(job_id)), 'r') as in_fd:
        for ln in in_fd:
            l_fields = ln.split('|')
            start = int(l_fields[0].strip())
            end = int(l_fields[1].strip())
            l_task.append((start, end))
        in_fd.close()
    logging.debug('[phrase_sim_graph_adj_single_job] Job %s: Read in %s tasks %s.' % (job_id, len(l_task), l_task))

    df_phrase_embed = pd.read_pickle(global_settings.g_phrase_embed_file_fmt.format(ds_name))
    logging.debug('[phrase_sim_graph_adj_single_job] Job %s: Load in %s phrase embeds.' % (job_id, len(df_phrase_embed)))

    with open(global_settings.g_phrase_sim_row_id_to_phrase_id_file_fmt.format(ds_name), 'r') as in_fd:
        d_row_id_to_phrase_id = json.load(in_fd)
        in_fd.close()
    logging.debug('[phrase_sim_graph_adj_single_job] Job %s: Load in row id to phrase id mapping with %s rows.'
                  % (job_id, len(d_row_id_to_phrase_id)))
    if len(d_row_id_to_phrase_id) != len(df_phrase_embed):
        raise Exception('[phrase_sim_graph_adj_single_job] Job %s: d_row_id_to_phrase_id mismatches with df_phrase_embed!'
                        % str(job_id))

    sp_adj_sum = None
    np_phrase_embed = np.stack(df_phrase_embed['phrase_embed'].to_list())
    for task_int in l_task:
        start = task_int[0]
        end = task_int[1]
        np_phrase_sim = np.matmul(np_phrase_embed[start:end], np.transpose(np_phrase_embed))
        if not np.isfinite(np_phrase_sim).all():
            raise Exception('[phrase_sim_graph_adj_single_job] Job %s: Invalid np_phrase_sim!' % str(job_id))
        logging.debug('[phrase_sim_graph_adj_single_job] Job %s: np_phrase_sim done for [%s, %s] in %s secs.'
                      % (job_id, start, end, time.time() - timer_start))
        tup_indices = np.nonzero(np_phrase_sim >= global_settings.g_phrase_sim_threshold)
        row_indices = start + tup_indices[0]
        col_indices = tup_indices[1]
        sp_adj = csr_matrix(([1] * len(row_indices), (row_indices, col_indices)),
                            shape=(np_phrase_sim.shape[1], np_phrase_sim.shape[1]),
                            dtype=np.int8)
        sp_adj = sp.triu(sp_adj, k=1, format='csr')
        sp_adj += sp_adj.transpose()
        if sp_adj_sum is None:
            sp_adj_sum = sp_adj
        else:
            sp_adj_sum += sp_adj
        logging.debug('[phrase_sim_graph_adj_single_job] Job %s: Add [%s, %s] to adjacency in %s secs.'
                      % (job_id, start, end, time.time() - timer_start))

    sp.save_npz(global_settings.g_phrase_sim_graph_adj_int_file_fmt.format(str(job_id)), sp_adj_sum)
    logging.debug('[phrase_sim_graph_adj_single_job] Job %s: All done in %s secs.'
                  % (job_id, time.time() - timer_start))


def merge_phrase_sim_graph_adj_int(ds_name):
    logging.debug('[merge_phrase_sim_graph_adj_int] Starts.')
    timer_start = time.time()

    sp_adj_sum = None
    for (dirpath, dirname, filenames) in walk(global_settings.g_phrase_cluster_int_folder):
        for filename in filenames:
            if filename[-4:] != '.npz' or filename[:25] != 'phrase_sim_graph_adj_int_':
                continue
            sp_adj = sp.load_npz(dirpath + filename)
            if sp_adj_sum is None:
                sp_adj_sum = sp_adj
            else:
                sp_adj_sum += sp_adj
            logging.debug('[merge_phrase_sim_graph_adj_int] Merged in %s in %s secs.'
                          % (filename, time.time() - timer_start))
    sp.save_npz(global_settings.g_phrase_sim_graph_adj_file_fmt.format(ds_name), sp_adj_sum)
    logging.debug('[merge_phrase_sim_graph_adj_int] All done in %s secs.' % str(time.time() - timer_start))


def find_effective_subgraph(trg_graph, trg_deg):
    '''
    trg_graph must be connected.
    number of nodes in trg_graph must be >= trg_degree + 1.
    Return a list of subgraphs. In each, any node has a degree >= trg_deg.
    '''
    trg_k = trg_deg + 1
    l_eff_nodes = [node for node in trg_graph.nodes() if trg_graph.degree(node) >= trg_deg]
    cur_eff_nodes_cnt = len(l_eff_nodes)
    if cur_eff_nodes_cnt < trg_k:
        return []

    cand_eff_graph = nx.subgraph(trg_graph, l_eff_nodes)
    l_eff_subgraphs = []
    for comp in nx.connected_components(cand_eff_graph):
        sub_trg_graph = nx.subgraph(cand_eff_graph, comp)
        l_eff_nodes = [node for node in sub_trg_graph.nodes() if sub_trg_graph.degree(node) >= trg_deg]
        if len(l_eff_nodes) == cur_eff_nodes_cnt:
            return [cand_eff_graph]
        # TODO
        # use queue to make this loop instead of recursion
        l_eff_subgraphs += find_effective_subgraph(sub_trg_graph, trg_deg)


def phrase_sim_phrase_clique_removal_for_connected_graph(trg_graph, start_clique_idx):
    l_nodes_by_deg = sorted(trg_graph.nodes(), key=lambda k: trg_graph.degree(k), reverse=True)
    if len(l_nodes_by_deg) <= 0:
        logging.error('[phrase_sim_phrase_clique_removal_for_connected_graph] l_nodes_by_deg is invalid.')
        return None

    d_cliques = dict()
    l_cand_nodes = [l_nodes_by_deg[0]]
    # target K of cliques = trg_deg + 1
    trg_deg = trg_graph.degree(l_nodes_by_deg[0])
    if len(l_nodes_by_deg) <= 1:
        if trg_deg == 1:
            d_cliques[0] = [l_nodes_by_deg[0]]
            return d_cliques
        else:
            raise Exception('[phrase_sim_phrase_clique_removal_for_connected_graph] '
                            'l_nodes_by_deg does not match trg_graph: %s, %s' % (l_nodes_by_deg, nx.info(trg_graph)))

    clique_idx = 1
    cur_node_idx = 1
    while cur_node_idx < len(l_nodes_by_deg):
        trg_k = trg_deg + 1
        cur_node = l_nodes_by_deg[cur_node_idx]
        cur_node_deg = trg_graph.degree(cur_node)
        # add all nodes of degrees >= trg_deg into as candidates
        if cur_node_deg >= trg_deg:
            l_cand_nodes.append(cur_node)
            cur_node_idx += 1
            continue
        # when candidates are not enough for a size trg_k clique, then there must be no size trg_k clique.
        # then add in more nodes with lower degrees and lower down trg_k.
        if len(l_cand_nodes) < trg_k:
            l_cand_nodes.append(cur_node)
            trg_deg = trg_graph.degree(cur_node)
            cur_node_idx += 1
            continue
        # up to here, potentially there exists at least one trg_k clique in the candidates.
        # check if candidates' effective degrees are all >= trg_deg
        cand_graph = nx.subgraph(trg_graph, l_cand_nodes)
        for comp in nx.connected_components(cand_graph):
            sub_cand_graph = nx.subgraph(cand_graph, comp)
            if len(sub_cand_graph) < trg_k:
                continue
        # TODO
        # not finished yet



def phrase_sim_graph_clique_removal(ds_name):
    '''
    Takes a phrase_sim_graph as input, and output a list of maximal cliques.
    The output cliques are not overlapped. Doing this for approximation. When a maximal clique is found, it is output,
    and its member nodes are then removed from the graph. The rest of the finding is based on the leftover graph.
    '''
    logging.debug('[phrase_sim_graph_clique_removal] Starts.')
    timer_start = time.time()

    phrase_sim_graph_adj = sp.load_npz(global_settings.g_phrase_sim_graph_adj_file_fmt.format(ds_name))
    logging.debug('[phrase_sim_graph_clique_removal] Load in phrase_sim_graph_adj in %s secs.'
                  % str(time.time() - timer_start))

    phrase_sim_graph = nx.from_scipy_sparse_matrix(phrase_sim_graph_adj)
    logging.debug('[phrase_sim_graph_clique_removal] Reconstruct phrase_sim_graph in %s secs: %s'
                  % (time.time() - timer_start, nx.info(phrase_sim_graph)))

    d_cliques = dict()
    clique_idx = 0
    for comp_idx, comp in enumerate(nx.connected_components(phrase_sim_graph)):
        if len(comp) <= 2:
            d_cliques[clique_idx] = list(comp)
            clique_idx += 1
            logging.debug('[phrase_sim_graph_clique_removal] Done clique #%s with %s nodes in %s secs.'
                          % (clique_idx, len(d_cliques[clique_idx]), time.time() - timer_start))
            continue
        sub_phrase_sim_graph = nx.subgraph(phrase_sim_graph, comp)
        phrase_sim_phrase_clique_removal_for_connected_graph(sub_phrase_sim_graph, clique_idx)


# TODO
# Do K-means with an initial cluster size which allows K-means to finish in a reasonable running time.
# So this initial cluster size shouldn't be large. Then firstly for each cluster we kick out points that are distant
# from the center by a threshold (e.g. 0.8), and recompute the center. Secondly, we classify every node to each cluster
# deterministically by the same threshold, and a node can be classified to multiple clusters. Theoretically, the above
# two steps can be conducted iteratively like what K-means does, but it may not be necessary for the sake of complexity.
# Finally, the nodes that are not classified to any clusters are forced to form single-element clusters.
def phrase_kmeans(init_cluster_size):
    logging.debug('[phrase_kmeans] Starts.')
    timer_start = time.time()





# This functions doesn't work. It will run out of memory.
def phrase_sim_graph_maximal_cliques(ds_name):
    logging.debug('[phrase_sim_graph_maximal_cliques] Starts.')
    timer_start = time.time()

    phrase_sim_graph_adj = sp.load_npz(global_settings.g_phrase_sim_graph_adj_file_fmt.format(ds_name))
    logging.debug('[phrase_sim_graph_maximal_cliques] Load in phrase_sim_graph_adj in %s secs.'
                  % str(time.time() - timer_start))

    phrase_sim_graph = nx.from_scipy_sparse_matrix(phrase_sim_graph_adj)
    logging.debug('[phrase_sim_graph_maximal_cliques] Reconstruct phrase_sim_graph in %s secs: %s'
                  % (time.time() - timer_start, nx.info(phrase_sim_graph)))

    with open(global_settings.g_phrase_row_id_to_phrase_id_file_fmt.format(ds_name), 'r') as in_fd:
        d_phrase_row_id_to_phrase_id = json.load(in_fd)
        in_fd.close()
    logging.debug('[phrase_sim_graph_maximal_cliques] Load in d_phrase_row_id_to_phrase_id with %s recs.'
                  % str(len(d_phrase_row_id_to_phrase_id)))

    l_cliques = list(nx.find_cliques(phrase_sim_graph))
    l_ready = [('pc#' + str(l_cliques.index(clique)), [d_phrase_row_id_to_phrase_id[pid] for pid in clique])
               for clique in l_cliques]
    df_clique = pd.DataFrame(l_ready, columns=['phrase_cluster_id', 'l_phrase_id'])
    df_clique = df_clique.set_index('phrase_cluster_id')
    pd.to_pickle(df_clique, global_settings.g_phrase_cluster_file_fmt.format(ds_name))
    logging.debug('[phrase_sim_graph_maximal_cliques] All done with %s phrase clusters in %s secs.'
                  % (len(df_clique), time.time() - timer_start))


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    cmd = sys.argv[1]

    if cmd == 'gen_phrase_cluster_tasks':
        ds_name = sys.argv[2]
        num_tasks = sys.argv[3]
        job_id = sys.argv[4]
        gen_phrase_clustering_tasks(ds_name, num_tasks, job_id)
    elif cmd == 'phrase_cluster':
        num_proc = sys.argv[2]
        job_id = sys.argv[3]
        phrase_clustering_multiproc(num_proc, job_id)
    elif cmd == 'gen_phrase_sim_jobs':
        ds_name = sys.argv[2]
        num_jobs = sys.argv[3]
        gen_phrase_sim_job(ds_name, num_jobs)
    # elif cmd == 'phrase_sim':
    #     # CAUTION:
    #     # This step needs to be done in multiple nodes, each running in a single process,
    #     # as the memory consumption is very considerable.
    #     ds_name = sys.argv[2]
    #     job_id = sys.argv[3]
    #     phrase_sim_single_job(ds_name, job_id)
    # elif cmd == 'phrase_sim_graph':
    #     num_tasks = sys.argv[2]
    #     build_phrase_sim_graph_multiproc(num_tasks)
    elif cmd == 'phrase_sim_graph_adj':
        # NOTE:
        # 'job_id' here must be consistent with the number of jobs generated in 'gen_phrase_sim_jobs'
        ds_name = sys.argv[2]
        job_id = sys.argv[3]
        phrase_sim_graph_adj_single_job(ds_name, job_id)
    elif cmd == 'merge_phrase_sim_graph_adj':
        ds_name = sys.argv[2]
        merge_phrase_sim_graph_adj_int(ds_name)
    elif cmd == 'phrase_clique':
        ds_name = sys.argv[2]
        phrase_sim_graph_maximal_cliques(ds_name)
    elif cmd == 'test':
        # with open(global_settings.g_phrase_cluster_int_folder + 'phrase_sim_graph_int_18.pickle', 'r') as in_fd:
        #     phrase_sim_graph_json = json.load(in_fd)
        #     in_fd.close()
        # phrase_sim_graph = nx.adjacency_graph(phrase_sim_graph_json)
        # from networkx.algorithms.community.kclique import k_clique_communities
        from networkx.algorithms.approximation.clique import clique_removal
        # timer_start = time.time()
        # l_phrase_cluster = list(k_clique_communities(phrase_sim_graph, k=3))
        # indep_set, l_clique = clique_removal(phrase_sim_graph)
        # l_node_by_degree = sorted(list(phrase_sim_graph.degree), key=lambda k: k[1], reverse=True)
        # max_degree = l_node_by_degree[0][1]
        # l_comp = list(nx.connected_components(phrase_sim_graph))
        # timer_elapse = time.time() - timer_start
        # print(timer_elapse)
        # breakpoint()
        # print(l_comp)
        # print(l_clique)
        ds_name = sys.argv[2]
        job_id = sys.argv[3]
        # test_phrase_sim_single_job(ds_name, job_id)