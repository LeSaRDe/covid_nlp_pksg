import logging
import math
import sys
import time
from os import walk
import threading
import multiprocessing

import pandas as pd
from scipy import stats
import numpy as np
import networkx as nx

import global_settings


def kstest_for_edges_single_proc(task_id, sample_size):
    logging.debug('[kstest_for_edges_single_proc] Proc %s: starts.' % str(task_id))
    timer_start = time.time()

    df_kstest_task = pd.read_pickle(global_settings.g_kstest_task_file_fmt.format(task_id))
    logging.debug('[kstest_for_edges_single_proc] Proc %s: load in df_kstest_task: %s in %s secs.'
                  % (task_id, len(df_kstest_task), time.time() - timer_start))

    l_ret = []
    cnt = 0
    for _, task in df_kstest_task.iterrows():
        node_1 = task['node_1']
        node_2 = task['node_2']
        node_1_dist = task['node_1_dist']
        node_2_dist = task['node_2_dist']

        l_ks_dim = []
        for dim in range(len(node_1_dist)):
            node_1_dim_mean, node_1_dim_std = node_1_dist[dim]
            node_2_dim_mean, node_2_dim_std = node_2_dist[dim]
            ks_score_1, p_val_1 = stats.kstest(
                stats.norm.rvs(loc=node_1_dim_mean, scale=node_1_dim_std, size=sample_size),
                stats.norm.cdf, args=(node_2_dim_mean, node_2_dim_std))
            ks_score_2, p_val_2 = stats.kstest(
                stats.norm.rvs(loc=node_2_dim_mean, scale=node_2_dim_std, size=sample_size),
                stats.norm.cdf, args=(node_1_dim_mean, node_1_dim_std))
            ks_score_mean = np.mean([ks_score_1, ks_score_2])
            l_ks_dim.append(ks_score_mean)
        l_ret.append((node_1, node_2, l_ks_dim))
        cnt += 1
        if cnt % 5000 == 0 and cnt >= 5000:
            logging.debug('[kstest_for_edges_single_proc] Proc %s: %s tasks done in %s secs.'
                          % (task_id, len(l_ret), time.time() - timer_start))

    logging.debug('[kstest_for_edges_single_proc] Proc %s: all %s tasks done in %s secs.'
                  % (task_id, cnt, time.time() - timer_start))
    df_ret = pd.DataFrame(l_ret, columns=['node_1', 'node_2', 'ks_score'])
    df_ret.to_pickle(global_settings.g_kstest_int_file_fmt.format(task_id))
    logging.debug('[kstest_for_edges_single_proc] Proc %s: all done in %s secs.' % (task_id, time.time() - timer_start))


def kstest_for_edges_multiproc(num_proc, job_id, sample_size, task_id_start=None, task_id_end=None):
    logging.debug('[kstest_for_edges_multiproc] starts.')
    timer_start = time.time()

    if task_id_start is None or task_id_end is None:
        l_task_ids = [str(job_id) + '#' + str(idx) for idx in range(int(num_proc))]
    else:
        l_task_ids = [str(job_id) + '#' + str(idx) for idx in range(int(task_id_start), int(task_id_end))]
    l_proc = []
    logging.debug('[kstest_for_edges_multiproc] l_task_ids = %s' % str(l_task_ids))
    for task_id in l_task_ids:
        p = multiprocessing.Process(target=kstest_for_edges_single_proc,
                                    args=(task_id, sample_size),
                                    name='Proc ' + str(task_id))
        p.start()
        l_proc.append(p)

    while len(l_proc) > 0:
        for p in l_proc:
            if p.is_alive():
                p.join(1)
            else:
                l_proc.remove(p)
                logging.debug('[kstest_for_edges_multiproc] %s is finished.' % p.name)
    logging.debug('[kstest_for_edges_multiproc] All done in %s secs.' % str(time.time() - timer_start))


def gen_kstest_tasks_single_proc(proc_id, num_proc, job_id, edge_id_start, edge_id_end,
                                 np_adj_embed_dist, pksg):
    logging.debug('[gen_kstest_tasks_single_proc] Proc %s: starts with %s edges: (%s, %s).'
                  % (proc_id, edge_id_end - edge_id_start, edge_id_start, edge_id_end))
    timer_start = time.time()

    # np_adj_embed_dist = np.load(global_settings.g_adj_embed_dist_file_fmt.format(ds_name))
    # logging.debug('[gen_kstest_tasks_single_proc] Proc %s: load in np_adj_embed_dist: %s in %s secs.'
    #               % (proc_id, np_adj_embed_dist.shape, time.time() - timer_start))
    #
    # pksg = nx.read_gpickle(global_settings.g_merged_tw_pksg_file_fmt.format(ds_name))
    # logging.debug('[gen_kstest_tasks_single_proc] Proc %s: load in pksg: %s in %s secs.'
    #               % (proc_id, nx.info(pksg), time.time() - timer_start))

    d_node_to_idx = dict()
    l_nodes = list(pksg.nodes)
    for node_idx, node in enumerate(l_nodes):
        d_node_to_idx[node] = node_idx
    l_edges = list(pksg.edges)[edge_id_start : edge_id_end]
    logging.debug('[gen_kstest_tasks_single_proc] Proc %s: takes %s edges.'
                  % (proc_id, len(l_edges)))
    l_nodes = None
    # pksg = None

    l_task = []
    cnt = 0
    for edge in l_edges:
        node_1 = edge[0]
        node_2 = edge[1]
        node_1_idx = d_node_to_idx[node_1]
        node_2_idx = d_node_to_idx[node_2]
        node_1_dist = np_adj_embed_dist[node_1_idx]
        node_2_dist = np_adj_embed_dist[node_2_idx]
        l_task.append((node_1, node_2, node_1_dist, node_2_dist))
        cnt += 1
        if cnt % 10000 == 0 and cnt >= 10000:
            logging.debug('[gen_kstest_tasks_single_proc] Proc %s: %s tasks added in %s secs.'
                          % (proc_id, cnt, time.time() - timer_start))
    logging.debug('[gen_kstest_tasks_single_proc] Proc %s: all %s tasks added in %s secs.'
                  % (proc_id, cnt, time.time() - timer_start))
    df_task = pd.DataFrame(l_task, columns=['node_1', 'node_2', 'node_1_dist', 'node_2_dist'])

    num_proc = int(num_proc)
    num_tasks = len(df_task)
    batch_size = math.ceil(num_tasks / num_proc)
    l_proc = []
    for i in range(0, num_tasks, batch_size):
        if i + batch_size < num_tasks:
            l_proc.append(df_task.iloc[i:i + batch_size])
        else:
            l_proc.append(df_task.iloc[i:])
    logging.debug('[gen_kstest_tasks] Proc %s: Need to generate %s tasks.' % (proc_id, len(l_proc)))

    for task_id, task in enumerate(l_proc):
        task_id = task_id + int(proc_id) * num_proc
        task_name = str(job_id) + '#' + str(task_id)
        pd.to_pickle(task, global_settings.g_kstest_task_file_fmt.format(task_name))
        logging.debug('[gen_kstest_tasks] Proc %s: output %s with %s kstest tasks generated.'
                      % (proc_id, task_name, len(task)))
    logging.debug('[gen_kstest_tasks] Proc %s: All done with %s kstest tasks generated.' % (proc_id, len(l_proc)))


def gen_kstest_tasks_multiproc(num_proc, ds_name, num_task_per_proc, job_id):
    logging.debug('[gen_kstest_tasks_multiproc] starts.')
    timer_start = time.time()

    np_adj_embed_dist = np.load(global_settings.g_adj_embed_dist_file_fmt.format(ds_name))
    logging.debug('[gen_kstest_tasks_multiproc]load in np_adj_embed_dist: %s in %s secs.'
                  % (np_adj_embed_dist.shape, time.time() - timer_start))

    pksg = nx.read_gpickle(global_settings.g_merged_tw_pksg_file_fmt.format(ds_name))
    logging.debug('[gen_kstest_tasks_multiproc] load in pksg: %s in %s secs.'
                  % (nx.info(pksg), time.time() - timer_start))

    num_proc = int(num_proc)
    num_edges = len(pksg.edges)
    batch_size = math.ceil(num_edges / num_proc)
    l_edge_range = []
    for i in range(0, num_edges, batch_size):
        if i + batch_size < num_edges:
            l_edge_range.append((i, i + batch_size))
        else:
            l_edge_range.append((i, num_edges))

    l_proc = []
    for proc_id, edge_range in enumerate(l_edge_range):
        # p = multiprocessing.Process(target=gen_kstest_tasks_single_proc,
        #                             args=(proc_id, num_task_per_proc, ds_name, job_id, edge_range[0], edge_range[1]),
        #                             name='Proc ' + str(proc_id))
        p = threading.Thread(target=gen_kstest_tasks_single_proc,
                                    args=(proc_id, num_task_per_proc, job_id, edge_range[0], edge_range[1],
                                          np_adj_embed_dist, pksg),
                                    name='Proc ' + str(proc_id))
        p.start()
        l_proc.append(p)

    while len(l_proc) > 0:
        for p in l_proc:
            if p.is_alive():
                p.join(1)
            else:
                l_proc.remove(p)
                logging.debug('[gen_kstest_tasks_multiproc] %s is finished.' % p.name)
    logging.debug('[gen_kstest_tasks_multiproc] All done in %s secs.' % str(time.time() - timer_start))


def pure_test():
    df_ks = pd.read_pickle(global_settings.g_kstest_int_file_fmt.format('0#0'))

    d_ks_score = dict()
    cnt = 0
    for _, ks_rec in df_ks.iterrows():
        node_1 = ks_rec['node_1']
        node_2 = ks_rec['node_2']
        ks_score_dim = ks_rec['ks_score']
        if (node_1, node_2) not in d_ks_score:
            d_ks_score[(node_1, node_2)] = [ks_score_dim]
        else:
            d_ks_score[(node_1, node_2)].append(ks_score_dim)
        cnt += 1
        if cnt % 300 == 0 and cnt / 300 >= 1000 and cnt > 300:
            break

    for (node_1, node_2) in d_ks_score:
        ks_score_mean = np.mean(d_ks_score[(node_1, node_2)])
        d_ks_score[(node_1, node_2)] = ks_score_mean

    df_phid_to_phstr = pd.read_pickle(global_settings.g_phrase_id_to_phrase_str_file_fmt.format('202001'))
    for (node_1, node_2) in d_ks_score:
        d_ks_score[(node_1, node_2)] = [df_phid_to_phstr.loc[node_1]['phrase_str'],
                                        df_phid_to_phstr.loc[node_2]['phrase_str'],
                                        d_ks_score[(node_1, node_2)]]

    print()


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    cmd = sys.argv[1]

    if cmd == 'gen_kstest':
        num_proc = sys.argv[2]
        ds_name = sys.argv[3]
        job_id = sys.argv[4]
        num_task_per_proc = int(sys.argv[5])
        gen_kstest_tasks_multiproc(num_proc, ds_name, num_task_per_proc, job_id)
    elif cmd == 'kstest':
        num_proc = sys.argv[2]
        job_id = sys.argv[3]
        sample_size = int(sys.argv[4])
        task_id_start = sys.argv[5]
        if task_id_start == 'None':
            task_id_start = None
        task_id_end = sys.argv[6]
        if task_id_end == 'None':
            task_id_end = None
        kstest_for_edges_multiproc(num_proc, job_id, sample_size, task_id_start, task_id_end)
    elif cmd == 'test':
        # num_proc = sys.argv[2]
        # job_id = sys.argv[3]
        # sample_size = int(sys.argv[4])
        # task_id_start = sys.argv[5]
        # if task_id_start == 'None':
        #     task_id_start = None
        # task_id_end = sys.argv[6]
        # if task_id_end == 'None':
        #     task_id_end = None
        # if task_id_start is None or task_id_end is None:
        #     print('None detected')
        # else:
        #     l_task_ids = [str(job_id) + '#' + str(idx) for idx in range(int(task_id_start), int(task_id_end))]
        #     print(l_task_ids)
        pure_test()
