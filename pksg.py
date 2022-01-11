import json
import logging
import math
import multiprocessing
import threading
import sys
import time
import traceback
from os import walk

import networkx as nx
from networkx.algorithms.approximation.clique import clique_removal
from networkx.drawing.nx_agraph import graphviz_layout
import pandas as pd
import numpy as np
from sklearn import preprocessing
import scipy.sparse as sp
import matplotlib.pyplot as plt

import global_settings
from semantic_units_extractor import SemUnitsExtractor
from sentiment_analysis import compute_sentiment_for_one_phrase_in_one_tw, sentiment_calibration

sys.path.insert(1, global_settings.g_lexvec_model_folder)
import model as lexvec


# PKSG
# Node:
#   Node label (string): phrase id, e.g. 'ph#123'.
#   Node attributes:
#       'str' (string): sorted lemmas linked by single spaces, e.g. 'mask wear'.
#       'pos' (dict): sorted POS keywords linked by single spaces with the number of occurrences as its value,
#                   e.g. {'NOUN VERB': 2, 'NOUN NOUN': 1}.
#                   Note that the POS tags may not aligned to the lemmas in 'str' in order as both of them have
#                   been sorted.
# Edge:
#   Edge label (tuple of strings): tuple of phrase ids, e.g. ('ph#123', 'ph#456').
#   Edge attributes:
#       'sent' (list of of lists of reals): each element is a list of reals representing a sentiment vector of
#                   of the format: (very negative, negative, neutral, positive, very positive). An edge may
#                   occur multiple times (in one tweet or multiple tweets), which makes this attrinbute a list.


def build_pksg_for_one_tw(task_id, tw_id, df_tw_phrase, df_tw_to_phid, df_phid_to_phstr, df_tw_sent, df_tw_sgraph,
                          sim_phid=None):
    '''
    if 'sim_phid' is not specified, then construct a clique upon all phrases of the tweet given by 'tw_id'.
    Otherwise, construct a star graph centered at 'sim_phrase_id' which links to every other phrase of the tweet.
    '''
    l_phid_phpos_in_tw = [item for item in df_tw_to_phid.loc[tw_id]['l_phrase_id']]
    l_phid_in_tw = [item[0] for item in l_phid_phpos_in_tw]
    l_tw_phtup = df_tw_phrase.loc[tw_id]['tw_phrase']
    l_sgraph_info = df_tw_sgraph.loc[tw_id]['tw_sgraph']
    l_parsed_sgraph_info = []
    for sgraph_info in l_sgraph_info:
        sgraph_json_str = sgraph_info[0]
        sgraph_start = sgraph_info[1]
        sgraph_end = sgraph_info[2]
        sgraph = nx.adjacency_graph(json.loads(sgraph_json_str))
        l_parsed_sgraph_info.append((sgraph, sgraph_start, sgraph_end))
    l_tw_phsent = df_tw_sent.loc[tw_id]['tw_phrase_sentiment']

    def edge_sentiment_aggregation(ph_pstn_in_tw_i, ph_pstn_in_tw_j):
        phsent_i = l_tw_phsent[ph_pstn_in_tw_i][3]
        phsent_j = l_tw_phsent[ph_pstn_in_tw_j][3]
        if phsent_i is None or phsent_j is None:
            return None
        _, edge_sent = sentiment_calibration(phsent_i, phsent_j)
        return edge_sent

    def add_ph_j_into_pksg(ph_pstn_in_tw_j, ph_pstn_in_tw_i, phid_i, phstart_i, phend_i, tw_pksg_graph):
        ph_tup_j = l_tw_phtup[ph_pstn_in_tw_j]
        phid_j, phpos_j = l_phid_phpos_in_tw[ph_pstn_in_tw_j]
        phspan_j = ph_tup_j[2]
        phstart_j = min([item[0] for item in phspan_j])
        phend_j = max([item[1] for item in phspan_j])

        if phid_i == phid_j:
            return tw_pksg_graph

        edge_phstart = min(phstart_i, phstart_j)
        edge_phend = max(phend_i, phend_j)
        edge_phspan = [(phstart_i, phend_i), (phstart_j, phend_j)]
        edge_phsent = compute_sentiment_for_one_phrase_in_one_tw(l_parsed_sgraph_info, edge_phstart,
                                                                 edge_phend, edge_phspan)
        if edge_phsent is None:
            edge_phsent = edge_sentiment_aggregation(ph_pstn_in_tw_i, ph_pstn_in_tw_j)
        if not tw_pksg_graph.has_edge(phid_i, phid_j):
            if edge_phsent is not None:
                tw_pksg_graph.add_edge(phid_i, phid_j, sent=[edge_phsent], weight=1)
            else:
                tw_pksg_graph.add_edge(phid_i, phid_j, sent=[], weight=1)
        else:
            if edge_phsent is not None:
                tw_pksg_graph.edges()[phid_i, phid_j]['sent'].append(edge_phsent)
                tw_pksg_graph.edges()[phid_i, phid_j]['weight'] += 1
        return tw_pksg_graph

    tw_pksg_graph = nx.Graph()
    # since a phrase id may occur multiple times in a tw, we add nodes and edges separately.

    # add nodes
    for ph_pstn_in_tw in range(len(l_tw_phtup)):
        phid, phpos = l_phid_phpos_in_tw[ph_pstn_in_tw]
        phstr_i = df_phid_to_phstr.loc[phid]['phrase_str']
        if not tw_pksg_graph.has_node(phid):
            tw_pksg_graph.add_node(phid, str=phstr_i, pos={phpos: 1})
        else:
            if phpos not in tw_pksg_graph.nodes(data=True)[phid]['pos']:
                tw_pksg_graph.nodes(data=True)[phid]['pos'][phpos] = 1
            else:
                tw_pksg_graph.nodes(data=True)[phid]['pos'][phpos] += 1

    # add edges
    if sim_phid is None:
        for ph_pstn_in_tw_i in range(len(l_tw_phtup) - 1):
            ph_tup_i = l_tw_phtup[ph_pstn_in_tw_i]
            phid_i, phpos_i = l_phid_phpos_in_tw[ph_pstn_in_tw_i]
            phspan_i = ph_tup_i[2]
            phstart_i = min([item[0] for item in phspan_i])
            phend_i = max([item[1] for item in phspan_i])

            for ph_pstn_in_tw_j in range(ph_pstn_in_tw_i + 1, len(l_tw_phtup)):
                tw_pksg_graph = add_ph_j_into_pksg(ph_pstn_in_tw_j, ph_pstn_in_tw_i, phid_i, phstart_i,
                                                   phend_i, tw_pksg_graph)
    else:
        if sim_phid not in l_phid_in_tw:
            raise Exception('[build_pksg_for_one_tw] Task %s: %s is not in %s.' % (task_id, sim_phid, tw_id))
        sim_ph_pstn_in_tw = l_phid_in_tw.index(sim_phid)
        sim_phtup = l_tw_phtup[sim_ph_pstn_in_tw]
        _, sim_phpos = l_phid_phpos_in_tw[sim_ph_pstn_in_tw]
        sim_phspan = sim_phtup[2]
        sim_phstart = min([item[0] for item in sim_phspan])
        sim_phend = max([item[1] for item in sim_phspan])

        for ph_pstn_in_tw_j in range(len(l_tw_phtup)):
            tw_pksg_graph = add_ph_j_into_pksg(ph_pstn_in_tw_j, sim_ph_pstn_in_tw, sim_phid, sim_phstart,
                                               sim_phend, tw_pksg_graph)

    return tw_pksg_graph


def build_tw_pksg_single_proc(task_id, ds_name, sim_phid=None):
    logging.debug('[build_pksg_single_proc] Proc %s: starts.' % str(task_id))
    timer_start = time.time()

    l_tw_id = []
    with open(global_settings.g_tw_pksg_task_file_fmt.format(str(task_id)), 'r') as in_fd:
        for ln in in_fd:
            l_tw_id.append(ln.strip())
        in_fd.close()
    logging.debug('[build_pksg_single_proc] Proc %s: load in %s tw ids for tasks.' % (task_id, len(l_tw_id)))

    df_tw_phrase = pd.read_pickle(global_settings.g_tw_phrase_file_fmt.format(ds_name))
    logging.debug('[build_pksg_single_proc] Proc %s: load in df_tw_phrase with %s recs in %s secs.'
                  % (task_id, len(df_tw_phrase), time.time() - timer_start))
    df_tw_to_phid = pd.read_pickle(global_settings.g_tw_to_phrase_id_file_fmt.format(ds_name))
    logging.debug('[build_pksg_single_proc] Proc %s: load in df_tw_to_phid with %s recs in %s secs.'
                  % (task_id, len(df_tw_to_phid), time.time() - timer_start))
    df_phid_to_phstr = pd.read_pickle(global_settings.g_phrase_id_to_phrase_str_file_fmt.format(ds_name))
    logging.debug('[build_pksg_single_proc] Proc %s: load in df_phid_to_phstr with %s recs in %s secs.'
                  % (task_id, len(df_phid_to_phstr), time.time() - timer_start))
    df_tw_sent = pd.read_pickle(global_settings.g_tw_phrase_sentiment_file_fmt.format(ds_name))
    df_phid_to_phstr = pd.read_pickle(global_settings.g_phrase_id_to_phrase_str_file_fmt.format(ds_name))
    logging.debug('[build_pksg_single_proc] Proc %s: load in df_tw_sent with %s recs in %s secs.'
                  % (task_id, len(df_tw_sent), time.time() - timer_start))
    df_tw_sgraph = pd.read_pickle(global_settings.g_tw_sgraph_file_fmt.format(ds_name))
    logging.debug('[build_pksg_single_proc] Proc %s: load in df_tw_sgraph with %s recs in %s secs.'
                  % (task_id, len(df_tw_sgraph), time.time() - timer_start))

    l_tw_pksg_rec = []
    for tw_id in l_tw_id:
        tw_pksg = build_pksg_for_one_tw(task_id, tw_id, df_tw_phrase, df_tw_to_phid, df_phid_to_phstr, df_tw_sent,
                                        df_tw_sgraph, sim_phid)
        l_tw_pksg_rec.append((tw_id, tw_pksg))
        if len(l_tw_pksg_rec) % 500 == 0 and len(l_tw_pksg_rec) >= 500:
            logging.debug('[build_pksg_single_proc] Proc %s: %s tw pksgs done in %s secs.'
                          % (task_id, len(l_tw_pksg_rec), time.time() - timer_start))
    logging.debug('[build_pksg_single_proc] Proc %s: %s tw pksgs done in %s secs.'
                  % (task_id, len(l_tw_pksg_rec), time.time() - timer_start))

    df_tw_pksg = pd.DataFrame(l_tw_pksg_rec, columns=['tw_id', 'tw_pksg'])
    df_tw_pksg = df_tw_pksg.set_index('tw_id')
    df_tw_pksg.to_pickle(global_settings.g_tw_pksg_int_file_fmt.format(task_id))
    logging.debug('[build_pksg_single_proc] Proc %s: all done with %s tw pksgs in %s secs.'
                  % (task_id, len(df_tw_pksg), time.time() - timer_start))


def build_tw_pksg_multiproc(num_proc, ds_name, job_id):
    logging.debug('[build_tw_pksg_multiproc] Starts.')
    timer_start = time.time()

    l_task_ids = [str(job_id) + '#' + str(idx) for idx in range(int(num_proc))]
    l_proc = []
    for task_id in l_task_ids:
        p = multiprocessing.Process(target=build_tw_pksg_single_proc,
                                    args=(task_id, ds_name),
                                    name='Proc ' + str(task_id))
        p.start()
        l_proc.append(p)

    while len(l_proc) > 0:
        for p in l_proc:
            if p.is_alive():
                p.join(1)
            else:
                l_proc.remove(p)
                logging.debug('[build_tw_pksg_multiproc] %s is finished.' % p.name)
    logging.debug('[build_tw_pksg_multiproc] All done in %s secs.' % str(time.time() - timer_start))


def gen_tw_pksg_tasks(num_proc, ds_name, job_id):
    logging.debug('[gen_tw_pksg_tasks] Starts.')
    timer_start = time.time()

    df_tw_phrase = pd.read_pickle(global_settings.g_tw_phrase_file_fmt.format(ds_name))
    logging.debug('[gen_tw_pksg_tasks] load in df_tw_phrase with %s recs in %s secs.' %
                  (len(df_tw_phrase), time.time() - timer_start))

    l_tw_id = df_tw_phrase.index.to_list()
    num_tw_id = len(l_tw_id)
    num_tasks = int(num_proc)
    batch_size = math.ceil(num_tw_id / num_tasks)
    l_tasks = []
    for i in range(0, num_tw_id, batch_size):
        if i + batch_size < num_tw_id:
            l_tasks.append(l_tw_id[i:i + batch_size])
        else:
            l_tasks.append(l_tw_id[i:])
    logging.debug('[gen_tw_pksg_tasks] Need to generate %s tasks.' % str(len(l_tasks)))

    for task_id, task in enumerate(l_tasks):
        task_name = str(job_id) + '#' + str(task_id)
        with open(global_settings.g_tw_pksg_task_file_fmt.format(task_name), 'w+') as out_fd:
            out_str = '\n'.join([item.strip() for item in task])
            out_fd.write(out_str)
            out_fd.close()
    logging.debug('[gen_tw_pksg_tasks] All done with %s tw pksg tasks generated.' % str(len(l_tasks)))


def merge_two_tw_pksg(tw_pksg_1, tw_pksg_2):
    if tw_pksg_1.number_of_nodes() == 0 or tw_pksg_2.number_of_nodes() == 0:
        raise Exception('[merge_two_tw_pksg] empty pksg occurs.')

    for node_1 in tw_pksg_1.nodes(data=True):
        if not tw_pksg_2.has_node(node_1):
            tw_pksg_2.add_node(node_1[0], pos=node_1[1]['pos'])
        else:
            for pos_1 in node_1[1]['pos']:
                if pos_1 not in tw_pksg_2.nodes(data=True)[node_1[0]]:
                    tw_pksg_2.nodes(data=True)[node_1[0]][pos_1] = node_1[1]['pos'][pos_1]
                else:
                    tw_pksg_2.nodes(data=True)[node_1[0]][pos_1] += node_1[1]['pos'][pos_1]

    for edge_1 in tw_pksg_1.edges(data=True):
        if not tw_pksg_2.has_edge(edge_1[0], edge_1[1]):
            tw_pksg_2.add_edge(edge_1[0], edge_1[1], sent=edge_1[2]['sent'], weight=edge_1[2]['weight'])
        else:
            tw_pksg_2.edges()[edge_1[0], edge_1[1]]['sent'] += edge_1[2]['sent']
            tw_pksg_2.edges()[edge_1[0], edge_1[1]]['weight'] += edge_1[2]['weight']

    return tw_pksg_2


def divide_and_conquer_merge_tw_pksg(l_tw_pksg):
    if len(l_tw_pksg) < 1:
        raise Exception('[divide_and_conquer_merge_tw_pksg] invalid l_tw_pksg')
    if len(l_tw_pksg) == 1:
        return l_tw_pksg[0]
    if len(l_tw_pksg) == 2:
        return merge_two_tw_pksg(l_tw_pksg[0], l_tw_pksg[1])

    batch_size = math.ceil(len(l_tw_pksg) / 2)
    ret_graph = merge_two_tw_pksg(divide_and_conquer_merge_tw_pksg(l_tw_pksg[:batch_size]),
                                  divide_and_conquer_merge_tw_pksg(l_tw_pksg[batch_size:]))
    return ret_graph


# def merge_tw_pksg(ds_name):
#     logging.debug('[merge_tw_pksg] starts.')
#     timer_start = time.time()
#
#     l_tw_pksg = []
#     for (dirpath, dirname, filenames) in walk(global_settings.g_ks_graph_int_folder):
#         for filename in filenames:
#             if filename[-7:] != '.pickle' or filename[:12] != 'tw_pksg_int_':
#                 continue
#             df_int = pd.read_pickle(dirpath + filename)
#             l_tw_pksg += df_int['tw_pksg'].to_list()
#             logging.debug('[merge_tw_pksg] read in %s tw pksg.' % str(len(df_int)))
#     logging.debug('[merge_tw_pksg] all %s tw pksg loaded in %s secs.' % (len(l_tw_pksg), time.time() - timer_start))
#
#     merged_pksg = divide_and_conquer_merge_tw_pksg(l_tw_pksg)
#     logging.debug('[merge_tw_pksg] done merging in %s secs: %s' % (time.time() - timer_start, nx.info(merged_pksg)))
#
#     nx.write_gpickle(merged_pksg, global_settings.g_merged_tw_pksg_file_fmt.format(ds_name))
#     logging.debug('[merge_tw_pksg] all done in %s secs.' % str(time.time() - timer_start))


def merge_tw_pksg_single_proc(task_id):
    logging.debug('[merge_tw_pksg_single_proc] Proc %s: starts.' % str(task_id))
    timer_start = time.time()

    df_tw_pksg_task = pd.read_pickle(global_settings.g_merged_tw_pksg_task_file_fmt.format(task_id))
    logging.debug('[merge_tw_pksg_single_proc] Proc %s: load in %s tw pksg in %s secs.'
                  % (task_id, len(df_tw_pksg_task), time.time() - timer_start))

    l_tw_pksg = df_tw_pksg_task['tw_pksg'].to_list()
    batch_size = 10
    cnt = 0
    while True:
        l_tasks = []
        num_tasks = len(l_tw_pksg)
        for i in range(0, num_tasks, batch_size):
            if i + batch_size < num_tasks:
                l_tasks.append(l_tw_pksg[i:i + batch_size])
            else:
                l_tasks.append(l_tw_pksg[i:])

        l_rets = []
        for task in l_tasks:
            merged_tw_pksg = divide_and_conquer_merge_tw_pksg(task)
            l_rets.append(merged_tw_pksg)

        if len(l_rets) <= 0:
            raise Exception('[merge_tw_pksg_single_proc] Proc %s: invalid l_rets')
        elif len(l_rets) == 1:
            cnt += 1
            nx.write_gpickle(l_rets[0], global_settings.g_merged_tw_pksg_int_file_fmt.format(task_id))
            logging.debug('[merge_tw_pksg_single_proc] Proc %s: All done in %s secs: %s'
                          % (task_id, time.time() - timer_start, nx.info(l_rets[0])))
            return
        else:
            cnt += 1
            logging.debug('[merge_tw_pksg_single_proc] Proc %s: %s interation done in %s secs. %s tw pksg left to merge.'
                          % (task_id, cnt, time.time() - timer_start, len(l_rets)))
            l_tw_pksg = l_rets


def merge_tw_pksg_multiproc(num_proc, ds_name, job_id):
    logging.debug('[merge_tw_pksg_multiproc] Starts.')
    timer_start = time.time()

    l_task_ids = [str(job_id) + '#' + str(idx) for idx in range(int(num_proc))]
    l_proc = []
    for task_id in l_task_ids:
        p = multiprocessing.Process(target=merge_tw_pksg_single_proc,
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
                logging.debug('[merge_tw_pksg_multiproc] %s is finished.' % p.name)

    l_merged_tw_pksg_int = []
    for (dirpath, dirname, filenames) in walk(global_settings.g_ks_graph_int_folder):
        for filename in filenames:
            if filename[-7:] != '.pickle' or filename[:19] != 'merged_tw_pksg_int_':
                continue
            merge_tw_pksg_int = nx.read_gpickle(dirpath + filename)
            l_merged_tw_pksg_int.append(merge_tw_pksg_int)
    merged_tw_pksg = divide_and_conquer_merge_tw_pksg(l_merged_tw_pksg_int)
    nx.write_gpickle(merged_tw_pksg, global_settings.g_merged_tw_pksg_file_fmt.format(ds_name))
    logging.debug('[merge_tw_pksg_multiproc] All done in %s secs.' % str(time.time() - timer_start))


def gen_merged_tw_pksg_tasks(num_proc, job_id):
    logging.debug('[gen_merged_tw_pksg_tasks] starts.')
    timer_start = time.time()

    l_tw_pksg = []
    for (dirpath, dirname, filenames) in walk(global_settings.g_ks_graph_int_folder):
        for filename in filenames:
            if filename[-7:] != '.pickle' or filename[:12] != 'tw_pksg_int_':
                continue
            df_int = pd.read_pickle(dirpath + filename)
            l_tw_pksg += df_int['tw_pksg'].to_list()
            logging.debug('[gen_merged_tw_pksg_tasks] read in %s tw pksg.' % str(len(df_int)))
    logging.debug('[gen_merged_tw_pksg_tasks] all %s tw pksg loaded in %s secs.'
                  % (len(l_tw_pksg), time.time() - timer_start))

    l_tw_pksg = [(item,) for item in l_tw_pksg]
    num_tw_pksg = len(l_tw_pksg)
    num_tasks = int(num_proc)
    batch_size = math.ceil(num_tw_pksg / num_tasks)
    l_tasks = []
    for i in range(0, num_tw_pksg, batch_size):
        if i + batch_size < num_tw_pksg:
            l_tasks.append(l_tw_pksg[i:i + batch_size])
        else:
            l_tasks.append(l_tw_pksg[i:])
    logging.debug('[gen_merged_tw_pksg_tasks] Need to generate %s tasks.' % str(len(l_tasks)))

    for task_id, task in enumerate(l_tasks):
        task_name = str(job_id) + '#' + str(task_id)
        df_task = pd.DataFrame(task, columns=['tw_pksg'])
        pd.to_pickle(df_task, global_settings.g_merged_tw_pksg_task_file_fmt.format(task_name))
    logging.debug('[gen_merged_tw_pksg_tasks] All done with %s tw pksg tasks generated.' % str(len(l_tasks)))


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    cmd = sys.argv[1]

    if cmd == 'gen_tw_pksg_tasks':
        num_proc = sys.argv[2]
        ds_name = sys.argv[3]
        job_id = sys.argv[4]
        gen_tw_pksg_tasks(num_proc, ds_name, job_id)
    elif cmd == 'tw_pksg':
        num_proc = sys.argv[2]
        ds_name = sys.argv[3]
        job_id = sys.argv[4]
        build_tw_pksg_multiproc(num_proc, ds_name, job_id)
    elif cmd == 'gen_merged_tw_pksg_tasks':
        num_proc = sys.argv[2]
        job_id = sys.argv[3]
        gen_merged_tw_pksg_tasks(num_proc, job_id)
    elif cmd == 'merge_tw_pksg':
        num_proc = sys.argv[2]
        ds_name = sys.argv[3]
        job_id = sys.argv[4]
        merge_tw_pksg_multiproc(num_proc, ds_name, job_id)
    # elif cmd == 'merge_tw_pksg':
    #     ds_name = sys.argv[2]
    #     merge_tw_pksg(ds_name)
    elif cmd == 'test':
        ds_name = '202001'
        task_id = 'test'
        tw_id = '1221853678988079106'
        df_tw_phrase = pd.read_pickle(global_settings.g_tw_phrase_file_fmt.format(ds_name))
        df_tw_to_phid = pd.read_pickle(global_settings.g_tw_to_phrase_id_file_fmt.format(ds_name))
        df_phid_to_phstr = pd.read_pickle(global_settings.g_phrase_id_to_phrase_str_file_fmt.format(ds_name))
        df_tw_sent = pd.read_pickle(global_settings.g_tw_phrase_sentiment_file_fmt.format(ds_name))
        df_tw_sgraph = pd.read_pickle(global_settings.g_tw_sgraph_file_fmt.format(ds_name))
        tw_pksg_graph = build_pksg_for_one_tw(task_id, tw_id, df_tw_phrase, df_tw_to_phid, df_phid_to_phstr,
                                              df_tw_sent, df_tw_sgraph)
        print(nx.info(tw_pksg_graph))
        print(tw_pksg_graph.nodes(data=True))
        print(tw_pksg_graph.edges(data=True))