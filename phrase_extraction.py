import logging
import traceback
from os import walk
import time
import json
import math
import multiprocessing
import sys
import functools
import operator

import pandas as pd
import networkx as nx

import global_settings

"""
Each tweet is associated with a list of phrases if any. 
Each phrase is of the form: 
([token_str_1, token_str_2], [POS_1, POS_2], [(token_1_start, token_1_end), (token_2_start, token_2_end)])
or
([token_str], [POS], [(token_start, token_end)])
where 'token_i_start' and 'token_i_end' are indexed relative to the tweet text as a whole.
Specifically, a tweet text is determined by the tweet cleaned text. 
"""


def extract_phrases_from_cls_json_str(cls_json_str):
    if cls_json_str is None:
        return None
    s_covered_nodes = []
    l_phrases = []
    try:
        cls_json = json.loads(cls_json_str)
        cls_graph = nx.adjacency_graph(cls_json)
        for edge in cls_graph.edges(data=True):
            node_1 = edge[0]
            node_2 = edge[1]
            node_1_txt = cls_graph.nodes(data=True)[node_1]['txt']
            node_1_pos = cls_graph.nodes(data=True)[node_1]['pos']
            node_1_start = cls_graph.nodes(data=True)[node_1]['start']
            node_1_end = cls_graph.nodes(data=True)[node_1]['end']
            node_2_txt = cls_graph.nodes(data=True)[node_2]['txt']
            node_2_pos = cls_graph.nodes(data=True)[node_2]['pos']
            node_2_start = cls_graph.nodes(data=True)[node_2]['start']
            node_2_end = cls_graph.nodes(data=True)[node_2]['end']
            # phrase_start = min(node_1_start, node_2_start)
            # phrase_end = max(node_1_end, node_2_end)
            phrase = ([node_1_txt, node_2_txt], [node_1_pos, node_2_pos],
                      [(node_1_start, node_1_end), (node_2_start, node_2_end)])
            s_covered_nodes.append(node_1)
            s_covered_nodes.append(node_2)
            l_phrases.append(phrase)
        s_covered_nodes = set(s_covered_nodes)
        if len(s_covered_nodes) < len(cls_graph.nodes):
            for node in cls_graph.nodes(data=True):
                if node[0] not in s_covered_nodes:
                    node_txt = node[1]['txt']
                    node_pos = node[1]['pos']
                    node_start = node[1]['start']
                    node_end = node[1]['end']
                    phrase = ([node_txt], [node_pos], [(node_start, node_end)])
                    l_phrases.append(phrase)
    except Exception as err:
        print('[extract_phrases_from_cls_json_str] %s' % err)
        traceback.print_exc()
    if len(l_phrases) > 0:
        return l_phrases
    return None


def extract_phrase_from_nps_str(nps):
    if nps is None or len(nps) <= 0:
        return None
    nps = [([noun_phrase[0]], ['NOUN'], [(noun_phrase[2], noun_phrase[3])]) for noun_phrase in nps]
    return nps


def tw_phrase_extraction_single_proc(task_id):
    logging.debug('[tw_phrase_extraction_single_proc] Proc %s: Starts.' % str(task_id))
    timer_start = time.time()

    df_tw_sem_unit = pd.read_pickle(global_settings.g_tw_phrase_task_file_fmt.format(task_id))
    logging.debug('[tw_phrase_extraction_single_proc] Proc %s: Load in %s sem unit recs.'
                  % (task_id, len(df_tw_sem_unit)))

    l_ready = []
    for tw_id, tw_sem_unit_rec in df_tw_sem_unit.iterrows():
        cls_json_str = tw_sem_unit_rec['cls_json_str']
        nps = tw_sem_unit_rec['nps']
        l_ready_phrase = []
        l_cls_phrase = extract_phrases_from_cls_json_str(cls_json_str)
        if l_cls_phrase is not None:
            l_ready_phrase += l_cls_phrase
        l_nps_phrase = extract_phrase_from_nps_str(nps)
        if l_nps_phrase is not None:
            l_ready_phrase += l_nps_phrase
        if len(l_ready_phrase) > 0:
            l_ready.append((tw_id, l_ready_phrase))
    df_phrase = pd.DataFrame(l_ready, columns=['tw_id', 'tw_phrase'])
    pd.to_pickle(df_phrase, global_settings.g_tw_phrase_int_file_fmt.format(task_id))
    logging.debug('[tw_phrase_extraction_single_proc] Proc %s: All done with %s tw phrase recs in %s secs.'
                  % (task_id, len(df_phrase), time.time() - timer_start))


def tw_phrase_extraction_multiproc(num_proc, job_id):
    logging.debug('[tw_phrase_extraction_multiproc] Starts.')
    timer_start = time.time()

    l_task_ids = [str(job_id) + '#' + str(idx) for idx in range(int(num_proc))]
    l_proc = []
    for task_id in l_task_ids:
        p = multiprocessing.Process(target=tw_phrase_extraction_single_proc,
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
                logging.debug('[tw_phrase_extraction_multiproc] %s is finished.' % p.name)
    logging.debug('[tw_phrase_extraction_multiproc] All done in %s secs.' % str(time.time() - timer_start))


def gen_tw_phrase_extraction_tasks(ds_name, num_tasks, job_id):
    logging.debug('[gen_tw_phrase_extraction_tasks] Starts.')
    timer_start = time.time()

    df_tw_sem_unit = pd.read_pickle(global_settings.g_tw_sem_unit_file_fmt.format(ds_name))
    num_sem_unit = len(df_tw_sem_unit)
    logging.debug('[gen_tw_phrase_extraction_tasks] Load in %s recs.' % str(num_sem_unit))

    num_tasks = int(num_tasks)
    batch_size = math.ceil(num_sem_unit / num_tasks)
    l_tasks = []
    for i in range(0, num_sem_unit, batch_size):
        if i + batch_size < num_sem_unit:
            l_tasks.append(df_tw_sem_unit.iloc[i:i + batch_size])
        else:
            l_tasks.append(df_tw_sem_unit.iloc[i:])
    logging.debug('[gen_tw_phrase_extraction_tasks] Need to generate %s tasks.' % str(len(l_tasks)))

    for idx, df_task in enumerate(l_tasks):
        pd.to_pickle(df_task, global_settings.g_tw_phrase_task_file_fmt.format(str(job_id) + '#' + str(idx)))
    logging.debug('[gen_tw_phrase_extraction_tasks] All done with %s tasks in %s secs.'
                  % (len(l_tasks), time.time() - timer_start))


def merge_tw_phrase_int(ds_name):
    logging.debug('[merge_tw_phrase_int]')
    timer_start = time.time()

    l_int = []
    for (dirpath, dirname, filenames) in walk(global_settings.g_tw_phrase_int_folder):
        for filename in filenames:
            if filename[-7:] != '.pickle' or filename[:14] != 'tw_phrase_int_':
                continue
            df_int = pd.read_pickle(dirpath + filename)
            l_int.append(df_int)
    logging.debug('[merge_tw_phrase_int] Read in %s int dfs.' % str(len(l_int)))
    df_merge = pd.concat(l_int)
    df_merge = df_merge.set_index('tw_id')
    pd.to_pickle(df_merge, global_settings.g_tw_phrase_file_fmt.format(ds_name))
    logging.debug('[merge_tw_phrase_int] All done with %s recs in %s secs.'
                  % (len(df_merge), time.time() - timer_start))


def map_between_tw_and_phrase(ds_name):
    logging.debug('[map_between_tw_and_phrase] Starts.')
    timer_start = time.time()

    df_tw_phrase = pd.read_pickle(global_settings.g_tw_phrase_file_fmt.format(ds_name))
    logging.debug('[map_between_tw_and_phrase] Load in %s tw phrase recs.' % str(len(df_tw_phrase)))

    phrase_id_prefix = 'ph#'
    phrase_id_suffix = 0
    d_phrase_id_to_phrase_str = dict()
    d_phrase_str_to_phrase_id = dict()
    d_phrase_id_to_tw = dict()
    d_tw_to_phrase_id = dict()
    for tw_id, tw_phrase_rec in df_tw_phrase.iterrows():
        l_phrase_tuple = tw_phrase_rec['tw_phrase']
        for phrase_tuple in l_phrase_tuple:
            l_phrase_token = phrase_tuple[0]
            l_phrase_token = functools.reduce(operator.iconcat, [sub_ph.split(' ') for sub_ph in l_phrase_token], [])
            l_phrase_pos = phrase_tuple[1]
            phrase_str = ' '.join(sorted(set([token.strip().lower() for token in l_phrase_token])))
            phrase_pos = ' '.join(sorted(l_phrase_pos))
            if phrase_str not in d_phrase_str_to_phrase_id:
                phrase_id = phrase_id_prefix + str(phrase_id_suffix)
                phrase_id_suffix += 1
                d_phrase_str_to_phrase_id[phrase_str] = phrase_id
                d_phrase_id_to_phrase_str[phrase_id] = phrase_str
                d_phrase_id_to_tw[phrase_id] = [tw_id]
                if phrase_id_suffix % 10000 == 0 and phrase_id_suffix >= 10000:
                    logging.debug('[map_between_tw_and_phrase] Log in %s phrases and %s tws in %s secs.'
                                  % (len(d_phrase_id_to_phrase_str), len(d_tw_to_phrase_id), time.time() - timer_start))
            else:
                phrase_id = d_phrase_str_to_phrase_id[phrase_str]
                if tw_id not in d_phrase_id_to_tw[phrase_id]:
                    d_phrase_id_to_tw[phrase_id].append(tw_id)
            if tw_id not in d_tw_to_phrase_id:
                d_tw_to_phrase_id[tw_id] = [(phrase_id, phrase_pos)]
            else:
                d_tw_to_phrase_id[tw_id].append((phrase_id, phrase_pos))

    l_ready = []
    for phrase_id in d_phrase_id_to_phrase_str:
        l_ready.append((phrase_id, d_phrase_id_to_phrase_str[phrase_id]))
    df_phrase_id_to_phrase_str = pd.DataFrame(l_ready, columns=['phrase_id', 'phrase_str'])
    df_phrase_id_to_phrase_str = df_phrase_id_to_phrase_str.set_index('phrase_id')
    pd.to_pickle(df_phrase_id_to_phrase_str, global_settings.g_phrase_id_to_phrase_str_file_fmt.format(ds_name))
    logging.debug('[map_between_tw_and_phrase] df_phrase_id_to_phrase_str done with %s recs.'
                  % str(len(df_phrase_id_to_phrase_str)))

    l_ready = []
    for phrase_str in d_phrase_str_to_phrase_id:
        l_ready.append((phrase_str, d_phrase_str_to_phrase_id[phrase_str]))
    df_phrase_str_to_phrase_id = pd.DataFrame(l_ready, columns=['phrase_str', 'phrase_id'])
    df_phrase_str_to_phrase_id = df_phrase_str_to_phrase_id.set_index('phrase_str')
    pd.to_pickle(df_phrase_str_to_phrase_id, global_settings.g_phrase_str_to_phrase_id_file_fmt.format(ds_name))
    logging.debug('[map_between_tw_and_phrase] df_phrase_str_to_phrase_id done with %s recs.'
                  % str(len(df_phrase_str_to_phrase_id)))

    l_ready = []
    for phrase_id in d_phrase_id_to_tw:
        l_ready.append((phrase_id, d_phrase_id_to_tw[phrase_id]))
    df_phrase_id_to_tw = pd.DataFrame(l_ready, columns=['phrase_id', 'l_tw'])
    df_phrase_id_to_tw = df_phrase_id_to_tw.set_index('phrase_id')
    pd.to_pickle(df_phrase_id_to_tw, global_settings.g_phrase_id_to_tw_file_fmt.format(ds_name))
    logging.debug('[map_between_tw_and_phrase] df_phrase_id_to_tw done with %s recs.'
                  % str(len(df_phrase_id_to_tw)))

    l_ready = []
    for tw_id in d_tw_to_phrase_id:
        l_ready.append((tw_id, d_tw_to_phrase_id[tw_id]))
    df_tw_to_phrase_id = pd.DataFrame(l_ready, columns=['tw_id', 'l_phrase_id'])
    df_tw_to_phrase_id = df_tw_to_phrase_id.set_index('tw_id')
    pd.to_pickle(df_tw_to_phrase_id, global_settings.g_tw_to_phrase_id_file_fmt.format(ds_name))
    logging.debug('[map_between_tw_and_phrase] df_tw_to_phrase_id done with %s recs.'
                  % str(len(df_tw_to_phrase_id)))

    logging.debug('[map_between_tw_and_phrase] All done in %s secs.' % str(time.time() - timer_start))


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    cmd = sys.argv[1]

    if cmd == 'gen_tasks':
        ds_name = sys.argv[2]
        num_tasks = sys.argv[3]
        job_id = sys.argv[4]
        gen_tw_phrase_extraction_tasks(ds_name, num_tasks, job_id)
    elif cmd == 'phrase':
        num_proc = sys.argv[2]
        job_id = sys.argv[3]
        tw_phrase_extraction_multiproc(num_proc, job_id)
    elif cmd == 'merge_int':
        ds_name = sys.argv[2]
        merge_tw_phrase_int(ds_name)
    elif cmd == 'mappings':
        ds_name = sys.argv[2]
        map_between_tw_and_phrase(ds_name)
