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

import global_settings


"""
Build sgraph
"""
def build_sgraph_from_json_str(sgraph_json_str):
    '''
    Return a directed graph with the start_char_idx and the end_char_idx relative to the tweet text as a whole.
    '''
    # logging.debug('[build_sgraph_from_json] Starts...')

    sgraph_json = json.loads(sgraph_json_str)
    l_snodes = sgraph_json['nodes']
    l_sedges = sgraph_json['edges']

    sgraph = nx.DiGraph()
    for snode in l_snodes:
        snode_id = snode['id']
        snode_pos = snode['pos']
        snode_sentiments = snode['sentiments']
        snode_token_str = snode['token_str']
        snode_token_start = snode['token_start']
        snode_token_end = snode['token_end']
        sgraph.add_node(snode_id, pos=snode_pos, sentiments=snode_sentiments, token_str=snode_token_str,
                        token_start=snode_token_start, token_end=snode_token_end)

    for sedge in l_sedges:
        src_id = sedge['src_id']
        trg_id = sedge['trg_id']
        sgraph.add_edge(src_id, trg_id)

    # build a segment tree over the binary constituent sentiment tree
    if len(sgraph.nodes()) == 1 and len(sgraph.edges()) == 0:
        for node in sgraph.nodes(data=True):
            if node[1]['token_start'] != -1 and node[1]['token_end'] != -1:
                return sgraph, node[1]['token_start'], node[1]['token_end']

    d_parents = dict()
    for node in sgraph.nodes(data=True):
        if node[1]['token_start'] != -1 and node[1]['token_end'] != -1:
            l_parents = list(sgraph.predecessors(node[0]))
            if len(l_parents) != 1:
                raise Exception('[build_sgraph_from_json] Leaf has multiple parents: %s, sgraph_json_str: %s'
                                % (node, sgraph_json_str))
            parent = l_parents[0]
            if parent not in d_parents:
                d_parents[parent] = [node]
            else:
                d_parents[parent].append(node)

    while len(d_parents) > 0:
        s_done_parents = set([])
        d_new_parents = dict()
        for parent in d_parents:
            if len(list(sgraph.successors(parent))) == len(d_parents[parent]):
                parent_start = min([child[1]['token_start'] for child in d_parents[parent]])
                parent_end = max([child[1]['token_end'] for child in d_parents[parent]])
                sgraph.nodes(data=True)[parent]['token_start'] = parent_start
                sgraph.nodes(data=True)[parent]['token_end'] = parent_end
                s_done_parents.add(parent)
                if sgraph.nodes(data=True)[parent]['pos'] == 'ROOT':
                    continue
                l_new_parents = list(sgraph.predecessors(parent))
                if len(l_new_parents) != 1:
                    raise Exception('[build_sgraph_from_json] Leaf has multiple parents: %s' % str(parent))
                new_parent = l_new_parents[0]
                if new_parent in d_new_parents:
                    d_new_parents[new_parent].append((parent, sgraph.nodes(data=True)[parent]))
                else:
                    d_new_parents[new_parent] = [(parent, sgraph.nodes(data=True)[parent])]
        for parent in s_done_parents:
            d_parents.pop(parent)
        for new_parent in d_new_parents:
            if not new_parent in d_parents:
                d_parents[new_parent] = d_new_parents[new_parent]
            else:
                d_parents[new_parent] += d_new_parents[new_parent]

    graph_start = min([node[1]['token_start'] for node in sgraph.nodes(data=True)])
    graph_end = max([node[1]['token_end'] for node in sgraph.nodes(data=True)])
    # logging.debug('[build_sgraph_from_json] sgraph is done: %s' % nx.info(sgraph))
    return sgraph, graph_start, graph_end


def build_tw_sgraph_single_proc(task_id):
    logging.debug('[build_tw_sgraph_single_proc] Proc %s: Starts.' % str(task_id))
    timer_start = time.time()

    with open(global_settings.g_tw_sent_int_file_fmt.format(task_id), 'r') as in_fd:
        l_tw_sent = json.load(in_fd)
        in_fd.close()
    logging.debug('[build_tw_sgraph_single_proc] Proc %s: Load in %s tw sent recs.' % (task_id, len(l_tw_sent)))

    ready_cnt = 0
    l_ready = []
    for d_tw_sent_rec in l_tw_sent:
        tw_id = d_tw_sent_rec['tw_id']
        l_sgraph_json_str = d_tw_sent_rec['tw_sgraph']
        if len(l_sgraph_json_str) <= 0:
            logging.error('[build_tw_sgraph_single_proc] Proc %s: No sgraph for tw_id: %s' % (task_id, tw_id))
            continue
        l_tw_sgraph = []
        for sgraph_json_str in l_sgraph_json_str:
            sgraph, graph_start, graph_end = build_sgraph_from_json_str(sgraph_json_str)
            sgraph_out_str = json.dumps(nx.adjacency_data(sgraph))
            l_tw_sgraph.append((sgraph_out_str, graph_start, graph_end))
        l_ready.append((tw_id, l_tw_sgraph))
        ready_cnt += 1
        if ready_cnt % 100 == 0 and ready_cnt >= 100:
            logging.debug('[build_tw_sgraph_single_proc] Proc %s: %s tw sgraphs done in %s secs.'
                          % (task_id, ready_cnt, time.time() - timer_start))
    logging.debug('[build_tw_sgraph_single_proc] Proc %s: %s tw sgraphs done in %s secs.'
                  % (task_id, ready_cnt, time.time() - timer_start))
    df_sgraph = pd.DataFrame(l_ready, columns=['tw_id', 'tw_sgraph'])
    pd.to_pickle(df_sgraph, global_settings.g_tw_sgraph_int_file_fmt.format(task_id))
    logging.debug('[build_tw_sgraph_single_proc] Proc %s: All done with %s tw sgraph recs in %s secs.'
                  % (task_id, len(df_sgraph), time.time() - timer_start))


def build_tw_sgraph_multiproc(num_proc, job_id):
    logging.debug('[build_tw_sgraph_multiproc] Starts.')
    timer_start = time.time()

    l_task_ids = [str(job_id) + '#' + str(idx) for idx in range(int(num_proc))]
    l_proc = []
    for task_id in l_task_ids:
        p = multiprocessing.Process(target=build_tw_sgraph_single_proc,
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
                logging.debug('[build_tw_sgraph_multiproc] %s is finished.' % p.name)
    logging.debug('[build_tw_sgraph_multiproc] All done in %s secs.' % str(time.time() - timer_start))


def merge_tw_sgraph_int(ds_name):
    logging.debug('[merge_tw_sgraph_int]')
    timer_start = time.time()

    l_int = []
    for (dirpath, dirname, filenames) in walk(global_settings.g_tw_sent_int_folder):
        for filename in filenames:
            if filename[-7:] != '.pickle' or filename[:14] != 'tw_sgraph_int_':
                continue
            df_int = pd.read_pickle(dirpath + filename)
            l_int.append(df_int)
    logging.debug('[merge_tw_sgraph_int] Read in %s int dfs.' % str(len(l_int)))
    df_merge = pd.concat(l_int)
    df_merge = df_merge.set_index('tw_id')
    pd.to_pickle(df_merge, global_settings.g_tw_sgraph_file_fmt.format(ds_name))
    logging.debug('[merge_tw_sgraph_int] All done with %s recs in %s secs.'
                  % (len(df_merge), time.time() - timer_start))


"""
Phrase Sentiment
"""
def sentiment_calibration(sent_vec_1, sent_vec_2):
    '''
    Return: (True for calibrated, sent_vec)
    '''
    sent_vec_1 = np.asarray(sent_vec_1)
    sent_vec_2 = np.asarray(sent_vec_2)
    sent_class_1 = np.abs(np.argmax(sent_vec_1) - 2)
    sent_class_2 = np.abs(np.argmax(sent_vec_2) - 2)
    if sent_class_1 > sent_class_2:
        return True, sent_vec_1
    elif sent_class_2 > sent_class_1:
        return True, sent_vec_2
    else:
        return False, (sent_vec_1 + sent_vec_2) / 2.0


def find_mininal_subtree_from_specified_root_for_phrase(sgraph, root, phrase_start, phrase_end):
    cur_node = (root, sgraph.nodes(data=True)[root])
    if phrase_start < cur_node[1]['token_start'] or phrase_end > cur_node[1]['token_end']:
        return None
    while True:
        if phrase_start >= cur_node[1]['token_start'] and phrase_end <= cur_node[1]['token_end']:
            go_deep = False
            for child in sgraph.successors(cur_node[0]):
                if phrase_start >= sgraph.nodes(data=True)[child]['token_start'] \
                        and phrase_end <= sgraph.nodes(data=True)[child]['token_end']:
                    cur_node = (child, sgraph.nodes(data=True)[child])
                    go_deep = True
                    break
            if go_deep:
                continue
            else:
                break
    return cur_node[0]


def compute_phrase_sentiment(sgraph, phrase_start, phrase_end, l_span):
    '''
    'phrase_start' and 'phrase_end' are the exact and absolute character indices of a phrase relative to the text.
    '''
    # find the minimal segment in sgraph that includes [phrase_start, phrase_end]
    root = None
    for node in sgraph.nodes():
        if sgraph.in_degree(node) == 0:
            root = node
    if root is None:
        raise Exception('[compute_phrase_sentiment] No root found.')

    phrase_sentiment = None
    min_subtree_root = find_mininal_subtree_from_specified_root_for_phrase(sgraph, root, phrase_start, phrase_end)
    if min_subtree_root is not None:
        phrase_sentiment = sgraph.nodes(data=True)[min_subtree_root]['sentiments']
    if phrase_sentiment is None:
        raise Exception('[compute_phrase_sentiment] Cannot get sentiment for phrase: %s, %s.'
                        % (phrase_start, phrase_end))
    elif len(l_span) > 1:
        # sentiment calibration:
        # when the input phrase as a whole is determined to be neutral, we look into the sentiment of each individual
        # token, and prefer to emphasize the most polarized sentiment.
        # in the case that one token is positive and the other is negative, we still prefer to emphasized the most
        # polarized one.
        primary_sentiment_class = np.argmax(phrase_sentiment)
        if primary_sentiment_class == 2:
            token_1_start = l_span[0][0]
            token_1_end = l_span[0][1]
            token_2_start = l_span[1][0]
            token_2_end = l_span[1][1]
            subtree_root_1 = min_subtree_root
            subtree_root_2 = min_subtree_root
            min_subtree_root_1 = find_mininal_subtree_from_specified_root_for_phrase(sgraph, subtree_root_1,
                                                                                     token_1_start, token_1_end)
            min_subtree_root_2 = find_mininal_subtree_from_specified_root_for_phrase(sgraph, subtree_root_2,
                                                                                     token_2_start, token_2_end)
            if min_subtree_root_1 is not None and min_subtree_root_2 is not None:
                token_1_sentiment = sgraph.nodes(data=True)[min_subtree_root_1]['sentiments']
                token_2_sentiment = sgraph.nodes(data=True)[min_subtree_root_2]['sentiments']
                if token_1_sentiment is None or token_2_sentiment is None:
                    raise Exception('[compute_phrase_sentiment] Something wrong when getting sentiments for %s.'
                                    % str(l_span))
                # TODO
                # may need a better function calibrates the sentiment based on token_1_sentiment and token_2_sentiment
                # token_1_sentiment_class = np.abs(np.argmax(token_1_sentiment) - 2)
                # token_2_sentiment_class = np.abs(np.argmax(token_2_sentiment) - 2)
                # if token_1_sentiment_class == 0 and token_2_sentiment_class == 0:
                #     return phrase_sentiment
                # elif token_1_sentiment_class > token_2_sentiment_class:
                #     return token_1_sentiment
                # elif token_2_sentiment_class > token_1_sentiment_class:
                #     return token_2_sentiment
                # else:
                #     return ((np.asarray(token_1_sentiment) + np.asarray(token_2_sentiment)) / 2.0).tolist()
                calibrated, calibrated_phrase_sentiment = sentiment_calibration(token_1_sentiment, token_2_sentiment)
                if calibrated:
                    return calibrated_phrase_sentiment.tolist()
                else:
                    return phrase_sentiment
            else:
                logging.error('[compute_phrase_sentiment] Something wrong sentiment calibration: sgraph: %s, '
                              'phrase_start: %s, phrase_end: %s, l_span: %s' % sgraph, phrase_start, phrase_end, l_span)
                return phrase_sentiment
    else:
        return phrase_sentiment


def compute_sentiment_for_one_phrase_in_one_tw(l_parsed_sgraph_info, phrase_start, phrase_end, l_phspan):
    '''
    'l_parsed_sgraph_info' is the list of sgraphs for the given tweet.
    Each element is a tuple: (sgraph in nx.Graph, sgraph_start, sgraph_end)
    'l_phspan' is for the given phrase. Each element is for a token in the phrase.
    '''
    phrase_sentiment = None
    for sgraph_info in l_parsed_sgraph_info:
        sgraph = sgraph_info[0]
        sgraph_start = sgraph_info[1]
        sgraph_end = sgraph_info[2]
        if phrase_start >= sgraph_start and phrase_end <= sgraph_end:
            phrase_sentiment = compute_phrase_sentiment(sgraph, phrase_start, phrase_end, l_phspan)
            break
        else:
            continue
    return phrase_sentiment


def compute_phrase_sentiment_for_one_tw(l_sgraph_info, l_phrase):
    '''
    'l_sgraph_info' and 'l_phrase' are for one tweet.
    Returns [([token_1, token_2], [pos_1, pos_2], [(token_1_start, token_1_end), (token_2_start, token_2_end)],
    phrase_sentiment), ...]
    '''
    if l_sgraph_info is None or len(l_sgraph_info) <= 0 or l_phrase is None or len(l_phrase) <= 0:
        logging.error('[compute_phrase_sentiment_for_one_tw] Invalid l_sgraph_info or l_phrase.')
        return None

    l_parsed_sgraph_info = []
    for sgraph_info in l_sgraph_info:
        sgraph_json_str = sgraph_info[0]
        sgraph_start = sgraph_info[1]
        sgraph_end = sgraph_info[2]
        sgraph = nx.adjacency_graph(json.loads(sgraph_json_str))
        l_parsed_sgraph_info.append((sgraph, sgraph_start, sgraph_end))

    l_ready = []
    l_leftover = []
    for phrase in l_phrase:
        l_token = phrase[0]
        l_pos = phrase[1]
        l_span = phrase[2]
        if len(l_token) == 1:
            phrase_start = l_span[0][0]
            phrase_end = l_span[0][1]
        elif len(l_token) == 2:
            phrase_token_1_start = l_span[0][0]
            phrase_token_1_end = l_span[0][1]
            phrase_token_2_start = l_span[1][0]
            phrase_token_2_end = l_span[1][1]
            phrase_start = min(phrase_token_1_start, phrase_token_2_start)
            phrase_end = max(phrase_token_1_end, phrase_token_2_end)
        else:
            logging.error('[compute_phrase_sentiment_for_one_tw] Invalid phrase: %s' % str(phrase))
            continue

        phrase_sentiment = compute_sentiment_for_one_phrase_in_one_tw(l_parsed_sgraph_info, phrase_start, phrase_end,
                                                                      l_span)
        # for sgraph_info in l_sgraph_info:
        #     sgraph_json_str = sgraph_info[0]
        #     sgraph_start = sgraph_info[1]
        #     sgraph_end = sgraph_info[2]
        #     sgraph = nx.adjacency_graph(json.loads(sgraph_json_str))
        #     if phrase_start >= sgraph_start and phrase_end <= sgraph_end:
        #         phrase_sentiment = compute_phrase_sentiment(sgraph, phrase_start, phrase_end, l_span)
        #         break
        #     else:
        #         continue
        if phrase_sentiment is None:
            # logging.error('[compute_phrase_sentiment_for_one_tw] Phrase %s cannot fit in any sgraph.' % str(phrase))
            l_leftover.append(phrase)
        l_ready.append((l_token, l_pos, l_span, phrase_sentiment))
    return l_ready, l_leftover


def compute_phrase_sentiment_single_proc(task_id):
    logging.debug('[compute_phrase_sentiment_single_proc] Proc %s: Starts.' % str(task_id))
    timer_start = time.time()

    df_task = pd.read_pickle(global_settings.g_tw_phrase_sentiment_task_file_fmt.format(task_id))
    logging.debug('[compute_phrase_sentiment_single_proc] Proc %s: Load in %s task recs.' % (task_id, len(df_task)))

    read_cnt = 0
    l_ready = []
    leftover_phrase_cnt = 0
    l_leftover_rec = []
    for tw_id, task_rec in df_task.iterrows():
        l_tw_phrase_info = task_rec['tw_phrase']
        l_tw_sgraph_info = task_rec['tw_sgraph']
        l_phrase_sentiment, l_leftover = compute_phrase_sentiment_for_one_tw(l_tw_sgraph_info, l_tw_phrase_info)
        l_ready.append((tw_id, l_phrase_sentiment))
        if len(l_leftover) > 0:
            l_leftover_rec.append((tw_id, l_leftover))
            leftover_phrase_cnt += len(l_leftover)
        read_cnt += 1
        if read_cnt % 5000 == 0 and read_cnt >= 5000:
            logging.debug('[compute_phrase_sentiment_single_proc] Proc %s: %s recs done in %s secs. '
                          'leftover_phrase_cnt: %s'
                          % (task_id, read_cnt, time.time() - timer_start, leftover_phrase_cnt))
    logging.debug('[compute_phrase_sentiment_single_proc] Proc %s: %s recs done in %s secs. '
                  'leftover_phrase_cnt: %s'
                  % (task_id, read_cnt, time.time() - timer_start, leftover_phrase_cnt))
    df_phrase_sentiment = pd.DataFrame(l_ready, columns=['tw_id', 'tw_phrase_sentiment'])
    pd.to_pickle(df_phrase_sentiment, global_settings.g_tw_phrase_sentiment_int_file_fmt.format(task_id))
    df_leftover = pd.DataFrame(l_leftover_rec, columns=['tw_id', 'tw_leftover_phrase'])
    pd.to_pickle(df_leftover, global_settings.g_tw_phrase_sentiment_leftover_int_file_fmt.format(task_id))
    logging.debug('[compute_phrase_sentiment_single_proc] Proc %s: All done with %s recs and %s leftover in %s secs.'
                  % (task_id, len(df_phrase_sentiment), len(df_leftover), time.time() - timer_start))


def compute_phrase_sentiment_multiproc(num_proc, job_id):
    logging.debug('[compute_phrase_sentiment_multiproc] Starts.')
    timer_start = time.time()

    l_task_ids = [str(job_id) + '#' + str(idx) for idx in range(int(num_proc))]
    l_proc = []
    for task_id in l_task_ids:
        p = multiprocessing.Process(target=compute_phrase_sentiment_single_proc,
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
                logging.debug('[compute_phrase_sentiment_multiproc] %s is finished.' % p.name)
    logging.debug('[compute_phrase_sentiment_multiproc] All done in %s secs.' % str(time.time() - timer_start))


def merge_phrase_sentiment_int(ds_name):
    logging.debug('[merge_phrase_sentiment_int]')
    timer_start = time.time()

    l_int = []
    for (dirpath, dirname, filenames) in walk(global_settings.g_tw_sent_int_folder):
        for filename in filenames:
            if filename[-7:] != '.pickle' or filename[:24] != 'tw_phrase_sentiment_int_':
                continue
            df_int = pd.read_pickle(dirpath + filename)
            l_int.append(df_int)
    logging.debug('[merge_phrase_sentiment_int] Read in %s phrase sentiment int dfs.' % str(len(l_int)))
    df_merge = pd.concat(l_int)
    df_merge = df_merge.set_index('tw_id')
    pd.to_pickle(df_merge, global_settings.g_tw_phrase_sentiment_file_fmt.format(ds_name))
    logging.debug('[merge_phrase_sentiment_int] Phrase sentiment done with %s recs in %s secs.'
                  % (len(df_merge), time.time() - timer_start))

    l_int = []
    for (dirpath, dirname, filenames) in walk(global_settings.g_tw_sent_int_folder):
        for filename in filenames:
            if filename[-7:] != '.pickle' or filename[:33] != 'tw_phrase_sentiment_leftover_int_':
                continue
            df_int = pd.read_pickle(dirpath + filename)
            l_int.append(df_int)
    logging.debug('[merge_phrase_sentiment_int] Read in %s phrase sentiment leftover int dfs.' % str(len(l_int)))
    df_merge = pd.concat(l_int)
    df_merge = df_merge.set_index('tw_id')
    pd.to_pickle(df_merge, global_settings.g_tw_phrase_sentiment_leftover_file_fmt.format(ds_name))
    logging.debug('[merge_phrase_sentiment_int] Phrase sentiment leftover done with %s recs in %s secs.'
                  % (len(df_merge), time.time() - timer_start))


def gen_phrase_sentiment_tasks(ds_name, num_tasks, job_id):
    logging.debug('[gen_phrase_sentiment_tasks] Starts.')
    timer_start = time.time()

    df_tw_sgraph = pd.read_pickle(global_settings.g_tw_sgraph_file_fmt.format(ds_name))
    num_tw_sgraph = len(df_tw_sgraph)
    logging.debug('[gen_phrase_sentiment_tasks] Load in %s tw sgraph recs.' % str(num_tw_sgraph))

    df_tw_phrase = pd.read_pickle(global_settings.g_tw_phrase_file_fmt.format(ds_name))
    num_tw_phrase = len(df_tw_phrase)
    logging.debug('[gen_phrase_sentiment_tasks] Load in %s tw phrase recs.' % str(num_tw_phrase))

    l_task_ready = []
    for tw_id, tw_phrase_rec in df_tw_phrase.iterrows():
        if not tw_id in df_tw_sgraph.index:
            continue
        tw_phrase = tw_phrase_rec['tw_phrase']
        tw_sgarph = df_tw_sgraph.loc[tw_id]['tw_sgraph']
        l_task_ready.append((tw_id, tw_phrase, tw_sgarph))
    num_tw_recs = len(l_task_ready)
    logging.debug('[gen_phrase_sentiment_tasks] %s tws for tasks.' % str(num_tw_recs))

    num_tasks = int(num_tasks)
    batch_size = math.ceil(num_tw_recs / num_tasks)
    l_tasks = []
    for i in range(0, num_tw_recs, batch_size):
        if i + batch_size < num_tw_recs:
            l_tasks.append(l_task_ready[i:i + batch_size])
        else:
            l_tasks.append(l_task_ready[i:])
    logging.debug('[gen_tw_phrase_extraction_tasks] Need to generate %s tasks.' % str(len(l_tasks)))

    for idx, task in enumerate(l_tasks):
        df_task = pd.DataFrame(task, columns=['tw_id', 'tw_phrase', 'tw_sgraph'])
        df_task = df_task.set_index('tw_id')
        pd.to_pickle(df_task, global_settings.g_tw_phrase_sentiment_task_file_fmt.format(str(job_id) + '#' + str(idx)))
    logging.debug('[gen_phrase_sentiment_tasks] All done with %s tasks in %s secs.'
                  % (len(l_tasks), time.time() - timer_start))


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    cmd = sys.argv[1]

    if cmd == 'build_sgraph':
        num_proc = sys.argv[2]
        job_id = sys.argv[3]
        build_tw_sgraph_multiproc(num_proc, job_id)
    elif cmd == 'merge_sgraph_int':
        ds_name = sys.argv[2]
        merge_tw_sgraph_int(ds_name)
    elif cmd == 'gen_phrase_sentiment_tasks':
        ds_name = sys.argv[2]
        num_tasks = sys.argv[3]
        job_id = sys.argv[4]
        gen_phrase_sentiment_tasks(ds_name, num_tasks, job_id)
    elif cmd == 'phrase_sentiment':
        num_proc = sys.argv[2]
        job_id = sys.argv[3]
        compute_phrase_sentiment_multiproc(num_proc, job_id)
    elif cmd == 'merge_phrase_sentiment_int':
        ds_name = sys.argv[2]
        merge_phrase_sentiment_int(ds_name)