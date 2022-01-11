import json
import logging
import math
import multiprocessing
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
from sentiment_analysis import compute_phrase_sentiment_for_one_tw, sentiment_calibration

sys.path.insert(1, global_settings.g_lexvec_model_folder)
import model as lexvec

"""
A query is just a piece of text. It can be a token, multiple tokens, a sentence, multiple sentences or a general text. 
Each query will be processed and finally represented as a set of phrases if any. The process follows the pipeline:
query string -> text clean (a set of cleaned sentences) -> semantic units (noun phrases and core clause structures 
for each sentence) -> phrases (extracted from the semantic units) -> phrase embedding (embedding vector for each phrase)

The extracted phrases are then compared to all phrases (or phrase clusters) in the knowledge base. Then the semantically
similar basic phrases are mapped back to their original tweets. And all directly related phrases from the tweets are
collected and added into the final output knowledge graph. 
"""


def extract_phrases_from_cls_graph(cls_graph):
    if cls_graph is None:
        return None
    s_covered_nodes = []
    l_phrases = []
    try:
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


def extract_phrase_from_nps(nps):
    if nps is None or len(nps) <= 0:
        return None
    nps = [([noun_phrase[0]], ['NOUN'], [(noun_phrase[2], noun_phrase[3])]) for noun_phrase in nps]
    return nps


def load_lexvec_model():
    lexvec_model = lexvec.Model(global_settings.g_lexvec_vect_file_path)
    embedding_len = len(lexvec_model.word_rep('the'))
    logging.debug('[load_lexvec_model] The length of embeddings is %s' % embedding_len)
    return lexvec_model, embedding_len


def phrase_embedding(lexvec_model, embedding_len, l_token):
    if lexvec_model is None:
        raise Exception('lexvec_model is not loaded!')
    phrase_vec = np.zeros(embedding_len)
    for word in l_token:
        word_vec = lexvec_model.word_rep(word)
        phrase_vec += word_vec
    if not np.isfinite(phrase_vec).all():
        logging.error('Invalid embedding for %s!' % str(l_token))
        phrase_vec = np.zeros(embedding_len)
    phrase_vec = np.asarray(phrase_vec, dtype=np.float32)
    phrase_vec = preprocessing.normalize(phrase_vec.reshape(1, -1))
    phrase_vec = phrase_vec[0]
    return phrase_vec


def query_str_to_phrases(query_str, query_id, lexvec_model, embedding_len):
    logging.debug('[query_str_to_phrases] Query %s: Starts.' % str(query_id))
    timer_start = time.time()

    sem_unit_ext_ins = SemUnitsExtractor(global_settings.g_sem_units_extractor_config_file)
    # text clean
    tw_clean_txt = sem_unit_ext_ins.text_clean(query_str)
    if tw_clean_txt is None or tw_clean_txt == '':
        logging.error('[query_str_to_phrases] Query %s: query_str is trivial: %s' % (query_id, query_str))
        return None

    # semantic units
    nx_cls, l_nps = sem_unit_ext_ins.extract_sem_units_from_text(tw_clean_txt, str(query_id))

    # phrases
    l_cls_phrase = extract_phrases_from_cls_graph(nx_cls)
    l_nps_phrase = extract_phrase_from_nps(l_nps)
    l_phrase = []
    if l_cls_phrase is not None and len(l_cls_phrase) > 0:
        l_phrase += l_cls_phrase
    if l_nps_phrase is not None and len(l_nps_phrase) > 0:
        l_phrase += l_nps_phrase
    if len(l_phrase) <= 0:
        logging.error('[query_str_to_phrases] Query %s: query_str has no non-trivial phrase: %s'
                      % (query_id, query_str))
        return None

    # phrase embedding
    l_phrase_ready = []
    for idx, phrase in enumerate(l_phrase):
        l_token = []
        for phrase_ele in phrase[0]:
            l_token += [token.strip().lower() for token in phrase_ele.split(' ')]
        l_token = list(set(l_token))
        phrase_embed = phrase_embedding(lexvec_model, embedding_len, l_token)
        phrase_id = str(query_id) + '#' + str(idx)
        l_phrase_ready.append((phrase_id, (phrase[0], phrase[1], phrase[2], phrase_embed)))

    df_phrase_ready = pd.DataFrame(l_phrase_ready, columns=['q_phrase_id', 'q_phrase'])
    df_phrase_ready = df_phrase_ready.set_index('q_phrase_id')
    pd.to_pickle(df_phrase_ready, global_settings.g_ks_graph_query_to_phrases_file_fmt.format(str(query_id)))
    logging.debug('[query_str_to_phrases] Query %s: All done with %s phrases in %s secs.'
                  % (query_id, len(df_phrase_ready), time.time() - timer_start))
    return df_phrase_ready


def find_similar_phrases_for_query_phrases(query_id, df_query_phrase, np_phrase_embed_trans,
                                           d_phrase_row_id_to_phrase_id):
    logging.debug('[find_similar_phrases_for_query_phrases] Query %s: Starts.' % str(query_id))
    timer_start = time.time()

    l_q_phrase_embed = []
    for q_phrase_id, q_phrase_rec in df_query_phrase.iterrows():
        q_phrase = q_phrase_rec['q_phrase']
        l_q_phrase_embed.append(q_phrase[3])
    np_q_phrase_embed = np.stack(l_q_phrase_embed)
    np_phrase_sim = np.matmul(np_q_phrase_embed, np_phrase_embed_trans)
    logging.debug('[find_similar_phrases_for_query_phrases] Query %s: Get np_phrase_sim in %s secs.'
                  % (query_id, time.time() - timer_start))

    sim_phrase_indices = np.nonzero(np_phrase_sim >= global_settings.g_phrase_sim_threshold)
    l_q_phrase_id = df_query_phrase.index.to_list()
    d_q_phrase_sim_phrases = {q_phrase_id: [] for q_phrase_id in l_q_phrase_id}
    for i in range(len(sim_phrase_indices[0])):
        q_phrase_id = l_q_phrase_id[sim_phrase_indices[0][i]]
        sim_phrase_row_id = d_phrase_row_id_to_phrase_id[str(sim_phrase_indices[1][i])]
        d_q_phrase_sim_phrases[q_phrase_id].append(sim_phrase_row_id)

    l_ready = []
    for q_phrase_id in d_q_phrase_sim_phrases:
        l_ready.append((q_phrase_id, d_q_phrase_sim_phrases[q_phrase_id]))
    df_q_phrase_to_sim_phrase = pd.DataFrame(l_ready, columns=['q_phrase_id', 'sim_phrase'])
    df_q_phrase_to_sim_phrase = df_q_phrase_to_sim_phrase.set_index('q_phrase_id')
    pd.to_pickle(df_q_phrase_to_sim_phrase,
                 global_settings.g_ks_graph_q_phrase_to_sim_phrase_file_fmt.format(str(query_id)))
    logging.debug('[find_similar_phrases_for_query_phrases] Query %s: Sim phrases are done in %s secs: %s'
                  % (query_id, time.time() - timer_start, [(q_phrase_id, len(d_q_phrase_sim_phrases[q_phrase_id]))
                                                           for q_phrase_id in d_q_phrase_sim_phrases]))
    return df_q_phrase_to_sim_phrase


def query_str_to_sim_phrases(query_id, query_str, ds_name):
    logging.debug('[query_str_to_sim_phrases] Query %s: Starts with string: %s' % (query_id, query_str))
    timer_start = time.time()

    lexvec_model, embedding_len = load_lexvec_model()
    df_query_phrase = query_str_to_phrases(query_str, query_id, lexvec_model, embedding_len)

    df_phrase_embed = pd.read_pickle(global_settings.g_phrase_embed_file_fmt.format(ds_name))
    np_phrase_embed = np.stack(df_phrase_embed['phrase_embed'].to_list())
    np_phrase_embed_trans = np.transpose(np_phrase_embed)
    logging.debug('[query_str_to_sim_phrases] Query %s: Load in %s phrase embeds.' % (query_id, len(df_phrase_embed)))

    with open(global_settings.g_phrase_row_id_to_phrase_id_file_fmt.format(ds_name), 'r') as in_fd:
        d_phrase_row_id_to_phrase_id = json.load(in_fd)
        in_fd.close()
    find_similar_phrases_for_query_phrases(query_id, df_query_phrase, np_phrase_embed_trans,
                                           d_phrase_row_id_to_phrase_id)
    logging.debug('[query_str_to_sim_phrases] Query %s: All done in %s secs.' % (query_id, time.time() - timer_start))


def gen_sim_phrase_to_ks_graph_tasks(query_id, ds_name, num_tasks):
    logging.debug('[gen_sim_phrase_to_ks_graph_tasks] Query %s: Starts.' % str(query_id))
    timer_start = time.time()

    df_q_phrase_to_sim_phrase = pd.read_pickle(global_settings.g_ks_graph_q_phrase_to_sim_phrase_file_fmt
                                               .format(str(query_id)))
    logging.debug('[gen_sim_phrase_to_ks_graph_tasks] Query %s: Load in %s q_phrases.'
                  % (query_id, len(df_q_phrase_to_sim_phrase)))

    df_phrase_id_to_tw = pd.read_pickle(global_settings.g_phrase_id_to_tw_file_fmt.format(ds_name))
    logging.debug('[gen_sim_phrase_to_ks_graph_tasks] Query %s: Load in df_phrase_id_to_tw with %s recs.'
                  % (query_id, len(df_phrase_id_to_tw)))

    l_recs = []
    for q_phrase_id, sim_phrase_rec in df_q_phrase_to_sim_phrase.iterrows():
        l_sim_phrase = sim_phrase_rec['sim_phrase']
        for sim_phrase in l_sim_phrase:
            l_tw_id = df_phrase_id_to_tw.loc[sim_phrase]
            for tw_id in l_tw_id:
                l_recs.append((q_phrase_id, sim_phrase, tw_id))
    num_recs = len(l_recs)
    logging.debug('[gen_sim_phrase_to_ks_graph_tasks] Query %s: %s task recs in total.' % (query_id, num_recs))

    num_tasks = int(num_tasks)
    batch_size = math.ceil(num_recs / num_tasks)
    l_tasks = []
    for i in range(0, num_recs, batch_size):
        if i + batch_size < num_recs:
            l_tasks.append(l_recs[i:i + batch_size])
        else:
            l_tasks.append(l_recs[i:])
    logging.debug('[gen_sim_phrase_to_ks_graph_tasks] Query %s: Need to generate %s tasks.' % (query_id, len(l_tasks)))

    for idx, task in enumerate(l_tasks):
        df_task = pd.DataFrame(task, columns=['q_phrase_id', 'sim_phrase', 'tw_id'])
        # df_task = df_task.set_index('q_phrase_id')
        task_id = str(query_id) + '#' + str(idx)
        pd.to_pickle(df_task, global_settings.g_ks_graph_task_file_fmt.format(task_id))
    logging.debug('[gen_sim_phrase_to_ks_graph_tasks] Query %s: All done in %s secs.'
                  % (query_id, time.time() - timer_start))


def build_ks_graph_for_one_sim_phrase_on_one_tw(task_id, sim_phrase_id, tw_id, df_tw_phrase, df_tw_to_phrase_id,
                                                df_phrase_id_to_phrase_str, df_tw_sent, df_tw_sgraph):
    '''
    'sim_phrase_id' gives the center node of the desired graph, and this node is a phrase of 'tw_id'.
    Link each of other nodes in 'tw_id' to 'sim_phrase_id' and build a star graph.
    Each node contains the phrase string, the POS, and the phrase id.
    Each edge contains the sentiment over the phrases at the ends.
    '''
    l_tw_phrase_id = [item[0] for item in df_tw_to_phrase_id.loc[tw_id]['l_phrase_id']]
    if sim_phrase_id not in l_tw_phrase_id:
        raise Exception('[build_ks_graph_for_one_sim_phrase_on_one_tw] Task %s: %s is not in %s.'
                        % (task_id, sim_phrase_id, tw_id))
    sim_phrase_idx_in_tw = l_tw_phrase_id.index(sim_phrase_id)
    l_tw_phrase_tup = df_tw_phrase.loc[tw_id]['tw_phrase']
    sim_phrase_tup = l_tw_phrase_tup[sim_phrase_idx_in_tw]
    sim_phrase_str = df_phrase_id_to_phrase_str.loc[sim_phrase_id]['phrase_str']
    sim_phrase_pos = ' '.join(sorted(sim_phrase_tup[1]))
    sim_phrase_span = sim_phrase_tup[2]
    sim_phrase_start = min([item[0] for item in sim_phrase_span])
    sim_phrase_end = max([item[1] for item in sim_phrase_span])

    l_sgraph_info = df_tw_sgraph.loc[tw_id]['tw_sgraph']

    tw_ks_graph = nx.Graph()
    tw_ks_graph.add_node(sim_phrase_id, str=[sim_phrase_str], pos=[sim_phrase_pos])
    for phrase_idx_in_tw in range(len(l_tw_phrase_tup)):
        relevant_phrase_tup = l_tw_phrase_tup[phrase_idx_in_tw]
        relevant_phrase_id = l_tw_phrase_id[phrase_idx_in_tw]
        tw_ks_graph.add_node(relevant_phrase_id,
                             str=[df_phrase_id_to_phrase_str.loc[relevant_phrase_id]['phrase_str']],
                             pos=[' '.join(sorted(relevant_phrase_tup[1]))])

        relevant_phrase_str = df_phrase_id_to_phrase_str.loc[relevant_phrase_id]['phrase_str']
        relevant_phrase_pos = ' '.join(sorted(relevant_phrase_tup[1]))
        relevant_phrase_span = relevant_phrase_tup[2]
        relevant_phrase_start = min([item[0] for item in relevant_phrase_span])
        relevant_phrase_end = max([item[1] for item in relevant_phrase_span])
        l_sent_phrase = [([sim_phrase_str, relevant_phrase_str], [sim_phrase_pos, relevant_phrase_pos],
                          [(sim_phrase_start, sim_phrase_end), (relevant_phrase_start, relevant_phrase_end)])]
        l_sent_ready, l_leftover = compute_phrase_sentiment_for_one_tw(l_sgraph_info, l_sent_phrase)
        if len(l_sent_ready) >= 1:
            edge_sent = l_sent_ready[0][3]
        else:
            l_tw_phrase_sent = df_tw_sent.loc[tw_id]['tw_phrase_sentiment']
            sim_phrase_sent = l_tw_phrase_sent[sim_phrase_idx_in_tw][3]
            relevant_phrase_sent = l_tw_phrase_sent[phrase_idx_in_tw][3]
            if sim_phrase_sent is None or relevant_phrase_sent is None:
                continue
            _, edge_sent = sentiment_calibration(sim_phrase_sent, relevant_phrase_sent)
        tw_ks_graph.add_edge(sim_phrase_id, relevant_phrase_id, sent=[((sim_phrase_id, relevant_phrase_id), edge_sent)])
    return tw_ks_graph


def divide_and_conquer_graph_compose(l_graph):
    batch_size = math.ceil(len(l_graph) / 2)
    if batch_size == len(l_graph):
        return nx.compose_all(l_graph)

    ret_graph = nx.compose(divide_and_conquer_graph_compose(l_graph[:batch_size]),
                           divide_and_conquer_graph_compose(l_graph[batch_size:]))
    return ret_graph


def build_ks_graph_single_proc(task_id, ds_name):
    logging.debug('[build_ks_graph_single_proc] Proc %s: Starts.' % str(task_id))
    timer_start = time.time()

    df_task = pd.read_pickle(global_settings.g_ks_graph_task_file_fmt.format(task_id))
    logging.debug('[build_ks_graph_single_proc] Proc %s: Load in %s task recs.' % (task_id, len(df_task)))

    df_tw_phrase = pd.read_pickle(global_settings.g_tw_phrase_file_fmt.format(ds_name))
    logging.debug('[build_ks_graph_single_proc] Proc %s: Load in %s tw phrase recs.' % (task_id, len(df_tw_phrase)))

    df_tw_to_phrase_id = pd.read_pickle(global_settings.g_tw_to_phrase_id_file_fmt.format(ds_name))
    logging.debug('[build_ks_graph_single_proc] Proc %s: Load in %s tw to phrase id recs.'
                  % (task_id, len(df_tw_to_phrase_id)))

    df_phrase_id_to_phrase_str = pd.read_pickle(global_settings.g_phrase_id_to_phrase_str_file_fmt.format(ds_name))
    logging.debug('[build_ks_graph_single_proc] Proc %s: Load in %s phrase id to phrase str recs.'
                  % (task_id, len(df_phrase_id_to_phrase_str)))

    df_tw_sent = pd.read_pickle(global_settings.g_tw_phrase_sentiment_file_fmt.format(ds_name))
    logging.debug('[build_ks_graph_single_proc] Proc %s: Load in %s tw sentiment recs.' % (task_id, len(df_tw_sent)))

    df_tw_sgraph = pd.read_pickle(global_settings.g_tw_sgraph_file_fmt.format(ds_name))
    logging.debug('[build_ks_graph_single_proc] Proc %s: Load in %s tw sgraph recs.' % (task_id, len(df_tw_sgraph)))

    d_tw_ks_graph = dict()
    done_cnt = 0
    for _, task_rec in df_task.iterrows():
        q_phrase_id = task_rec['q_phrase_id']
        sim_phrase_id = task_rec['sim_phrase']
        l_tw_id = task_rec['tw_id']
        for tw_id in l_tw_id:
            tw_ks_graph = build_ks_graph_for_one_sim_phrase_on_one_tw(task_id, sim_phrase_id, tw_id, df_tw_phrase,
                                                                      df_tw_to_phrase_id, df_phrase_id_to_phrase_str,
                                                                      df_tw_sent, df_tw_sgraph)
            if q_phrase_id not in d_tw_ks_graph:
                d_tw_ks_graph[q_phrase_id] = [tw_ks_graph]
            else:
                d_tw_ks_graph[q_phrase_id].append(tw_ks_graph)
            done_cnt += 1
            if done_cnt % 10 == 0 and done_cnt >= 10:
                logging.debug('[build_ks_graph_single_proc] Proc %s: %s tw_ks_graph done in %s secs.'
                              % (task_id, done_cnt, time.time() - timer_start))
    logging.debug('[build_ks_graph_single_proc] Proc %s: %s tw_ks_graph done in %s secs.'
                  % (task_id, done_cnt, time.time() - timer_start))

    l_ready = []
    for q_phrase_id in d_tw_ks_graph:
        # tw_ks_graph_sum = nx.Graph()
        tw_ks_graph_sum = divide_and_conquer_graph_compose(d_tw_ks_graph[q_phrase_id])
        # for tw_ks_graph in d_tw_ks_graph[q_phrase_id]:
        #     tw_ks_graph_sum = nx.compose(tw_ks_graph_sum, tw_ks_graph)
        # tw_ks_graph_sum_json_str = json.dumps(nx.adjacency_data(tw_ks_graph_sum))
        l_ready.append((q_phrase_id, tw_ks_graph_sum))
    df_ready = pd.DataFrame(l_ready, columns=['q_phrase_id', 'ks_graph'])
    df_ready = df_ready.set_index('q_phrase_id')
    pd.to_pickle(df_ready, global_settings.g_ks_graph_int_file_fmt.format(task_id))
    logging.debug('[build_ks_graph_single_proc] Proc %s: All done with %s ks_graph recs in %s secs.'
                  % (task_id, len(df_ready), time.time() - timer_start))


def build_ks_graph_multiproc(num_proc, query_id, ds_name):
    logging.debug('[build_ks_graph_multiproc] Starts.')
    timer_start = time.time()

    l_task_ids = [str(query_id) + '#' + str(idx) for idx in range(int(num_proc))]
    l_proc = []
    for task_id in l_task_ids:
        p = multiprocessing.Process(target=build_ks_graph_single_proc,
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
                logging.debug('[build_ks_graph_multiproc] %s is finished.' % p.name)
    logging.debug('[build_ks_graph_multiproc] All done in %s secs.' % str(time.time() - timer_start))


def merge_ks_graph_int(query_id, ds_name):
    logging.debug('[merge_ks_graph_int] Starts.')
    timer_start = time.time()

    d_ks_graph = dict()
    for (dirpath, dirname, filenames) in walk(global_settings.g_ks_graph_int_folder):
        for filename in filenames:
            if filename[-7:] != '.pickle' or filename[:13] != 'ks_graph_int_':
                continue
            df_ks_graph_int = pd.read_pickle(dirpath + filename)
            for q_phrase_id, ks_graph_rec in df_ks_graph_int.iterrows():
                # q_phrase_id = ks_graph_rec['q_phrase_id']
                ks_graph = ks_graph_rec['ks_graph']
                # ks_graph = nx.adjacency_graph(json.loads(ks_graph_json_str))
                if q_phrase_id not in d_ks_graph:
                    d_ks_graph[q_phrase_id] = [ks_graph]
                else:
                    d_ks_graph[q_phrase_id].append(ks_graph)

    l_ready = []
    for q_phrase_id in d_ks_graph:
        ks_graph_sum = divide_and_conquer_graph_compose(d_ks_graph[q_phrase_id])
        l_ready.append((q_phrase_id, ks_graph_sum))
        logging.debug('[merge_ks_graph_int] Done ks_graph_sum for %s in %s secs: %s'
                      % (q_phrase_id, time.time() - timer_start, nx.info(ks_graph_sum)))
    df_ready = pd.DataFrame(l_ready, columns=['q_phrase_id', 'ks_graph'])
    df_ready = df_ready.set_index('q_phrase_id')
    pd.to_pickle(df_ready, global_settings.g_ks_graph_file_fmt.format(ds_name + '_' + query_id))
    logging.debug('[merge_ks_graph_int] All done with %s ks_graph in %s secs.'
                  % (len(df_ready), time.time() - timer_start))


def ks_graph_phrase_clustering(q_phrase_id, ks_graph, d_phrase_id_to_phrase_row_id, d_phrase_row_id_to_phrase_id,
                               sp_phrase_sim_adj):
    logging.debug('[ks_graph_phrase_clustering] Q_Phrase %s: Starts.' % str(q_phrase_id))
    timer_start = time.time()

    l_phrase_id = list(ks_graph.nodes())
    l_phrase_row_id = [int(d_phrase_id_to_phrase_row_id[phrase_id]) for phrase_id in l_phrase_id]

    sp_sub_phrase_sim_adj = sp.vstack([sp_phrase_sim_adj.getrow(i) for i in l_phrase_row_id])
    sp_sub_phrase_sim_adj = sp.hstack([sp_sub_phrase_sim_adj.getcol(j) for j in l_phrase_row_id])
    logging.debug('[ks_graph_phrase_clustering] Q_Phrase %s: sp_sub_phrase_sim_adj done in %s secs, shape: %s'
                  % (q_phrase_id, time.time() - timer_start, sp_sub_phrase_sim_adj.shape))

    phrase_sim_graph = nx.from_scipy_sparse_matrix(sp_sub_phrase_sim_adj)
    logging.debug('[ks_graph_phrase_clustering] Q_Phrase %s: phrase_sim_graph done in %s secs: %s'
                  % (q_phrase_id, time.time() - timer_start, nx.info(phrase_sim_graph)))

    # CAUTION
    # clique_removal is implemented by a recursion. thus, it can easily explode the stack.
    # not suitable for large scale graphs.
    # _, l_phrase_cliques = clique_removal(phrase_sim_graph)
    l_phrase_cliques = []
    while len(phrase_sim_graph.nodes()) > 0:
        l_phrase_cliques += list(nx.find_cliques(phrase_sim_graph))
        l_done_node = []
        for phrase_clique in l_phrase_cliques:
            l_done_node += phrase_clique
        phrase_sim_graph.remove_nodes_from(set(l_done_node))
    logging.debug('[ks_graph_phrase_clustering] Q_Phrase %s: Obtain %s phrase cliques in %s secs.'
                  % (q_phrase_id, len(l_phrase_cliques), time.time() - timer_start))

    l_pcid_to_pid = []
    for clique_idx, phrase_clique in enumerate(l_phrase_cliques):
        cluster_id = q_phrase_id + '#' + str(clique_idx)
        l_cluster_member = \
            [d_phrase_row_id_to_phrase_id[str(l_phrase_row_id[phrase_row_id])] for phrase_row_id in phrase_clique]
        l_pcid_to_pid.append((cluster_id, l_cluster_member))
    df_pcid_to_pic = pd.DataFrame(l_pcid_to_pid, columns=['phrase_cluster_id', 'l_phrase_id'])
    df_pcid_to_pic = df_pcid_to_pic.set_index('phrase_cluster_id')

    d_pid_to_pcid = dict()
    for pcid, pc_member_rec in df_pcid_to_pic.iterrows():
        l_phrase_id = pc_member_rec['l_phrase_id']
        for phrase_id in l_phrase_id:
            if phrase_id not in d_pid_to_pcid:
                d_pid_to_pcid[phrase_id] = [pcid]
            else:
                d_pid_to_pcid[phrase_id].append(pcid)

    l_pid_to_pcid = []
    for pid in d_pid_to_pcid:
        l_pcid = d_pid_to_pcid[pid]
        l_pid_to_pcid.append((pid, l_pcid))
    df_pid_to_pcid = pd.DataFrame(l_pid_to_pcid, columns=['phrase_id', 'l_phrase_cluster_id'])
    df_pid_to_pcid = df_pid_to_pcid.set_index('phrase_id')

    logging.debug('[ks_graph_phrase_clustering] Q_Phrase %s: All done with %s clusters in %s secs.'
                  % (q_phrase_id, len(df_pcid_to_pic), time.time() - timer_start))
    return df_pcid_to_pic, df_pid_to_pcid


def ks_graph_contraction_single_proc(q_phrase_id, query_id, ds_name):
    logging.debug('[ks_graph_contraction_single_proc] Q_Phrase %s: Start.' % str(q_phrase_id))
    timer_start = time.time()

    with open(global_settings.g_phrase_row_id_to_phrase_id_file_fmt.format(ds_name), 'r') as in_fd:
        d_phrase_row_id_to_phrase_id = json.load(in_fd)
        in_fd.close()
    logging.debug('[ks_graph_contraction_single_proc] Q_Phrase %s: Load in d_phrase_row_id_to_phrase_id with %s recs.'
                  % (q_phrase_id, len(d_phrase_row_id_to_phrase_id)))

    d_phrase_id_to_phrase_row_id = dict()
    for phrase_row_id in d_phrase_row_id_to_phrase_id:
        d_phrase_id_to_phrase_row_id[d_phrase_row_id_to_phrase_id[phrase_row_id]] = phrase_row_id
    logging.debug('[ks_graph_contraction_single_proc] Q_Phrase %s: d_phrase_id_to_phrase_row_id done with %s recs '
                  'in %s secs.'
                  % (q_phrase_id, len(d_phrase_id_to_phrase_row_id), time.time() - timer_start))

    sp_phrase_sim_adj = sp.load_npz(global_settings.g_phrase_sim_graph_adj_file_fmt.format(ds_name))
    logging.debug('[ks_graph_contraction_single_proc] Q_Phrase %s: Load in np_phrase_sim_adj with shape %s'
                  % (q_phrase_id, sp_phrase_sim_adj.shape))

    df_ks_graph = pd.read_pickle(global_settings.g_ks_graph_file_fmt.format(ds_name + '_' + query_id))
    logging.debug('[ks_graph_contraction_single_proc] Q_Phrase %s: Load in df_ks_graph with %s recs.'
                  % (q_phrase_id, len(df_ks_graph)))

    ks_graph = df_ks_graph.loc[q_phrase_id]['ks_graph']
    df_pcid_to_pid, df_pid_to_pcid = ks_graph_phrase_clustering(q_phrase_id, ks_graph, d_phrase_id_to_phrase_row_id,
                                                                d_phrase_row_id_to_phrase_id, sp_phrase_sim_adj)
    pd.to_pickle(df_pcid_to_pid, global_settings.g_phrase_cluster_id_to_phrase_id_file_fmt
                 .format(ds_name + '_' + q_phrase_id))
    pd.to_pickle(df_pid_to_pcid, global_settings.g_phrase_id_to_phrase_cluster_id_file_fmt
                 .format(ds_name + '_' + q_phrase_id))
    logging.debug('[ks_graph_contraction_single_proc] Q_Phrase %s: Output df_pcid_to_pic and df_pid_to_pcid done.'
                  % str(q_phrase_id))

    ks_ctr_graph = nx.Graph()
    for node in ks_graph.nodes(data=True):
        phrase_id = node[0]
        phrase_str = node[1]['str']
        phrase_pos = node[1]['pos']
        l_pcid = df_pid_to_pcid.loc[phrase_id]['l_phrase_cluster_id']
        for pcid in l_pcid:
            if not ks_ctr_graph.has_node(pcid):
                ks_ctr_graph.add_node(pcid, str=[phrase_str[0]], pos=[phrase_pos[0]])
            else:
                ks_ctr_graph.nodes(data=True)[pcid]['str'] += phrase_str
                ks_ctr_graph.nodes(data=True)[pcid]['pos'] += phrase_pos

    for edge in ks_graph.edges(data=True):
        phrase_id_1 = edge[0]
        phrase_id_2 = edge[1]
        sent = edge[2]['sent']
        l_pcid_1 = df_pid_to_pcid.loc[phrase_id_1]['l_phrase_cluster_id']
        l_pcid_2 = df_pid_to_pcid.loc[phrase_id_2]['l_phrase_cluster_id']
        for pcid_1 in l_pcid_1:
            for pcid_2 in l_pcid_2:
                if pcid_1 == pcid_2:
                    continue
                if not ks_ctr_graph.has_edge(pcid_1, pcid_2):
                    avg_sent = np.asarray(sent[0][1]) if sent[0][1] is not None else None
                    ks_ctr_graph.add_edge(pcid_1, pcid_2, sent=[sent[0]], avg_sent=avg_sent)
                else:
                    ks_ctr_graph.edges[pcid_1, pcid_2]['sent'] += sent
                    l_sent_vec = [np.asarray(item[1]) for item in
                                  [sent for sent in ks_ctr_graph.edges[pcid_1, pcid_2]['sent']
                                   if sent[1] is not None]]
                    if len(l_sent_vec) > 0:
                        avg_sent = np.mean(l_sent_vec)
                        ks_ctr_graph.edges[pcid_1, pcid_2]['avg_sent'] = avg_sent

    df_ks_ctr_graph = pd.DataFrame([(q_phrase_id, ks_ctr_graph)], columns=['q_phrase_id', 'ks_ctr_graph'])
    pd.to_pickle(df_ks_ctr_graph, global_settings.g_ks_ctr_graph_int_file_fmt.format(ds_name + '_' + q_phrase_id))
    logging.debug('[ks_graph_contraction_single_proc] Q_Phrase %s: All done with ks_ctr_graph in %s secs: %s'
                  % (q_phrase_id, time.time() - timer_start, nx.info(ks_ctr_graph)))


def ks_graph_contraction_multiproc(query_id, ds_name):
    logging.debug('[ks_graph_contraction_for_query] Starts.')
    timer_start = time.time()

    df_ks_graph = pd.read_pickle(global_settings.g_ks_graph_file_fmt.format(ds_name + '_' + query_id))
    logging.debug('[ks_graph_contraction_for_query] Load in df_ks_graph with %s recs.' % str(len(df_ks_graph)))

    l_proc = []
    for q_phrase_id, ks_graph_rec in df_ks_graph.iterrows():
        p = multiprocessing.Process(target=ks_graph_contraction_single_proc,
                                    args=(q_phrase_id, query_id, ds_name),
                                    name='Proc ' + str(q_phrase_id))
        p.start()
        l_proc.append(p)

    while len(l_proc) > 0:
        for p in l_proc:
            if p.is_alive():
                p.join(1)
            else:
                l_proc.remove(p)
                logging.debug('[ks_graph_contraction_for_query] %s is finished.' % p.name)
    logging.debug('[ks_graph_contraction_for_query] All done in %s secs.' % str(time.time() - timer_start))


def draw_ks_graph_with_polarized_edges(query_id, ds_name):
    logging.debug('[draw_ks_graph_with_polarized_edges] Starts.')
    df_ks_graph = pd.read_pickle(global_settings.g_ks_graph_file_fmt.format(ds_name + '_' + query_id))
    logging.debug('[draw_ks_graph_with_polarized_edges] Load in df_ks_graph with %s recs.' % str(len(df_ks_graph)))

    def color_edge_by_sentiment(sent_vec):
        if sent_vec == None:
            return None
        sent_class = np.argmax(np.asarray(sent_vec))
        if sent_class == 0:
            return 'darkblue'
        elif sent_class == 1:
            return 'cornflowerblue'
        elif sent_class == 3:
            return 'lightcoral'
        elif sent_class == 4:
            return 'darkred'
        else:
            return None

    for q_phrase_id, ks_graph_rec in df_ks_graph.iterrows():
        # if q_phrase_id != 'q#0#0':
        #     continue
        ks_graph = df_ks_graph.loc[q_phrase_id]['ks_graph']
        logging.debug('[draw_ks_graph_with_polarized_edges] Load in ks_graph for %s' % q_phrase_id)

        l_edge = [edge for edge in ks_graph.edges()
                  if color_edge_by_sentiment(ks_graph.edges[edge]['sent'][0][1]) is not None]
        l_edge_color = [color_edge_by_sentiment(ks_graph.edges[edge]['sent'][0][1]) for edge in l_edge]

        l_node = []
        for edge in l_edge:
            l_node.append(edge[0])
            l_node.append(edge[1])
        l_node = list(set(l_node))

        ks_graph_draw = nx.subgraph(ks_graph, l_node)
        logging.debug('[draw_ks_graph_with_polarized_edges] ks_graph_draw done.')

        plt.figure(1, figsize=(100, 100), tight_layout={'pad': 1, 'w_pad': 50, 'h_pad': 50, 'rect': None})
        # ks_graph_layout = nx.spring_layout(ks_graph, k=1.0)
        ks_graph_layout = graphviz_layout(ks_graph_draw, prog="sfdp")
        logging.debug('[draw_ks_graph_with_polarized_edges] ks_graph_layout done.')

        nx.draw_networkx_edges(ks_graph_draw, pos=ks_graph_layout, edgelist=l_edge, edge_color=l_edge_color, width=2.0)
        logging.debug('[draw_ks_graph_with_polarized_edges] draw_networkx_edges done.')

        nx.draw_networkx_nodes(ks_graph_draw, pos=ks_graph_layout, nodelist=l_node, node_size=10)
        logging.debug('[draw_ks_graph_with_polarized_edges] draw_networkx_nodes done.')

        d_label = {node: ks_graph_draw.nodes(data=True)[node]['str'][0] for node in l_node}
        nx.draw_networkx_labels(ks_graph_draw, pos=ks_graph_layout, labels=d_label, font_size=20)
        logging.debug('[draw_ks_graph_with_polarized_edges] draw_networkx_labels done.')

        plt.savefig(global_settings.g_ks_graph_folder + 'polarized_ks_graph_{0}.png'
                    .format(ds_name + '_' + q_phrase_id), format="PNG")
        logging.debug('[draw_ks_graph_with_polarized_edges] Done for %s' % q_phrase_id)
        # plt.show()


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    cmd = sys.argv[1]

    # Example: "Wearing mask is helpful."
    if cmd == 'sim_phrases':
        ds_name = sys.argv[2]
        query_id = sys.argv[3]
        query_str = sys.argv[4]
        query_str_to_sim_phrases(query_id, query_str, ds_name)
    elif cmd == 'gen_ks_graph_tasks':
        # NOTE
        # Don't generate too many tasks as each task needs to consume a lot of memory.
        ds_name = sys.argv[2]
        query_id = sys.argv[3]
        num_tasks = sys.argv[4]
        gen_sim_phrase_to_ks_graph_tasks(query_id, ds_name, num_tasks)
    elif cmd == 'ks_graph':
        ds_name = sys.argv[2]
        query_id = sys.argv[3]
        num_proc = sys.argv[4]
        build_ks_graph_multiproc(num_proc, query_id, ds_name)
    elif cmd == 'merge_ks_graph':
        ds_name = sys.argv[2]
        query_id = sys.argv[3]
        merge_ks_graph_int(query_id, ds_name)
    elif cmd == 'ks_ctr_graph':
        ds_name = sys.argv[2]
        query_id = sys.argv[3]
        ks_graph_contraction_multiproc(query_id, ds_name)
    elif cmd == 'draw_ks_graph':
        ds_name = sys.argv[2]
        query_id = sys.argv[3]
        draw_ks_graph_with_polarized_edges(query_id, ds_name)
    elif cmd == 'test':
        q_phrase_id = 'q#0#0'
        ds_name = '202001'
        query_id = 'q#0'
        ks_graph_contraction_single_proc(q_phrase_id, query_id, ds_name)
