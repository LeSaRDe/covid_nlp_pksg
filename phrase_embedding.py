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
from sklearn import preprocessing

import global_settings
sys.path.insert(1, global_settings.g_lexvec_model_folder)
import model as lexvec



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


def phrase_embedding_single_proc(task_id):
    logging.debug('[phrase_embedding_single_proc] Proc %s: Starts.' % str(task_id))
    timer_start = time.time()

    df_task = pd.read_pickle(global_settings.g_phrase_embed_task_file_fmt.format(task_id))
    logging.debug('[phrase_embedding_single_proc] Proc %s: Load in %s phrase recs.' % (task_id, len(df_task)))

    lexvec_model, embedding_len = load_lexvec_model()

    l_phrase_embed_rec = []
    phrase_cnt = 0
    for phrase_id, phrase_rec in df_task.iterrows():
        l_token = [token.strip() for token in phrase_rec['phrase_str'].split(' ')]
        phrase_embed = phrase_embedding(lexvec_model, embedding_len, l_token)
        l_phrase_embed_rec.append((phrase_id, phrase_embed))
        phrase_cnt += 1
        if phrase_cnt % 5000 == 0 and phrase_cnt >= 5000:
            logging.debug('[phrase_embedding_single_proc] Proc %s: %s phrase embeds done in %s secs.'
                          % (task_id, phrase_cnt, time.time() - timer_start))
    logging.debug('[phrase_embedding_single_proc] Proc %s: %s phrase embeds done in %s secs.'
                  % (task_id, phrase_cnt, time.time() - timer_start))

    df_phrase_embed = pd.DataFrame(l_phrase_embed_rec, columns=['phrase_id', 'phrase_embed'])
    pd.to_pickle(df_phrase_embed, global_settings.g_phrase_embed_int_file_fmt.format(task_id))
    logging.debug('[phrase_embedding_single_proc] Proc %s: All done with %s phrase embeds in %s secs.'
                  % (task_id, len(df_phrase_embed), time.time() - timer_start))


def phrase_embedding_multiproc(num_proc, job_id):
    logging.debug('[phrase_embedding_multiproc] Starts.')
    timer_start = time.time()

    l_task_ids = [str(job_id) + '#' + str(idx) for idx in range(int(num_proc))]
    l_proc = []
    for task_id in l_task_ids:
        p = multiprocessing.Process(target=phrase_embedding_single_proc,
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
                logging.debug('[phrase_embedding_multiproc] %s is finished.' % p.name)
    logging.debug('[phrase_embedding_multiproc] All done in %s secs.' % str(time.time() - timer_start))


def gen_phrase_embedding_tasks(ds_name, num_tasks, job_id):
    logging.debug('[gen_phrase_embedding_tasks] Starts.')
    timer_start = time.time()

    df_phrase_id_to_phrase_str = pd.read_pickle(global_settings.g_phrase_id_to_phrase_str_file_fmt.format(ds_name))
    num_phrase_rec = len(df_phrase_id_to_phrase_str)
    logging.debug('[gen_phrase_embedding_tasks] Load in %s recs.' % str(num_phrase_rec))

    num_tasks = int(num_tasks)
    batch_size = math.ceil(num_phrase_rec / num_tasks)
    l_tasks = []
    for i in range(0, num_phrase_rec, batch_size):
        if i + batch_size < num_phrase_rec:
            l_tasks.append(df_phrase_id_to_phrase_str.iloc[i:i + batch_size])
        else:
            l_tasks.append(df_phrase_id_to_phrase_str.iloc[i:])
    logging.debug('[gen_phrase_embedding_tasks] Need to generate %s tasks.' % str(len(l_tasks)))

    for idx, df_task in enumerate(l_tasks):
        pd.to_pickle(df_task, global_settings.g_phrase_embed_task_file_fmt.format(str(job_id) + '#' + str(idx)))
    logging.debug('[gen_phrase_embedding_tasks] All done with %s tasks in %s secs.'
                  % (len(l_tasks), time.time() - timer_start))


def merge_phrase_embedding_int(ds_name):
    logging.debug('[merge_phrase_embedding_int]')
    timer_start = time.time()

    l_int = []
    for (dirpath, dirname, filenames) in walk(global_settings.g_phrase_embed_int_folder):
        for filename in filenames:
            if filename[-7:] != '.pickle' or filename[:17] != 'phrase_embed_int_':
                continue
            df_int = pd.read_pickle(dirpath + filename)
            l_int.append(df_int)
    logging.debug('[merge_phrase_embedding_int] Read in %s int dfs.' % str(len(l_int)))
    df_merge = pd.concat(l_int)
    df_merge = df_merge.set_index('phrase_id')
    pd.to_pickle(df_merge, global_settings.g_phrase_embed_file_fmt.format(ds_name))
    logging.debug('[merge_phrase_embedding_int] All done with %s recs in %s secs.'
                  % (len(df_merge), time.time() - timer_start))


def phrase_row_id_to_phrase_id(ds_name):
    logging.debug('[phrase_row_id_to_phrase_id] Starts.')
    timer_start = time.time()

    df_phrase_embed = pd.read_pickle(global_settings.g_phrase_embed_file_fmt.format(ds_name))
    logging.debug('[phrase_row_id_to_phrase_id] Load in %s phrase embeds.' % str(len(df_phrase_embed)))
    d_row_id_to_phrase_id = dict()
    for row_id, phrase_id in enumerate(df_phrase_embed.index):
        d_row_id_to_phrase_id[row_id] = phrase_id

    with open(global_settings.g_phrase_row_id_to_phrase_id_file_fmt.format(ds_name), 'w+') as out_fd:
        json.dump(d_row_id_to_phrase_id, out_fd)
        out_fd.close()
    logging.debug('[phrase_row_id_to_phrase_id] All done in %s secs.' % str(time.time() - timer_start))


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    cmd = sys.argv[1]

    if cmd == 'gen_phrase_embed_tasks':
        ds_name = sys.argv[2]
        num_tasks = sys.argv[3]
        job_id = sys.argv[4]
        gen_phrase_embedding_tasks(ds_name, num_tasks, job_id)
    elif cmd == 'phrase_embed':
        num_proc = sys.argv[2]
        job_id = sys.argv[3]
        phrase_embedding_multiproc(num_proc, job_id)
    elif cmd == 'merge_phrase_embed_int':
        ds_name = sys.argv[2]
        merge_phrase_embedding_int(ds_name)
    elif cmd == 'phrase_row_id_to_phrase_id':
        ds_name = sys.argv[2]
        phrase_row_id_to_phrase_id(ds_name)
    elif cmd == 'test':
        lexvec_model, embed_len = load_lexvec_model()
        phrase_embed_1 = phrase_embedding(lexvec_model, embed_len, ['white', 'people'])
        phrase_embed_2 = phrase_embedding(lexvec_model, embed_len, ['assume'])
        from scipy.spatial.distance import cosine
        sim = 1.0 - cosine(phrase_embed_1, phrase_embed_2)
        print(sim)