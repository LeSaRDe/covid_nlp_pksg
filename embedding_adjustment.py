import logging
import math
import time
import sys
from os import walk

import scipy as sp
from scipy import sparse
import numpy as np
import torch as th
import torch.nn.functional as F
import networkx as nx
from sklearn import preprocessing
import pandas as pd

import global_settings


def retrieve_orig_embed_for_pksg(pksg, ds_name):
    logging.debug('[retrieve_orig_embed_for_pksg] starts.')
    timer_start = time.time()

    df_phrase_embed = pd.read_pickle(global_settings.g_phrase_embed_file_fmt.format(ds_name))
    logging.debug('[retrieve_orig_embed_for_pksg] load in df_phrase_embed with %s recs in %s secs.'
                  % (len(df_phrase_embed), time.time() - timer_start))

    pksg_nodes = pksg.nodes()
    df_phrase_embed = df_phrase_embed.reindex(pksg_nodes)
    np_orig_ph_embed = np.stack(df_phrase_embed['phrase_embed'].to_list())
    logging.debug('[retrieve_orig_embed_for_pksg] all done with np_orig_ph_embed in %s secs: %s'
                  % (time.time() - timer_start, np_orig_ph_embed.shape))
    return np_orig_ph_embed


def compute_measure_matrix_from_pksg(pksg, tau):
    logging.debug('[compute_measure_matrix_from_pksg] starts.')
    timer_start = time.time()

    A = nx.linalg.adjacency_matrix(pksg)
    A = preprocessing.normalize(A, axis=1, norm='l1')
    A = (1-tau) * A
    A = A.astype(np.float32)
    T = sparse.diags([tau], shape=A.shape, dtype=np.float32)
    M = T + A
    logging.debug('[compute_measure_matrix_from_pksg] all done with M in %s secs: %s'
                  % (time.time() - timer_start, M.shape))
    return M


def custom_cosine_loss(t_adj_embed_t, t_adj_emebd_t_1):
    t_adj_embed_t_norm = th.nn.functional.normalize(t_adj_embed_t)
    t_adj_embed_t_1_norm = th.nn.functional.normalize(t_adj_emebd_t_1)
    # cos sim
    t_ret = th.einsum('ij..., ij...->i', t_adj_embed_t_norm, t_adj_embed_t_1_norm)
    # cos dis
    t_ret = 1.0 - t_ret
    # low bound
    t_ret[t_ret < 0.0] = 0.0
    # up bound
    t_ret[t_ret > 2.0] = 2.0
    # t_ret = th.abs(1.0 - t_ret)
    t_ret = th.mean(t_ret)
    return t_ret


def adjust_embed(np_measure_mat, np_orig_embed, max_epoch, term_threshold, use_cuda=True):
    logging.debug('[adjust_embed] starts.')
    timer_start = time.time()

    th.autograd.set_detect_anomaly(True)
    np_measure_mat = sparse.coo_matrix(np_measure_mat)
    np_measure_mat_values = np_measure_mat.data
    np_measure_mat_indices = np.vstack((np_measure_mat.row, np_measure_mat.col))
    t_measure_mat_indices = th.LongTensor(np_measure_mat_indices)
    t_measure_mat_values = th.FloatTensor(np_measure_mat_values)
    if use_cuda:
        t_measure_mat = th.sparse.FloatTensor(t_measure_mat_indices, t_measure_mat_values, th.Size(np_measure_mat.shape)).to('cuda')
    else:
        t_measure_mat = th.sparse.FloatTensor(t_measure_mat_indices, t_measure_mat_values, th.Size(np_measure_mat.shape))
    t_measure_mat.requires_grad = False
    if use_cuda:
        t_adj_embed = th.from_numpy(np_orig_embed).to('cuda')
    else:
        t_adj_embed = th.from_numpy(np_orig_embed)
    t_adj_embed = th.nn.functional.normalize(t_adj_embed)
    t_adj_embed.requires_grad = True

    # we compute the values in t_adj_embed from the backpropagation.

    # Potential choices: Adagrad > Adamax > AdamW > Adam
    optimizer = th.optim.Adagrad([t_adj_embed])
    # SGD may have much higher GPU memory consumption and need more epochs to converge
    # optimizer = th.optim.SGD([t_adj_embed], lr=0.1)

    for i in range(max_epoch):
        optimizer.zero_grad()
        t_adj_embed_t_1 = th.matmul(t_measure_mat, t_adj_embed)
        cos_loss = custom_cosine_loss(t_adj_embed, t_adj_embed_t_1)
        total_loss = cos_loss
        logging.debug('[adjust_embed] epoch %s: total loss = %s' % (i, total_loss))
        if total_loss <= term_threshold:
            logging.debug('[adjust_embed] done at epoch %s in %s secs.' % (i, time.time() - timer_start))
            return t_adj_embed
        total_loss.backward(create_graph=True)
        optimizer.step()
    t_adj_embed = th.nn.functional.normalize(t_adj_embed)
    logging.debug('[adjust_embed] done when out of epoches in %s secs: %s'
                  % (time.time() - timer_start, t_adj_embed.shape))
    return t_adj_embed


def adjust_embed_wrapper(ds_name, l_tau, max_epoch, term_threshold, use_cuda=True):
    logging.debug('[adjust_embed_wrapper] starts.')
    timer_start = time.time()

    pksg = nx.read_gpickle(global_settings.g_merged_tw_pksg_file_fmt.format(ds_name))
    logging.debug('[adjust_embed_wrapper] load in pksg in %s secs: %s'
                  % (time.time() - timer_start, nx.info(pksg)))

    np_orig_ph_embed = retrieve_orig_embed_for_pksg(pksg, ds_name)

    for tau in l_tau:
        logging.debug('[adjust_embed_wrapper] running with tau = %s' % str(tau))
        np_measure_mat = compute_measure_matrix_from_pksg(pksg, tau)
        t_adj_embed = adjust_embed(np_measure_mat, np_orig_ph_embed, max_epoch, term_threshold, use_cuda)
        save_name = ds_name + '#' + str(tau)
        th.save(t_adj_embed, global_settings.g_adj_embed_file_fmt.format(save_name))
        logging.debug('[adjust_embed_wrapper] done with tau = %s in %s secs.' % (tau, time.time() - timer_start))

    logging.debug('[adjust_embed_wrapper] all done in %s secs.' % str(time.time() - timer_start))


def compute_adjusted_embedding_distributions(ds_name):
    logging.debug('[compute_adjusted_embedding_distributions] starts.')
    timer_start = time.time()

    l_adj_embed = []
    for (dirpath, dirname, filenames) in walk(global_settings.g_adj_embed_folder):
        for filename in filenames:
            if filename[-3:] != '.pt' or filename[:13] != 'adj_ph_embed_':
                continue
            t_adj_embed = th.load(dirpath + filename)
            logging.debug('[compute_adjusted_embedding_distributions] load in %s: %s in %s secs.'
                          % (filename, t_adj_embed.shape, time.time() - timer_start))
            l_adj_embed.append(t_adj_embed.cpu().detach().numpy())
    logging.debug('[compute_adjusted_embedding_distributions] load in %s adjusted embeddings.' % str(len(l_adj_embed)))

    np_adj_embed_mean = np.mean(l_adj_embed, axis=0)
    np_adj_embed_std = np.std(l_adj_embed, axis=0)
    np_adj_embed_dist = np.stack([np_adj_embed_mean, np_adj_embed_std], axis=2)
    logging.debug('[compute_adjusted_embedding_distributions] done with np_adj_embed_dist: %s in %s secs.'
                  % (np_adj_embed_dist.shape, time.time() - timer_start))
    np.save(global_settings.g_adj_embed_dist_file_fmt.format(ds_name), np_adj_embed_dist)
    logging.debug('[compute_adjusted_embedding_distributions] all done in %s secs.' % str(time.time() - timer_start))


def extend_adj_embed_dist_to_samples(ds_name, sample_size, num_batch):
    logging.debug('[extend_adj_embed_dist_to_samples] starts.')
    timer_start = time.time()

    np_adj_embed_dist = np.load(global_settings.g_adj_embed_dist_file_fmt.format(ds_name))
    logging.debug('[extend_adj_embed_dist_to_samples] load in np_adj_embed_dist: %s in %s secs.'
                  % (np_adj_embed_dist.shape, time.time() - timer_start))

    pksg = nx.read_gpickle(global_settings.g_merged_tw_pksg_file_fmt.format(ds_name))
    logging.debug('[extend_adj_embed_dist_to_samples] load in pksg: %s in %s secs.'
                  % (nx.info(pksg), time.time() - timer_start))

    if pksg.number_of_nodes() != np_adj_embed_dist.shape[0]:
        raise Exception('[extend_adj_embed_dist_to_samples] np_adj_embed_dist does not match pksg.')

    l_pksg_nodes = list(pksg.nodes)

    rng = np.random.default_rng()
    batch_size = math.ceil(np_adj_embed_dist.shape[0] / num_batch)
    cnt = 0
    batch_id = 0
    l_np_dim_sample = []
    for ph_row_id, ph_dist in enumerate(np_adj_embed_dist):
        # ph_dist should be (300, 2)
        l_ph_dim_sample = []
        for ph_dim_dist in ph_dist:
            # ph_dim_dist should be (2,)
            ph_dim_sample = rng.normal(loc=ph_dim_dist[0], scale=ph_dim_dist[1], size=sample_size)
            ph_dim_sample = ph_dim_sample.astype(np.float32)
            l_ph_dim_sample.append(ph_dim_sample)
        np_dim_sample = np.stack(l_ph_dim_sample)
        l_np_dim_sample.append((l_pksg_nodes[ph_row_id], np_dim_sample))
        cnt += 1
        if cnt % batch_size == 0 and cnt >= batch_size:
            df_np_dim_sample = pd.DataFrame(l_np_dim_sample, columns=['pksg_node_id', 'np_dim_sample'])
            df_np_dim_sample.to_pickle(global_settings.g_adj_embed_samples_file_fmt
                                       .format(ds_name + '#' + str(sample_size) + '_' + str(batch_id)))

            logging.debug('[extend_adj_embed_dist_to_samples] batch %s df_np_dim_sample: %s done in %s secs.'
                          % (batch_id, len(df_np_dim_sample), time.time() - timer_start))
            batch_id += 1
            l_np_dim_sample = []
            df_np_dim_sample = None
        # if cnt % 10000 == 0 and cnt >= 10000:
        #     logging.debug('[extend_adj_embed_dist_to_samples] %s np_dim_sample done in %s secs.'
        #                   % (cnt, time.time() - timer_start))

    if len(l_np_dim_sample) > 0:
        df_np_dim_sample = pd.DataFrame(l_np_dim_sample, columns=['pksg_node_id', 'np_dim_sample'])
        df_np_dim_sample.to_pickle(global_settings.g_adj_embed_samples_file_fmt
                                   .format(ds_name + '#' + str(sample_size) + '_' + str(batch_id)))
        logging.debug('[extend_adj_embed_dist_to_samples] batch %s df_np_dim_sample: %s done in %s secs.'
                      % (batch_id, len(df_np_dim_sample), time.time() - timer_start))

    # np_adj_embed_sample = np.stack(l_np_dim_sample)
    # logging.debug('[extend_adj_embed_dist_to_samples] done with np_adj_embed_sample: %s in %s secs.'
    #               % (np_adj_embed_sample.shape, time.time() - timer_start))
    # np.save(global_settings.g_adj_embed_samples_file_fmt(ds_name + '#' + str(sample_size)))
    logging.debug('[extend_adj_embed_dist_to_samples] all done in %s secs.' % str(time.time() - timer_start))


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    cmd = sys.argv[1]

    if cmd == 'adj_embed':
        # this procedure is memory consuming. if GPU is in use, it may ran out of GPU memory.
        # running this with pure CPU is somewhat slower than GPU, but it can run through.
        ds_name = sys.argv[2]
        tau_stride = float(sys.argv[3])
        max_epoch = int(sys.argv[4])
        term_threshold = float(sys.argv[5])
        use_cuda = bool(sys.argv[6])
        tau = sys.argv[7]
        l_tau = [np.round(float(tau), 2)]
        # l_tau = np.round(np.arange(0.0, 1.0, tau_stride), 2)
        adjust_embed_wrapper(ds_name, l_tau, max_epoch, term_threshold, use_cuda)
    elif cmd == 'adj_embed_dist':
        ds_name = sys.argv[2]
        compute_adjusted_embedding_distributions(ds_name)
    elif cmd == 'adj_embed_samples':
        ds_name = sys.argv[2]
        sample_size = int(sys.argv[3])
        num_batch = int(sys.argv[4])
        extend_adj_embed_dist_to_samples(ds_name, sample_size, num_batch)
