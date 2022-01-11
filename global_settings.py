"""
INPUTS
"""
# g_tw_work_folder = '/scratch/mf3jh/data/covid/'
g_tw_work_folder = '/home/mf3jh/workspace/data/covid/'
g_tw_raw_data_folder = g_tw_work_folder + 'tw_raw_data/'
g_tw_raw_data_input_folder = g_tw_raw_data_folder + '202001/'

"""
GENERATED
"""
# raw tw info
g_raw_tw_info_folder = g_tw_work_folder + 'raw_tw_info/'
g_raw_tw_info_int_folder = g_raw_tw_info_folder + 'int/'
g_raw_tw_info_task_file_fmt = g_raw_tw_info_int_folder + 'raw_tw_info_task_{0}.txt'
g_raw_tw_info_int_file_fmt = g_raw_tw_info_int_folder + 'raw_tw_info_int_{0}.pickle'
g_raw_tw_info_file_fmt = g_raw_tw_info_folder + 'raw_tw_info_{0}.pickle'

# tw cleansing
g_tw_clean_folder = g_tw_work_folder + 'tw_clean/'
g_tw_clean_int_folder = g_tw_clean_folder + 'int/'
g_tw_clean_task_file_fmt = g_tw_clean_int_folder + 'tw_clean_task_{0}.pickle'
g_tw_clean_int_file_fmt = g_tw_clean_int_folder + 'tw_clean_int_{0}.pickle'
g_tw_clean_file_fmt = g_tw_clean_folder + 'tw_clean_txt_{0}.pickle'

# sentiments
g_tw_sent_folder = g_tw_work_folder + 'tw_sent/'
g_tw_sent_int_folder = g_tw_sent_folder + 'int/'
g_tw_sent_task_file_fmt = g_tw_sent_int_folder + 'tw_sent_task_{0}.json'
g_tw_sent_int_file_fmt = g_tw_sent_int_folder + 'tw_sent_int_{0}.json'
g_tw_sgraph_int_file_fmt = g_tw_sent_int_folder + 'tw_sgraph_int_{0}.pickle'
g_tw_sgraph_file_fmt = g_tw_sent_folder + 'tw_sgraph_{0}.pickle'
g_tw_phrase_sentiment_task_file_fmt = g_tw_sent_int_folder + 'tw_phrase_sentiment_task_{0}.pickle'
g_tw_phrase_sentiment_int_file_fmt = g_tw_sent_int_folder + 'tw_phrase_sentiment_int_{0}.pickle'
g_tw_phrase_sentiment_file_fmt = g_tw_sent_folder + 'tw_phrase_sentiment_{0}.pickle'
g_tw_phrase_sentiment_leftover_int_file_fmt = g_tw_sent_int_folder + 'tw_phrase_sentiment_leftover_int_{0}.pickle'
g_tw_phrase_sentiment_leftover_file_fmt = g_tw_sent_folder + 'tw_phrase_sentiment_leftover_{0}.pickle'

# semantic units
g_tw_sem_unit_folder = g_tw_work_folder + 'sem_unit/'
g_tw_sem_unit_int_folder = g_tw_sem_unit_folder + 'int/'
g_tw_sem_unit_task_file_fmt = g_tw_sem_unit_int_folder + 'tw_sem_unit_task_{0}.pickle'
g_tw_sem_unit_int_file_fmt = g_tw_sem_unit_int_folder + 'tw_sem_unit_int_{0}.pickle'
g_tw_sem_unit_file_fmt = g_tw_sem_unit_folder + 'tw_sem_unit_{0}.pickle'

# phrases
g_tw_phrase_folder = g_tw_work_folder + 'tw_phrase/'
g_tw_phrase_int_folder = g_tw_phrase_folder + 'int/'
g_tw_phrase_task_file_fmt = g_tw_phrase_int_folder + 'tw_phrase_task_{0}.pickle'
g_tw_phrase_int_file_fmt = g_tw_phrase_int_folder + 'tw_phrase_int_{0}.pickle'
g_tw_phrase_file_fmt = g_tw_phrase_folder + 'tw_phrase_{0}.pickle'
g_tw_to_phrase_id_file_fmt = g_tw_phrase_folder + 'tw_to_phrase_id_{0}.pickle'
g_phrase_id_to_phrase_str_file_fmt = g_tw_phrase_folder + 'phrase_id_to_phrase_str_{0}.pickle'
g_phrase_str_to_phrase_id_file_fmt = g_tw_phrase_folder + 'phrase_str_to_phrase_id_{0}.pickle'
g_phrase_id_to_tw_file_fmt = g_tw_phrase_folder + 'phrase_id_to_tw_{0}.pickle'

# phrase embedding
g_phrase_embed_folder = g_tw_work_folder + 'phrase_embed/'
g_phrase_embed_int_folder = g_phrase_embed_folder + 'int/'
g_phrase_embed_task_file_fmt = g_phrase_embed_int_folder + 'phrase_embed_task_{0}.pickle'
g_phrase_embed_int_file_fmt = g_phrase_embed_int_folder + 'phrase_embed_int_{0}.pickle'
g_phrase_embed_file_fmt = g_phrase_embed_folder + 'phrase_embed_{0}.pickle'
# g_tw_to_phrase_id_int_file_fmt = g_phrase_embed_int_folder + 'tw_to_phrase_id_int_{0}.pickle'
g_phrase_row_id_to_phrase_id_file_fmt = g_phrase_embed_folder + 'phrase_row_id_to_phrase_id_{0}.json'

# phrase clustering
g_phrase_cluster_folder = g_tw_work_folder + 'phrase_cluster/'
g_phrase_cluster_int_folder = g_phrase_cluster_folder + 'int/'
g_phrase_cluster_task_file_fmt = g_phrase_cluster_int_folder + 'phrase_cluster_task_{0}.pickle'
g_phrase_cluster_int_file_fmt = g_phrase_cluster_int_folder + 'phrase_cluster_int_{0}.pickle'
g_phrase_cluster_pid_to_cid_int_file_fmt = g_phrase_cluster_int_folder + 'pid_to_cid_int_{0}.pickle'
g_phrase_sim_task_file_fmt = g_phrase_cluster_int_folder + 'phrase_sim_task_{0}.txt'
g_phrase_sim_row_id_to_phrase_id_file_fmt = g_phrase_cluster_folder + 'phrase_sim_row_id_to_phrase_id_{0}.json'
g_phrase_sim_int_file_fmt = g_phrase_cluster_int_folder + 'phrase_sim_int_{0}.pickle'
g_phrase_sim_graph_int_file_fmt = g_phrase_cluster_int_folder + 'phrase_sim_graph_int_{0}.pickle'
g_phrase_sim_graph_adj_int_file_fmt = g_phrase_cluster_int_folder + 'phrase_sim_graph_adj_int_{0}.npz'
g_phrase_sim_graph_adj_file_fmt = g_phrase_cluster_folder + 'phrase_sim_graph_adj_{0}.npz'
g_phrase_sim_graph_file = g_phrase_cluster_int_folder + 'phrase_sim_graph_{0}.json'
g_phrase_cluster_file_fmt = g_phrase_cluster_folder + 'phrase_cluster_{0}.pickle'

# knowledge graph
g_ks_graph_folder = g_tw_work_folder + 'ks_graph/'
g_ks_graph_int_folder = g_ks_graph_folder + 'int/'
g_ks_graph_query_to_phrases_file_fmt = g_ks_graph_folder + 'query_to_phrases_{0}.pickle'
g_ks_graph_q_phrase_to_sim_phrase_file_fmt = g_ks_graph_folder + 'q_phrase_to_sim_phrase_{0}.pickle'
g_ks_graph_task_file_fmt = g_ks_graph_int_folder + 'ks_graph_task_{0}.pickle'
g_ks_graph_int_file_fmt = g_ks_graph_int_folder + 'ks_graph_int_{0}.pickle'
g_ks_graph_file_fmt = g_ks_graph_folder + 'ks_graph_{0}.pickle'
g_phrase_cluster_id_to_phrase_id_file_fmt = g_ks_graph_int_folder + 'pcid_to_pid_{0}.pickle'
g_phrase_id_to_phrase_cluster_id_file_fmt = g_ks_graph_int_folder + 'pid_to_pcid_{0}.pickle'
g_ks_ctr_graph_int_file_fmt = g_ks_graph_int_folder + 'ks_ctr_graph_int_{0}.pickle'

g_tw_pksg_task_file_fmt = g_ks_graph_int_folder + 'tw_pksg_task_{0}.txt'
g_tw_pksg_int_file_fmt = g_ks_graph_int_folder + 'tw_pksg_int_{0}.pickle'
g_merged_tw_pksg_task_file_fmt = g_ks_graph_int_folder + 'merged_tw_pksg_task_{0}.pickle'
g_merged_tw_pksg_int_file_fmt = g_ks_graph_int_folder + 'merged_tw_pksg_int_{0}.pickle'
g_merged_tw_pksg_file_fmt = g_ks_graph_folder + 'merged_tw_pksg_{0}.pickle'

# embedding adjustment
g_adj_embed_folder = g_tw_work_folder + 'adj_embed/'
g_adj_embed_file_fmt = g_adj_embed_folder + 'adj_ph_embed_{0}.pt'
g_adj_embed_dist_file_fmt = g_adj_embed_folder + 'adj_ph_embed_dist_{0}.npy'
g_adj_embed_samples_file_fmt = g_adj_embed_folder + 'adj_ph_embed_samples_{0}.pickle'

# curvature
g_curvature_folder = g_tw_work_folder + 'curvature/'
g_curvature_int_folder = g_curvature_folder + 'int/'
g_kstest_task_file_fmt = g_curvature_int_folder + 'kstest_task_{0}.pickle'
g_kstest_int_file_fmt = g_curvature_int_folder + 'kstest_int_{0}.pickle'


"""
CONFIGURATIONS
"""
g_sem_units_extractor_config_file = 'sem_units_ext.conf'
g_lexvec_model_folder = '/home/mf3jh/workspace/lib/lexvec/python/lexvec/'
g_lexvec_vect_file_path = '/home/mf3jh/workspace/lib/lexvec/lexvec.commoncrawl.ngramsubwords.300d.W.pos.bin'
g_phrase_sim_threshold = 0.8
g_phrase_embed_dim = 300


def env_check():
    import os
    from os import path
    if not path.exists(g_tw_work_folder):
        raise Exception('g_tw_work_folder does not exist!')
    if not path.exists(g_tw_raw_data_folder):
        raise Exception('g_tw_raw_data_folder does not exist!')
    if not path.exists(g_raw_tw_info_folder):
        os.mkdir(g_raw_tw_info_folder)
    if not path.exists(g_raw_tw_info_int_folder):
        os.mkdir(g_raw_tw_info_int_folder)
    if not path.exists(g_tw_clean_folder):
        os.mkdir(g_tw_clean_folder)
    if not path.exists(g_tw_clean_int_folder):
        os.mkdir(g_tw_clean_int_folder)
    if not path.exists(g_tw_sent_folder):
        os.mkdir(g_tw_sent_folder)
    if not path.exists(g_tw_sent_int_folder):
        os.mkdir(g_tw_sent_int_folder)
    if not path.exists(g_tw_sem_unit_folder):
        os.mkdir(g_tw_sem_unit_folder)
    if not path.exists(g_tw_sem_unit_int_folder):
        os.mkdir(g_tw_sem_unit_int_folder)
    if not path.exists(g_tw_phrase_folder):
        os.mkdir(g_tw_phrase_folder)
    if not path.exists(g_tw_phrase_int_folder):
        os.mkdir(g_tw_phrase_int_folder)
    if not path.exists(g_phrase_embed_folder):
        os.mkdir(g_phrase_embed_folder)
    if not path.exists(g_phrase_embed_int_folder):
        os.mkdir(g_phrase_embed_int_folder)
    if not path.exists(g_phrase_cluster_folder):
        os.mkdir(g_phrase_cluster_folder)
    if not path.exists(g_phrase_cluster_int_folder):
        os.mkdir(g_phrase_cluster_int_folder)
    if not path.exists(g_ks_graph_folder):
        os.mkdir(g_ks_graph_folder)
    if not path.exists(g_ks_graph_int_folder):
        os.mkdir(g_ks_graph_int_folder)
    if not path.exists(g_adj_embed_folder):
        os.mkdir(g_adj_embed_folder)
    if not path.exists(g_curvature_folder):
        os.mkdir(g_curvature_folder)
    if not path.exists(g_curvature_int_folder):
        os.mkdir(g_curvature_int_folder)



env_check()