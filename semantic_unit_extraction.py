import logging
from os import walk
import time
import json
import math
import multiprocessing
import sys

import pandas as pd
import networkx as nx

import global_settings
from semantic_units_extractor import SemUnitsExtractor


def gen_sem_unit_extraction_tasks(ds_name, num_tasks, job_id):
    logging.debug('[gen_sem_unit_extraction_tasks] Starts.')
    timer_start = time.time()

    df_tw_clean_txt = pd.read_pickle(global_settings.g_tw_clean_file_fmt.format(ds_name))
    df_tw_clean_txt = df_tw_clean_txt.loc[df_tw_clean_txt['tw_clean_txt'].notnull()]
    num_clean_tw = len(df_tw_clean_txt)
    logging.debug('[gen_sem_unit_extraction_tasks] Load in %s tw clean texts.' % str(num_clean_tw))

    num_tasks = int(num_tasks)
    batch_size = math.ceil(num_clean_tw / num_tasks)
    l_tasks = []
    for i in range(0, num_clean_tw, batch_size):
        if i + batch_size < num_clean_tw:
            l_tasks.append(df_tw_clean_txt.iloc[i:i + batch_size])
        else:
            l_tasks.append(df_tw_clean_txt.iloc[i:])
    logging.debug('[gen_sem_unit_extraction_tasks] Need to generate %s tasks.' % str(len(l_tasks)))

    for idx, df_task in enumerate(l_tasks):
        pd.to_pickle(df_task, global_settings.g_tw_sem_unit_task_file_fmt.format(str(job_id) + '#' + str(idx)))
    logging.debug('[gen_sem_unit_extraction_tasks] All done with %s tasks in %s secs.'
                  % (len(l_tasks), time.time() - timer_start))


def sem_unit_extraction_single_proc(task_id):
    logging.debug('[sem_unit_extraction_single_proc] Proc %s: Starts.' % str(task_id))
    timer_start = time.time()

    df_tw_clean_txt = pd.read_pickle(global_settings.g_tw_sem_unit_task_file_fmt.format(task_id))
    logging.debug('[sem_unit_extraction_single_proc] Proc %s: Load in %s tw clean texts.'
                  % (task_id, len(df_tw_clean_txt)))
    l_tasks = []
    for tw_id, tw_clean_txt_rec in df_tw_clean_txt.iterrows():
        tw_clean_txt = tw_clean_txt_rec['tw_clean_txt']
        l_tasks.append((tw_id, tw_clean_txt))

    sem_unit_ext_ins = SemUnitsExtractor(global_settings.g_sem_units_extractor_config_file)
    sem_unit_ext_ins.sem_unit_extraction_thread(l_tasks, task_id, global_settings.g_tw_sem_unit_int_file_fmt, False)
    logging.debug('[sem_unit_extraction_single_proc] Proc %s: All done in %s secs.'
                  % (task_id, time.time() - timer_start))


def sem_unit_extraction_multiproc(num_proc, job_id):
    logging.debug('[sem_unit_extraction_multiproc] Starts.')
    timer_start = time.time()

    l_task_ids = [str(job_id) + '#' + str(idx) for idx in range(int(num_proc))]
    l_proc = []
    for task_id in l_task_ids:
        p = multiprocessing.Process(target=sem_unit_extraction_single_proc,
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
                logging.debug('[sem_unit_extraction_multiproc] %s is finished.' % p.name)
    logging.debug('[sem_unit_extraction_multiproc] All done in %s secs.' % str(time.time() - timer_start))


def merge_sem_unit_int(ds_name):
    logging.debug('[merge_sem_unit_int]')
    timer_start = time.time()

    l_int = []
    for (dirpath, dirname, filenames) in walk(global_settings.g_tw_sem_unit_int_folder):
        for filename in filenames:
            if filename[-7:] != '.pickle' or filename[:16] != 'tw_sem_unit_int_':
                continue
            df_int = pd.read_pickle(dirpath + filename)
            l_int.append(df_int)
    logging.debug('[merge_sem_unit_int] Read in %s int dfs.' % str(len(l_int)))
    df_merge = pd.concat(l_int)
    df_merge = df_merge.set_index('tw_id')
    pd.to_pickle(df_merge, global_settings.g_tw_sem_unit_file_fmt.format(ds_name))
    logging.debug('[merge_sem_unit_int] All done with %s recs in %s secs.'
                  % (len(df_merge), time.time() - timer_start))


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    cmd = sys.argv[1]

    if cmd == 'gen_tasks':
        ds_name = sys.argv[2]
        num_tasks = sys.argv[3]
        job_id = sys.argv[4]
        gen_sem_unit_extraction_tasks(ds_name, num_tasks, job_id)
    elif cmd == 'sem_unit':
        num_proc = sys.argv[2]
        job_id = sys.argv[3]
        sem_unit_extraction_multiproc(num_proc, job_id)
    elif cmd == 'merge_int':
        ds_name = sys.argv[2]
        merge_sem_unit_int(ds_name)
    elif cmd == 'test':
        sem_unit_ext_ins = SemUnitsExtractor(global_settings.g_sem_units_extractor_config_file)
        df_tw_clean_txt = pd.read_pickle(global_settings.g_tw_clean_file_fmt.format('202001'))
        test_txt = df_tw_clean_txt.loc['1222986721392046081']['tw_clean_txt']
        sem_unit_ext_ins.extract_sem_units_from_text(test_txt, 0)
