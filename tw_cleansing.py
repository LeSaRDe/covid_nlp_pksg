import logging
from os import walk
import time
import json
import math
import multiprocessing
import sys

import pandas as pd

import global_settings
from semantic_units_extractor import SemUnitsExtractor


"""
Given a pickle file produced by 'extract_raw_tw_info.py', clean each raw tweet text without sentence splitting.
Output a pickle file with 'tw_id' as the index and 'tw_clean_txt' carrying the cleaned text.
Also output a JSON file for the Java sentiment analysis. The JSON file itself is an array of JSON objects. And each
JSON object corresponds to a tweet containing 'tw_id' (String) and 'tw_clean_txt' (Strings). 
"""


def tw_clean_single_proc(task_id):
    logging.debug('[tw_clean_single_proc] Proc %s: Starts...' % str(task_id))
    timer_start = time.time()

    df_tw_clean_task = pd.read_pickle(global_settings.g_tw_clean_task_file_fmt.format(task_id))
    df_tw_clean_task['tw_clean_txt'] = None
    logging.debug('[tw_clean_single_proc] Proc %s: Load in %s tasks.' % (task_id, len(df_tw_clean_task)))
    sem_unit_ext_ins = SemUnitsExtractor(global_settings.g_sem_units_extractor_config_file)

    for tw_id, tw_clean_task in df_tw_clean_task.iterrows():
        tw_raw_txt = tw_clean_task['tw_raw_txt']
        tw_clean_txt = sem_unit_ext_ins.text_clean(tw_raw_txt)
        if not tw_clean_txt is None and tw_clean_txt != '':
            df_tw_clean_task.at[tw_id, 'tw_clean_txt'] = tw_clean_txt
    pd.to_pickle(df_tw_clean_task, global_settings.g_tw_clean_int_file_fmt.format(task_id))
    logging.debug('[tw_clean_single_proc] Proc %s: All done with %s recs in %s secs.'
                  % (task_id, len(df_tw_clean_task), time.time() - timer_start))


def tw_clean_multiproc(num_proc, job_id):
    '''
    'num_proc' = 'num_tasks'@gen_tw_clean_tasks
    'job_id' = 'job_id'@gen_tw_clean_tasks
    '''
    logging.debug('[tw_clean_multiproc] Starts')
    timer_start = time.time()

    l_task_ids = [str(job_id) + '#' + str(idx) for idx in range(int(num_proc))]
    l_proc = []
    for task_id in l_task_ids:
        p = multiprocessing.Process(target=tw_clean_single_proc,
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
                logging.debug('[tw_clean_multiproc] %s is finished.' % p.name)
    logging.debug('[tw_clean_multiproc] All done in %s secs.' % str(time.time() - timer_start))


def gen_tw_clean_tasks(ds_name, num_tasks, job_id):
    '''
    Exclusively consider English and non-empty texts.
    '''
    logging.debug('[gen_tw_clean_tasks] Starts with %s' % ds_name)
    timer_start = time.time()

    df_raw_tw_info = pd.read_pickle(global_settings.g_raw_tw_info_file_fmt.format(ds_name))
    logging.debug('[gen_tw_clean_tasks] Load in df_raw_tw_info with %s recs.' % str(len(df_raw_tw_info)))

    df_txt = df_raw_tw_info.loc[(df_raw_tw_info['tw_lang'] == 'en') & (df_raw_tw_info['tw_raw_txt'].notnull())]
    df_txt = df_txt[['tw_raw_txt']]
    num_txt = len(df_txt)
    logging.debug('[gen_tw_clean_tasks] Fetch %s valid texts in %s secs.' % (num_txt, time.time() - timer_start))

    num_tasks = int(num_tasks)
    batch_size = math.ceil(num_txt / num_tasks)
    l_tasks = []
    for i in range(0, num_txt, batch_size):
        if i + batch_size < num_txt:
            l_tasks.append(df_txt.iloc[i:i + batch_size])
        else:
            l_tasks.append(df_txt.iloc[i:])
    logging.debug('[gen_tw_clean_tasks] Need to generate %s tasks.' % str(len(l_tasks)))

    for idx, df_task in enumerate(l_tasks):
        pd.to_pickle(df_task, global_settings.g_tw_clean_task_file_fmt.format(str(job_id) + '#' + str(idx)))
    logging.debug('[gen_tw_clean_tasks] All done with %s tasks in %s secs.' % (len(l_tasks), time.time() - timer_start))


def merge_tw_clean_int(ds_name):
    logging.debug('[merge_tw_clean_int] Starts...')
    timer_start = time.time()

    l_int = []
    for (dirpath, dirname, filenames) in walk(global_settings.g_tw_clean_int_folder):
        for filename in filenames:
            if filename[-7:] != '.pickle' or filename[:13] != 'tw_clean_int_':
                continue
            df_int = pd.read_pickle(dirpath + filename)
            df_int = df_int[['tw_clean_txt']]
            l_int.append(df_int)
    logging.debug('[merge_tw_clean_int] Read in %s int dfs.' % str(len(l_int)))
    df_merge = pd.concat(l_int)
    pd.to_pickle(df_merge, global_settings.g_tw_clean_file_fmt.format(ds_name))

    # l_json_out = []
    # for tw_id, tw_clean_txt_rec in df_merge.iterrows():
    #     tw_clean_txt = tw_clean_txt_rec['tw_clean_txt']
    #     l_json_out.append({'tw_id': tw_id, 'tw_clean_txt': tw_clean_txt})
    # with open(global_settings.g_tw_clean_file_for_java_fmt.format(ds_name), 'w+') as out_fd:
    #     json.dump(l_json_out, out_fd)
    #     out_fd.close()
    logging.debug('[merge_tw_clean_int] All done in %s secs.' % str(time.time() - timer_start))


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    cmd = sys.argv[1]

    if cmd == 'gen_tasks':
        ds_name = sys.argv[2]
        num_tasks = sys.argv[3]
        job_id = sys.argv[4]
        gen_tw_clean_tasks(ds_name, num_tasks, job_id)
    elif cmd == 'tw_clean':
        num_proc = sys.argv[2]
        job_id = sys.argv[3]
        tw_clean_multiproc(num_proc, job_id)
    elif cmd == 'merge_int':
        ds_name = sys.argv[2]
        merge_tw_clean_int(ds_name)
    elif cmd == 'test':
        tw_clean_single_proc('0#5')
