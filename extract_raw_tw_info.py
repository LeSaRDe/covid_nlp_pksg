import logging
from os import walk
import time
import json
import math
import multiprocessing
import sys

import pandas as pd

import global_settings


"""
Given a set of raw tweet objects, extract the following fields:
['tw_id', 'tw_type', 'tw_datetime', 'tw_lang', 'tw_usr_id', 'tw_src_id', 'tw_src_usr_id', 'tw_raw_txt', 'full_txt_flag']
'tw_id' is the index.
'full_txt_flag'=True : 'tw_raw_txt' takes from 'text'
'full_txt_flag'=Flase :  'tw_raw_txt' takes from 'full_text'
Output a pickle file for the given set. 
"""


def get_tw_id(tw_json):
    return tw_json['id_str']


def get_tw_type(tw_json):
    if 'in_reply_to_status_id_str' in tw_json \
            and tw_json['in_reply_to_status_id_str'] != '' \
            and not tw_json['in_reply_to_status_id_str'] is None:
        t_type = 'r'
    elif 'retweeted_status' in tw_json and tw_json['retweeted_status'] is not None:
        t_type = 't'
    elif 'quoted_status' in tw_json and tw_json['quoted_status'] is not None:
        t_type = 'q'
    else:
        t_type = 'n'
    return t_type


def get_tw_lang(tw_json):
    return tw_json['lang']


def get_tw_usr_id(tw_json):
    if 'user' not in tw_json:
        return None
    return tw_json['user']['id_str']


def get_tw_src_id(tw_json, tw_type):
    src_id = None
    if tw_type == 'n':
        return None
    elif tw_type == 'r' and 'in_reply_to_status_id_str' in tw_json:
        src_id = tw_json['in_reply_to_status_id_str']
    elif tw_type == 'q' and 'quoted_status_id_str' in tw_json:
        src_id = tw_json['quoted_status_id_str']
    elif tw_type == 't' and 'retweeted_status' in tw_json and 'id_str' in tw_json['retweeted_status']:
        src_id = tw_json['retweeted_status']['id_str']
    return src_id


def get_tw_src_usr_id(tw_json, tw_type):
    src_usr_id = None
    if tw_type == 'n':
        return None
    elif tw_type == 'q' and 'quoted_status' in tw_json and 'user' in tw_json['quoted_status']:
        src_usr_id = tw_json['quoted_status']['user']['id_str']
    elif tw_type == 'r':
        src_usr_id = tw_json['in_reply_to_user_id_str']
    elif tw_type == 'q' and 'retweeted_status' in tw_json and 'user' in tw_json['retweeted_status']:
        src_usr_id = tw_json['retweeted_status']['user']['id_str']
    return src_usr_id


def translate_month(month_str):
    month = None
    if month_str == 'Jan':
        month = '01'
    elif month_str == 'Feb':
        month = '02'
    elif month_str == 'Mar':
        month = '03'
    elif month_str == 'Apr':
        month = '04'
    elif month_str == 'May':
        month = '05'
    elif month_str == 'Jun':
        month = '06'
    elif month_str == 'Jul':
        month = '07'
    elif month_str == 'Aug':
        month = '08'
    elif month_str == 'Sep':
        month = '09'
    elif month_str == 'Oct':
        month = '10'
    elif month_str == 'Nov':
        month = '11'
    elif month_str == 'Dec':
        month = '12'
    else:
        raise Exception('Wrong month exists! user_time = %s' % month_str)
    return month


def get_tw_datetime(tw_json):
    '''
    Converte the datetime in the raw tweet object to the formart: YYYYMMDDHHMMSS
    '''
    if 'created_at' not in tw_json:
        return None
    date_fields = [item.strip() for item in tw_json['created_at'].split(' ')]
    mon_str = translate_month(date_fields[1])
    day_str = date_fields[2]
    year_str = date_fields[5]
    time_str = ''.join([item.strip() for item in date_fields[3].split(':')])
    return year_str + mon_str + day_str + time_str


def get_tw_raw_txt(tw_json, tw_type, tw_lang):
    '''
    Return (text, True/False for full_text)
    '''
    if tw_type == 'n' or tw_type == 'r' or tw_type == 'q':
        if tw_lang == 'en':
            if 'full_text' in tw_json:
                return (tw_json['full_text'], True)
            elif 'text' in tw_json:
                return (tw_json['text'], False)
            else:
                return None
    elif tw_type == 't':
        if tw_lang == 'en':
            if 'retweeted_status' in tw_json and 'full_text' in tw_json['retweeted_status']:
                return (tw_json['retweeted_status']['full_text'], True)
            elif 'retweeted_status' in tw_json and 'text' in tw_json['retweeted_status']:
                return (tw_json['retweeted_status']['text'], False)
            else:
                return None
    else:
        return None


def extract_raw_tw_info_single_proc(task_id):
    logging.debug('[extract_raw_tw_info_single_proc] Proc %s: Starts.' % str(task_id))
    timer_start = time.time()

    l_tw_json = []
    with open(global_settings.g_raw_tw_info_task_file_fmt.format(str(task_id)), 'r') as in_fd:
        for ln in in_fd:
            tw_str = ln.strip()
            tw_json = json.loads(tw_str)
            l_tw_json.append(tw_json)
        in_fd.close()
    logging.debug('[extract_raw_tw_info_single_proc] Proc %s: %s tw objs in total.' % (task_id, len(l_tw_json)))

    total_cnt = 0
    ready_cnt = 0
    l_ready = []
    for tw_json in l_tw_json:
        tw_id = get_tw_id(tw_json)
        tw_type = get_tw_type(tw_json)
        tw_datetime = get_tw_datetime(tw_json)
        tw_lang = get_tw_lang(tw_json)
        tw_usr_id = get_tw_usr_id(tw_json)
        tw_src_id = get_tw_src_id(tw_json, tw_type)
        tw_src_usr_id = get_tw_src_usr_id(tw_json, tw_type)
        tw_txt_ret = get_tw_raw_txt(tw_json, tw_type, tw_lang)
        if tw_txt_ret is None:
            tw_raw_txt = None
            full_txt_flag = False
        else:
            tw_raw_txt = tw_txt_ret[0]
            full_txt_flag = tw_txt_ret[1]
        if tw_id is None or tw_type is None or tw_datetime is None or tw_usr_id is None:
            logging.error('[extract_raw_tw_info_single_proc] Proc %s: Incorrect tw json: %s' % (task_id, str(tw_json)))
        else:
            l_ready.append((tw_id, tw_type, tw_datetime, tw_lang, tw_usr_id, tw_src_id, tw_src_usr_id,
                            tw_raw_txt, full_txt_flag))
            ready_cnt += 1
        total_cnt += 1
        if total_cnt % 5000 == 0 and total_cnt >= 5000:
            logging.debug('[extract_raw_tw_info_single_proc] Proc %s: total_cnt = %s ready_cnt = %s done in %s secs.'
                          % (task_id, total_cnt, ready_cnt, time.time() - timer_start))
    df_ready = pd.DataFrame(l_ready, columns=['tw_id', 'tw_type', 'tw_datetime', 'tw_lang', 'tw_usr_id', 'tw_src_id',
                                              'tw_src_usr_id', 'tw_raw_txt', 'full_txt_flag'])
    pd.to_pickle(df_ready, global_settings.g_raw_tw_info_int_file_fmt.format(task_id))
    logging.debug('[extract_raw_tw_info_single_proc] Proc %s: All done with %s recs in %s secs.'
                  % (task_id, len(df_ready), time.time() - timer_start))


def extract_raw_tw_info_multiproc(num_proc, job_id):
    '''
    Task ids are of the format: job_id#x
    where x ranges in [0, num_proc-1]
    !!! num_proc, job_id must be the same as those passed into gen_extract_raw_tw_info_tasks()
    '''
    logging.debug('[extract_raw_tw_info] Starts with %s procs on job %s.' % (int(num_proc), job_id))
    timer_start = time.time()

    l_task_ids = [str(job_id) + '#' + str(idx) for idx in range(int(num_proc))]
    l_proc = []
    for task_id in l_task_ids:
        p = multiprocessing.Process(target=extract_raw_tw_info_single_proc,
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
                logging.debug('[extract_raw_tw_info_multiproc] %s is finished.' % p.name)
    logging.debug('[extract_raw_tw_info_multiproc] All done in %s secs.' % str(time.time() - timer_start))


def gen_extract_raw_tw_info_tasks(num_proc, job_id):
    '''
    Task ids are of the format: job_id#x
    where x ranges in [0, num_proc-1]
    '''
    logging.debug('[gen_extract_raw_tw_info_tasks] Starts with %s procs on job %s.' % (num_proc, job_id))
    timer_start = time.time()

    done_cnt = 0
    l_tw_str = []
    for (dirpath, dirname, filenames) in walk(global_settings.g_tw_raw_data_input_folder):
        for filename in filenames:
            if filename[-5:] != '.json':
                continue
            with open(dirpath + filename, 'r') as in_fd:
                ln = in_fd.readline()
                l_tw_ln = json.loads(ln)
                in_fd.close()
            for tw_str in l_tw_ln:
                l_tw_str.append(tw_str)
            done_cnt += 1
            if done_cnt % 500 == 0 and done_cnt >= 500:
                logging.debug('[gen_extract_raw_tw_info_tasks] %s raw files scanned in %s secs.'
                              % (done_cnt, time.time() - timer_start))
    logging.debug('[gen_extract_raw_tw_info_tasks] %s raw files scanned in %s secs.'
                  % (done_cnt, time.time() - timer_start))
    num_tasks = len(l_tw_str)
    logging.debug('[gen_extract_raw_tw_info_tasks] %s tw objs.' % str(num_tasks))

    num_proc = int(num_proc)
    batch_size = math.ceil(num_tasks / num_proc)
    l_tasks = []
    for i in range(0, num_tasks, batch_size):
        if i + batch_size < num_tasks:
            l_tasks.append(l_tw_str[i:i + batch_size])
        else:
            l_tasks.append(l_tw_str[i:])

    l_task_ids = []
    for idx, task in enumerate(l_tasks):
        task_id = str(job_id) + '#' + str(idx)
        with open(global_settings.g_raw_tw_info_task_file_fmt.format(task_id), 'w+') as out_fd:
            out_str = '\n'.join(task)
            out_fd.write(out_str)
            out_fd.close()
        l_task_ids.append(task_id)
    logging.debug('[gen_extract_raw_tw_info_tasks] Task data ready.')


def merge_raw_tw_info_int(ds_name):
    logging.debug('[merge_raw_tw_info_int] Starts...')
    timer_start = time.time()

    l_int = []
    for (dirpath, dirname, filenames) in walk(global_settings.g_raw_tw_info_int_folder):
        for filename in filenames:
            if filename[-7:] != '.pickle':
                continue
            df_int = pd.read_pickle(dirpath + filename)
            l_int.append(df_int)
    logging.debug('[merge_raw_tw_info_int] Read in %s int dfs.' % str(len(l_int)))
    df_merge = pd.concat(l_int)
    df_merge = df_merge.set_index('tw_id')
    df_merge = df_merge[~df_merge.index.duplicated(keep='first')]
    pd.to_pickle(df_merge, global_settings.g_raw_tw_info_file_fmt.format(ds_name))
    logging.debug('[merge_raw_tw_info_int] All done in %s secs.' % str(time.time() - timer_start))


def simplify_retweet_txt(ds_name):
    logging.debug('[simplify_retweet_txt] Starts...')
    timer_start = time.time()

    df_raw_tw_info = pd.read_pickle(global_settings.g_raw_tw_info_file_fmt.format(ds_name))
    df_raw_tw_info['rt_ref'] = None
    logging.debug('[simplify_retweet_txt] Load in df_raw_tw_info with %s recs.' % str(len(df_raw_tw_info)))

    total_cnt = 0
    ref_cnt = 0
    for tw_id, raw_tw_info in df_raw_tw_info.iterrows():
        total_cnt += 1
        if total_cnt % 5000 == 0 and total_cnt >= 5000:
            logging.debug('[simplify_retweet_txt] %s recs scanned. ref_cnt = %s. in %s secs.'
                          % (total_cnt, ref_cnt, time.time() - timer_start))
        if raw_tw_info['tw_type'] != 't':
            continue
        tw_src_id = raw_tw_info['tw_src_id']
        if tw_src_id in df_raw_tw_info.index and not df_raw_tw_info.loc[tw_src_id]['tw_raw_txt'] is None:
            df_raw_tw_info.at[tw_id, 'tw_raw_txt'] = None
            df_raw_tw_info.at[tw_id, 'rt_ref'] = tw_src_id
            ref_cnt += 1
    logging.debug('[simplify_retweet_txt] %s recs scanned. ref_cnt = %s. in %s secs.'
                  % (total_cnt, ref_cnt, time.time() - timer_start))

    pd.to_pickle(df_raw_tw_info, global_settings.g_raw_tw_info_file_fmt.format(ds_name))
    logging.debug('[simplify_retweet_txt] All done in %s secs.' % str(time.time() - timer_start))



if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    cmd = sys.argv[1]

    if cmd == 'gen_tasks':
        num_proc = sys.argv[2]
        job_id = sys.argv[3]
        gen_extract_raw_tw_info_tasks(num_proc, job_id)
    elif cmd == 'extract_raw_tw_info':
        num_proc = sys.argv[2]
        job_id = sys.argv[3]
        extract_raw_tw_info_multiproc(num_proc, job_id)
    elif cmd == 'merge_int':
        ds_name = sys.argv[2]
        merge_raw_tw_info_int(ds_name)
    elif cmd == 'retweet_ref':
        ds_name = sys.argv[2]
        simplify_retweet_txt(ds_name)
