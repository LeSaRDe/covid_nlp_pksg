import logging
import time
import json
import math
import sys

import pandas as pd

import global_settings


def gen_sentiment_tasks(ds_name, num_tasks, job_id):
    logging.debug('[gen_sentiment_tasks] Starts.')
    timer_start = time.time()

    df_tw_clean_txt = pd.read_pickle(global_settings.g_tw_clean_file_fmt.format(ds_name))
    df_tw_clean_txt = df_tw_clean_txt.loc[df_tw_clean_txt['tw_clean_txt'].notnull()]
    num_clean_tw = len(df_tw_clean_txt)
    logging.debug('[gen_sentiment_tasks] Load in %s tw clean texts.' % str(num_clean_tw))

    num_tasks = int(num_tasks)
    batch_size = math.ceil(num_clean_tw / num_tasks)
    l_tasks = []
    for i in range(0, num_clean_tw, batch_size):
        if i + batch_size < num_clean_tw:
            l_tasks.append(df_tw_clean_txt.iloc[i:i + batch_size])
        else:
            l_tasks.append(df_tw_clean_txt.iloc[i:])
    logging.debug('[gen_tw_clean_tasks] Need to generate %s tasks.' % str(len(l_tasks)))

    for idx, df_task in enumerate(l_tasks):
        l_json_out = []
        for tw_id, tw_clean_txt_rec in df_task.iterrows():
            tw_clean_txt = tw_clean_txt_rec['tw_clean_txt']
            l_json_out.append({'tw_id': tw_id, 'tw_clean_txt': tw_clean_txt})
        with open(global_settings.g_tw_sent_task_file_fmt.format(str(job_id) + '#' + str(idx)), 'w+') as out_fd:
            json.dump(l_json_out, out_fd)
            out_fd.close()
    logging.debug('[gen_tw_clean_tasks] All done with %s tasks in %s secs.'
                  % (len(l_tasks), time.time() - timer_start))


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    cmd = sys.argv[1]

    if cmd == 'gen_tasks':
        ds_name = sys.argv[2]
        num_tasks = sys.argv[3]
        job_id = sys.argv[4]
        gen_sentiment_tasks(ds_name, num_tasks, job_id)
    elif cmd == 'test':
        text_bulk = "The IndianArmy has prepared a quarantine facility in Manesar for around 300 students, who are expected to reach Delhi on Saturday after they are airlifted from the coronavirus-hit Wuhan. The students will be monitored there for a few weeks."
        # sem_unit_ext_ins = SemUnitsExtractor(global_settings.g_sem_units_extractor_config_file)
        # sentence_split(sem_unit_ext_ins, text_bulk)