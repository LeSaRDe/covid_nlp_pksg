'''
OBJECTIVES:
    Define the class of semantic units extractor.
'''
import logging
import json
import re
from os import path, walk
from collections import deque
import threading
import math
import time
import sqlite3

import spacy
from gensim.parsing import preprocessing
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd

import sd_2_usd



class SemUnitsExtractor:
    def __init__(self, config_file_path):
        if config_file_path is None or config_file_path == '':
            raise Exception('[SemUnitsExtractor:__init__] Cannot find configuration file!')
        with open(config_file_path, 'r') as in_fd:
            d_conf = json.load(in_fd)
            in_fd.close()
        try:
            self.m_conf_spacy_model_name = d_conf['spacy_model']
            self.m_conf_stopwords_path = d_conf['stopwords_path']
            self.m_conf_spacy_coref_greedyness = float(d_conf['spacy_coref_greedyness'])
            self.m_s_ner_tags = set(d_conf['ner'])
            self.m_s_core_deps = set(d_conf['core_dep'])
            self.m_b_keep_neg = bool(d_conf['keep_neg'])
            self.m_s_neg_cues = set(d_conf['neg_cue'])
            self.m_spacy_model_ins = self.spacy_init(self.m_conf_spacy_model_name, self.m_conf_spacy_coref_greedyness)
            self.m_s_stopwords = self.load_stopwords(self.m_conf_stopwords_path)
            self.m_d_sd_2_usd = sd_2_usd.g_d_sd_to_usd
        except Exception as err:
            logging.error('[SemUnitsExtractor:__init__] ' + err)
            return

    def spacy_init(self, spacy_model_name, coref_greedyness):
        '''
        Load the spaCy model with a specific name.
        :param coref_greedyness:
        :return:
        '''
        spacy_model = spacy.load(spacy_model_name)
        # neuralcoref.add_to_pipe(spacy_model, greedyness=coref_greedyness)
        return spacy_model

    def load_stopwords(self, stopwords_path):
        if not path.exists(stopwords_path):
            raise Exception('[SemUnitsExtractor:__init__] No stopwords file!')
        s_stopwords = set([])
        with open(stopwords_path, 'r') as in_fd:
            ln = in_fd.readline()
            while ln:
                sw = ln.strip()
                s_stopwords.add(sw)
                ln = in_fd.readline()
            in_fd.close()
        return s_stopwords

    def word_clean(self, word):
        '''
        Clean the input token.
        1. Remove all non-word characters
        2. Remove HTML and other tags
        3. Remove punctuations
        4. Remove unnecessary white spaces
        5. Remove numbers
        6. Remove short tokens
        :param
            word: A token.
        :return:
            A cleaned token.
        '''
        clean_text = re.sub(r'[^\w\s]', ' ', word)
        clean_text = preprocessing.strip_tags(clean_text)
        clean_text = preprocessing.strip_punctuation(clean_text)
        clean_text = preprocessing.strip_multiple_whitespaces(clean_text)
        clean_text = preprocessing.strip_numeric(clean_text)
        clean_text = preprocessing.strip_short(clean_text, minsize=3)
        return clean_text

    def is_trivial_token(self, spacy_token):
        '''
        Determine if the input token is trivial.
        NOTE:
        This method is not perfect. Any token in a sentence with a capped leading letter can be recognized as an
        entity. In this case, this method cannot detect if the token is trivial.
        NOTE:
        Before calling this function, the dep of the input token should be changed to neg if it is a negation token.
        TODO:
        Any better method to address the above issue?
        How should we deal with special entities? A capped token can be easily a false special entity.
        :param
            spacy_token: A spaCy annotated token.
        :return:
            True -- trivial
            False -- non-trivial
        '''
        clean_token = self.word_clean(spacy_token.lemma_)
        if clean_token == '': #or not clean_token.isascii():
            return True
        if self.m_b_keep_neg:
            if (spacy_token.is_stop or clean_token.lower() in self.m_s_stopwords) and spacy_token.dep_ != 'neg':
                return True
        else:
            if (spacy_token.is_stop or clean_token.lower() in self.m_s_stopwords):
                return True
        return False

    def spacy_pipeline_parse(self, raw_text):
        '''
        Obtain POS, NER, Dep Parse Trees, Noun Chunks and other linguistic features from the text. The text is firstly
        segmented into sentences, and then parsed. spaCy uses Universal Dependencies and POS tags.
        :param
            text: a raw text that may contains multiple sentences.
        :return:
            A list of tagged sentences, and the parsed doc.
        TODO:
            SpaCy 2.3 seems not so right working with Python 3.8. Also, the 'pipe' method in 'Language' returns a
            Doc generator rather than a straight Doc, which makes it awkward to extract sentences. If we don't use
            'pipe', we lose our chance to take advantage of 'n_process' (v2.3) or 'n_threads' (v2.2). As an alternative,
            we multiprocess our sem unit extraction procedure, and each proc runs with a single proc SpaCy instance.
            We may need to test the compatibility of SpaCy furthermore and find better solution of parallelism.
        '''
        if raw_text is None:
            return None, None
        # l_sents = []
        doc = self.m_spacy_model_ins(raw_text, disable=['ner'])
        # parse_doc_gen = self.m_spacy_model_ins.pipe(raw_text, disable=['ner'], n_process=10)
        # for doc in parse_doc_gen:
        #     l_sents += list(doc.sents)
        # doc = self.m_spacy_model_ins.pipe(raw_text, disable=['ner'], n_process=10)
        return list(doc.sents), doc

    # @profile
    def spacy_extract_nps_from_sent(self, spacy_sent):
        '''
        Extract noun phrases from a spaCy parsed sentence. The noun phrases contain non-stopwords and lemmas. Also, each
        lemma in a noun phrase appears only once.
        :param
            spacy_sent: A spaCy parsed sentence. Should not be None.
        :return:
            A list of noun phrase tuples. Each tuple = (noun phrase string, token index set, start_char_pos, end_char_pos)
        '''
        l_nps = []
        for noun_phrase in spacy_sent.noun_chunks:
            l_sal_lemmas = []
            s_np_idx = set([])
            for word in noun_phrase:
                word_lemma = self.word_clean(word.lemma_)
                if not self.is_trivial_token(word) \
                        and word_lemma not in l_sal_lemmas:
                    l_sal_lemmas.append(word_lemma)
                    s_np_idx.add(word.i)
            sal_np = ' '.join(l_sal_lemmas)
            if sal_np != '' and sal_np != ' ' and sal_np not in [item[1] for item in l_nps]:
                l_nps.append((sal_np, s_np_idx, noun_phrase.start_char, noun_phrase.end_char))
        return l_nps

    def spacy_extract_npool_from_sent(self, spacy_sent):
        '''
        Extract non-trivial nouns from a spacy parsed sentence. Stopwords are removed. All resulting nouns are converted to
        lemmas.
        :param
            spacy_sent: A spacy parsed sentence. Should not be None.
        :return:
            A set of non-trivial nouns.
        '''
        s_npool = set([])
        for token in spacy_sent:
            if token.pos_ == 'NOUN' and not token.is_stop and token.lemma_ not in self.m_s_stopwords:
                s_npool.add(token.lemma_)
        return s_npool

    def token_to_np(self, spacy_token, l_nps):
        '''
        Substitude a token with a noun phrase if any. All inputs should be the same sentence.
        :param
            spacy_token: a spaCy token.
            l_nps: a list of noun phrase tuples.
        :return:
            if the input token in contained in a noun phrase, then return the noun phrase string with 'NOUN' as its
            POS tag.
            otherwise, return the cleaned token with its original POS tag.
        '''
        if spacy_token is None:
            raise Exception('[SemUnitsExtractor:token_to_np] spacy_token is None!')
        if len(l_nps) == 0:
            clean_token = self.word_clean(spacy_token.lemma_)
            return clean_token, spacy_token.pos_, spacy_token.idx, spacy_token.idx + len(spacy_token)
        ret_str = self.word_clean(spacy_token.lemma_)
        token_idx = spacy_token.i
        for np_tpl in l_nps:
            if token_idx in np_tpl[1]:
                ret_str = np_tpl[0]
                return ret_str, 'NOUN', np_tpl[2], np_tpl[3]
        return ret_str, spacy_token.pos_, spacy_token.idx, spacy_token.idx + len(spacy_token)

    def neg_tag_substitute(self, spacy_sent):
        '''
        Substitute the dependency tags of negation tokens to 'neg'. Due to the imperfection of dependency parsers,
        some negation tokens, e.g. 'neither', may not be tagged as 'neg'. We fix this. The negation token list comes
        from the paper 'Negation Scope Detection for Twitter Sentiment Analysis'. Also, we add 'noone', 'couldnt',
        'wont' and 'arent' that are not included in this paper but in "Sentiment Symposium Tutorial" by Potts 2011.
        :param
            spacy_sent: A spaCy parsed sentence.
        :return:
            A modified spaCy sentence with all considered negation tokens tagged as 'neg'.
        '''
        for token in spacy_sent:
            if token.text in self.m_s_neg_cues:
                token.dep_ = 'neg'
        return spacy_sent

    def build_node_id(self, sent_id, node):
        return sent_id + '|' + node

    def find_nearest_ancestor_in_nx_graph(self, nx_dep_tree, sent_id, spacy_token):
        conj_jump = False
        if spacy_token.dep_ == 'conj':
            conj_jump = True
        parent = spacy_token.head
        while parent != spacy_token:
            clean_parent = self.word_clean(parent.lemma_)
            clean_parent_id = self.build_node_id(sent_id, clean_parent)
            if clean_parent_id in nx_dep_tree.nodes:
                if conj_jump:
                    spacy_token = parent
                    parent = spacy_token.head
                    conj_jump = False
                    continue
                return clean_parent
            else:
                spacy_token = parent
                parent = spacy_token.head
        return None

    # @profile
    def extract_cls_from_sent(self, sent_id, spacy_sent, l_nps):
        '''
        Extract core clause structures from a spacy parsed sentence. Each structure is represented as a graph (a tree
        without root). Each graph contains salient clause structures led by verbs. These structures are inspired by the
        five clause structures of simple sentences:
          1. <intransitive_verb subj>
          2. <transitive_verb subj obj>
          3. <transitive_verb subj iobj dobj>
          4. <subj subj_comp>
          5. <transitive_verb subj obj obj_comp>
        The dependency relations involved in these clause structures are various, and are (partially) enumerated in
        'g_s_core_deps'. For tokens in a noun phrase, substitute the noun phrase for the tokens. We always add lemmas
        as vertices into the resulting graph.
        NOTE:
        We temporarily use the dependency parser in spaCy to do this. Though, it may not be the best solution. Also,
        the parsing results from the spaCy parser may not the same as the visualization from their online demo.
        TODO:
        We may need to compare between the spaCy parser and the biaffine attention parser in AllenNLP to get the better
        one, or we may design a "ensemble" parser taking advantage of both of them.
        TODO:
        1. Negation: We would like to attach a negation word to its most dependent word. This dependent word can be a
        verb, noun, adjective or something else.
        # 2. conj
        3. More precise non-trivial words
        4. Co-references so far seem rather unstable in performance. Also, since a message may contain some content that
        cannot be easily resolved (e.g. pictures and videos), it may lead to further misleading to co-reference
        resolution. Thus, we temporarily do not do this.
        :param
            spacy_sent: A spaCy parsed sentence.
            l_nps: The list of noun phrase tuples contained in the sentence (the noun phrases should have been cleaned).
        :return:
            A NetworkX undirected and unweighted graph that represents the core clause structures. Vertices are strings.
            Edges are induced (i.e. not exactly) by dependencies.
            The node ids in the resulting graph are unique ids rather than actual texts on the nodes. The node labels
            are the actual texts.
        NOTE:
        A vertex in the resulting graph is a composed string of the format: [tw_id]|[sent_idx]|[token]
        '''
        if spacy_sent is None:
            raise Exception('[SemUnitsExtractor:extract_cls_from_sent] spacy_sent is invalid!')
        root = None
        for token in spacy_sent:
            if token.dep_ == 'ROOT':
                root = token
        # The root can be a negation token. So we put 'neg_tag_substitute' after fetching the root.
        spacy_sent = self.neg_tag_substitute(spacy_sent)
        q_nodes = deque()
        nx_cls = nx.Graph()
        if not self.is_trivial_token(root):
            cur_node, cur_pos, cur_start, cur_end = self.token_to_np(root, l_nps)
            nx_cls.add_node(self.build_node_id(sent_id, cur_node), txt=cur_node, pos=cur_pos, type='root',
                            start=cur_start, end=cur_end)
        for child in root.children:
            q_nodes.append(child)
        while len(q_nodes) > 0:
            cur = q_nodes.pop()
            if sd_2_usd.sd_to_usd(cur.dep_) in self.m_s_core_deps and not self.is_trivial_token(cur):
                cur_node, cur_pos, cur_start, cur_end = self.token_to_np(cur, l_nps)
                if cur_node not in nx_cls.nodes:
                    nearest_ancestor = self.find_nearest_ancestor_in_nx_graph(nx_cls, sent_id, cur)
                    if nearest_ancestor is None:
                        nx_cls.add_node(self.build_node_id(sent_id, cur_node), txt=cur_node, pos=cur_pos, type='root',
                                        start=cur_start, end=cur_end)
                    elif nearest_ancestor != cur_node:
                        nx_cls.add_node(self.build_node_id(sent_id, cur_node), txt=cur_node, pos=cur_pos, type='node',
                                        start=cur_start, end=cur_end)
                        nx_cls.add_edge(self.build_node_id(sent_id, nearest_ancestor),
                                        self.build_node_id(sent_id, cur_node))
            for child in cur.children:
                q_nodes.append(child)
        l_roots = [node[0] for node in nx_cls.nodes(data=True) if node[1]['type'] == 'root']
        for i in range(0, len(l_roots) - 1):
            for j in range(i + 1, len(l_roots)):
                nx_cls.add_edge(l_roots[i], l_roots[j])
        if len(nx_cls.nodes) == 1 and list(nx_cls.nodes)[0] in self.m_s_neg_cues:
            return None
        return nx_cls

    # @profile
    def extract_sem_units_from_text(self, raw_txt, txt_id):
        '''
        Extract semantic units for a given piece of raw text.
        :param
            raw_txt: A piece of raw text which can contain multiple sentences.
            txt_id: The unique ID for the raw_txt.
        :return:
            The core clause structure graph (the union graph from all sentences), the list of noun phrases.
        TODO:
        We may add other semantic units.
        '''
        if raw_txt is None or len(raw_txt) == 0:
            logging.debug('[SemUnitsExtractor:extract_sem_units_from_text] A trivial text occurs.')
            return None, None

        l_nps = []
        nx_cls = nx.Graph()
        l_spacy_sents, spacy_doc = self.spacy_pipeline_parse(raw_txt)
        for sent_id, spacy_sent in enumerate(l_spacy_sents):
            l_nps += self.spacy_extract_nps_from_sent(spacy_sent)
            nx_cls = self.union_nx_graphs(nx_cls,
                                          self.extract_cls_from_sent(str(txt_id) + '|' + str(sent_id),
                                                                     spacy_sent, l_nps))
        return nx_cls, l_nps

    def text_clean(self, raw_text):
        if raw_text is None or raw_text == '':
            return None
        l_clean_sents = []
        l_dirty_sents = raw_text.split('\n')
        for raw_dirt_sent in l_dirty_sents:
            # remove url
            clean_text = re.sub(r'url: [\S]*', '', raw_dirt_sent)
            clean_text = re.sub(r'http[\S]*', '', clean_text)
            # remove hashed ids
            clean_text = re.sub(r'@un:\s[\S]{22}\s', ' ', clean_text)
            clean_text = re.sub(r'@[^\s]+', ' ', clean_text)
            clean_text = re.sub(r'\s[\S]{22}\s', ' ', clean_text)
            # remove # symbol
            clean_text = re.sub(r'#', ' ', clean_text)
            # remove unnecessary white spaces
            clean_text = preprocessing.strip_multiple_whitespaces(clean_text)
            clean_text = clean_text.strip()
            trivial_test = re.match(r'.*[a-zA-A]*', clean_text)
            if trivial_test is not None:
                l_clean_sents.append(clean_text)
        return '\n'.join(l_clean_sents)

    def union_nx_graphs(self, nx_1, nx_2):
        if nx_1 is None or nx_2 is None:
            raise Exception('[SemUnitsExtractor:union_nx_graphs] nx_1 or nx_2 is empty!')
        for node in nx_2.nodes(data=True):
            nx_1.add_node(node[0], txt=node[1]['txt'], pos=node[1]['pos'], start=node[1]['start'], end=node[1]['end'])
        for edge in nx_2.edges:
            nx_1.add_edge(edge[0], edge[1])
        return nx_1

    def output_nx_graph(self, nx_graph, fig_path):
        if nx_graph is None or len(nx_graph.nodes) == 0:
            return
        plt.figure(1, figsize=(15, 15), tight_layout={'pad': 1, 'w_pad': 200, 'h_pad': 200, 'rect': None})
        pos = nx.spring_layout(nx_graph, k=0.8)
        x_values, y_values = zip(*pos.values())
        x_max = max(x_values)
        x_min = min(x_values)
        x_margin = (x_max - x_min) * 0.25
        plt.xlim(x_min - x_margin, x_max + x_margin)
        d_node_labels = {node[0]: node[1]['txt'] + ':' + node[1]['pos'] for node in nx_graph.nodes(data=True)}
        nx.draw_networkx_nodes(nx_graph, pos, node_size=50)
        nx.draw_networkx_labels(nx_graph, pos, labels=d_node_labels, font_size=25, font_family="sans-serif")
        nx.draw_networkx_edges(nx_graph, pos, width=2, edge_color='b')
        plt.savefig(fig_path, format="PNG")
        plt.clf()
        # plt.show()

    def task_multithreads(self, op_func, l_tasks, num_threads, job_id, output_format=None, output_db_path=None,
                          en_draw=False, other_params=()):
        '''
        A multithreading wrapper for a list a task with texts.
        :param
            op_func: The thread function to process a subset of tasks.
            l_tasks: A list of tasks. Each task is of the format: (task_id, text)
            num_threads: The number of threads
            output_folder: For outputs
            en_draw: True - draw outputs if necessary
        :return:
            No direct return value but outputs in the output folder and draws if any.
        '''
        timer_1 = time.time()
        logging.debug('[SemUnitsExtractor:task_multithreads] %s tasks in total.' % len(l_tasks))
        batch_size = math.ceil(len(l_tasks) / num_threads)
        l_l_subtasks = []
        for i in range(0, len(l_tasks), batch_size):
            if i + batch_size < len(l_tasks):
                l_l_subtasks.append(l_tasks[i:i + batch_size])
            else:
                l_l_subtasks.append(l_tasks[i:])
        logging.debug('[SemUnitsExtractor:task_multithreads] %s threads.' % len(l_l_subtasks))

        l_threads = []
        t_id = 0
        for l_each_batch in l_l_subtasks:
            t = threading.Thread(target=op_func, args=(l_each_batch, job_id + '_' + str(t_id), output_format,
                                                       en_draw) + other_params)
            t.setName('t_mul_task_' + str(t_id))
            t.start()
            l_threads.append(t)
            t_id += 1

        while len(l_threads) > 0:
            for t in l_threads:
                if t.is_alive():
                    t.join(1)
                else:
                    l_threads.remove(t)
                    logging.debug('[SemUnitsExtractor:task_multithreads] Thread %s is finished.' % t.getName())

        if output_db_path is not None:
            self.output_sem_units_to_db(output_format, output_db_path)
        logging.debug('[SemUnitsExtractor:task_multithreads] All done in %s sec for %s tasks.'
                      % (time.time() - timer_1, len(l_tasks)))

    def sem_unit_extraction_thread(self, l_tasks, thread_id, output_format, en_draw):
        '''
        Extract semantic units for a list of tasks with texts.
        TODO:
        Add modifier structures and noun pools.
        :param
            l_tasks: A list of tasks with texts, each of which is of the format: (task_id, text)
            thread_id: Thread name
            output_folder: For outputs
            en_draw: Draw outputs if necessary
        :return:
            No direct return value but outputs in the output folder and draws if any.
            Each core clause structure (of a sentence) is stored in a gml file representing an undirected and unweighted
            graph. The gml file name is of the format: [task_id]|[sent_id]_cls_graph.gml
            Each noun phrase output is also stored in a file with its name of the format: [task_id]|[sent_id]_nps.txt
        '''
        logging.debug('[SemUnitsExtractor:sem_unit_extraction_thread] Thread %s: Starts with %s tasks.'
                      % (thread_id, len(l_tasks)))
        timer_start = time.time()
        cnt = 0
        l_out_recs = []
        for tw_id, sent_txt in l_tasks:
            nx_cls, l_nps = self.extract_sem_units_from_text(sent_txt, tw_id + '|' + str(cnt))
            if (nx_cls is None or len(nx_cls.nodes()) <= 0) and (l_nps is None or len(l_nps) <= 0):
                continue
            if len(nx_cls.nodes()) <= 0:
                cls_str = None
            else:
                cls_str = json.dumps(nx.adjacency_data(nx_cls))
            if en_draw:
                self.output_nx_graph(nx_cls, output_format + tw_id + '_cls_graph.png')
            if len(l_nps) <= 0:
                nps = None
            # else:
            #     nps = '\n'.join([item[0] + '|' + str(item[2]) + '|' + str(item[3]) for item in l_nps])
            out_rec = (tw_id, cls_str, l_nps)
            l_out_recs.append(out_rec)
            cnt += 1
            if cnt % 5000 == 0 and cnt >= 5000:
                logging.debug('[SemUnitsExtractor:sem_unit_extraction_thread] Thread %s: %s sentences are done in %s secs.'
                              % (thread_id, cnt, str(time.time() - timer_start)))
        logging.debug('[SemUnitsExtractor:sem_unit_extraction_thread] Thread %s: All %s sentences are done in %s secs.'
                      % (thread_id, cnt, str(time.time() - timer_start)))
        out_df = pd.DataFrame(l_out_recs, columns=['tw_id', 'cls_json_str', 'nps'])
        if output_format is not None:
            out_df.to_pickle(output_format.format(str(thread_id)))
        logging.debug('[SemUnitsExtractor:sem_unit_extraction_thread] %s: All done with %s recs in %s secs.'
                      % (thread_id, len(out_df), str(time.time() - timer_start)))

    def output_sem_units_to_db(self, sem_unit_folder, db_path):
        db_conn = sqlite3.connect(db_path)
        db_cur = db_conn.cursor()
        sql_str = '''create table if not exists ven_tw_sem_units (tw_id text primay key, cls_json_str text, nps_str text)'''
        db_cur.execute(sql_str)
        sql_str = '''insert into ven_tw_sem_units (tw_id, cls_json_str, nps_str) values (?, ?, ?)'''
        timer_start = time.time()
        cnt = 0
        for (dirpath, dirname, filenames) in walk(sem_unit_folder):
            for filename in filenames:
                if filename[:14] == 'sem_units_int_' and filename[-5:] == '.json':
                    with open(dirpath + '/' + filename, 'r') as in_fd:
                        sem_units_json = json.load(in_fd)
                        in_fd.close()
                        for tw_id in sem_units_json:
                            cls_json_str = sem_units_json[tw_id]['cls']
                            nps_str = sem_units_json[tw_id]['nps']
                            try:
                                db_cur.execute(sql_str, (tw_id, cls_json_str, nps_str))
                            except Exception as err:
                                logging.debug('[SemUnitsExtractor:output_sem_units_to_db] %s' % err)
                                pass
                            cnt += 1
                            if cnt % 10000 == 0 and cnt >= 10000:
                                db_conn.commit()
                                logging.debug('[SemUnitsExtractor:output_sem_units_to_db] %s sem units are written in %s secs.'
                                              % (cnt, time.time() - timer_start))
        db_conn.commit()
        logging.debug('[SemUnitsExtractor:output_sem_units_to_db] %s sem units are written in %s secs.'
                      % (cnt, time.time() - timer_start))
        db_conn.close()
        logging.debug('[SemUnitsExtractor:output_sem_units_to_db] All done in %s secs.'
                      % str(time.time() - timer_start))

    def sem_unit_stats(self, sem_unit_folder, output_folder):
        l_degrees = []
        l_nodes = []
        l_edges = []
        l_nps = []
        cnt = 0
        for (dirpath, dirname, filenames) in walk(sem_unit_folder):
            for filename in filenames:
                if filename[-14:] == '_cls_graph.gml':
                    sem_unit_name = filename[:-14]
                    try:
                        cls_graph = nx.read_gml(dirpath + '/' + filename)
                        l_nodes.append(cls_graph.number_of_nodes())
                        l_edges.append(cls_graph.number_of_edges())
                        l_degrees.append(sum(dict(cls_graph.degree()).values()) / float(cls_graph.number_of_nodes()))
                    except:
                        pass
                elif filename[-8:] == '_nps.txt':
                    sem_unit_name = filename[:-8]
                    with open(dirpath + '/' + filename, 'r') as in_fd:
                        lns = in_fd.readlines()
                        ln_cnt = len(lns)
                        l_nps.append(ln_cnt)
                        in_fd.close()
                cnt += 1
                if cnt % 10000 == 0 and cnt >= 10000:
                    with open(output_folder + 'sem_unit_stats_' + str(cnt) + '_.txt', 'w+') as out_fd:
                        out_fd.write('cls: avg. degrees:\n')
                        out_fd.write(','.join([str(num) for num in l_degrees]))
                        out_fd.write('\n')
                        out_fd.write('cls: nodes:\n')
                        out_fd.write(','.join([str(num) for num in l_nodes]))
                        out_fd.write('\n')
                        out_fd.write('cls: edges:\n')
                        out_fd.write(','.join([str(num) for num in l_edges]))
                        out_fd.write('\n')
                        out_fd.write('nps:\n')
                        out_fd.write(','.join([str(num) for num in l_nps]))
                        out_fd.write('\n')
                        out_fd.close()
                    logging.debug('%s sem unit stats are done.' % cnt)
        with open(output_folder + 'sem_unit_stats_' + str(cnt) + '_.txt', 'w+') as out_fd:
            out_fd.write('cls: avg. degrees:\n')
            out_fd.write(','.join([str(num) for num in l_degrees]))
            out_fd.write('\n')
            out_fd.write('cls: nodes:\n')
            out_fd.write(','.join([str(num) for num in l_nodes]))
            out_fd.write('\n')
            out_fd.write('cls: edges:\n')
            out_fd.write(','.join([str(num) for num in l_edges]))
            out_fd.write('\n')
            out_fd.write('nps:\n')
            out_fd.write(','.join([str(num) for num in l_nps]))
            out_fd.write('\n')
            out_fd.close()