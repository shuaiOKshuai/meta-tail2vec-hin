import numpy as np
import os
import random
import tensorflow as tf
import tqdm
import math
import csv
import sys


def cal_embedding(travel, hop_num_list, p_lambda, base_embeddings):
    weights = np.ndarray(shape=(len(travel),))
    index = 0
    for j in range(len(hop_num_list)):
        for k in range(hop_num_list[j]):
            weights[index] = math.exp(p_lambda * j)
            index += 1
    norm_weights = weights / weights.sum()
    index = 0
    temp_embeddings = np.zeros(shape=(len(travel), 128))
    for node in travel:
        temp_embeddings[index] = np.array(base_embeddings[node]).astype(np.float)
        index += 1
    embeddings = np.sum(np.multiply(temp_embeddings, norm_weights.reshape((-1, 1))), axis=0)
    return embeddings.tolist()


def cal_embedding_hin(travel, base_embeddings, class_num):
    embeddings = list()
    for i in range(class_num):
        if str(i) not in travel:
            temp_embeddings = np.zeros(shape=(1, 128))
        else:
            temp_embeddings = np.zeros(shape=(len(travel[str(i)]), 128))
            index = 0
            for n in travel[str(i)]:
                temp_embeddings[index] = np.array(base_embeddings[n]).astype(np.float)
                index += 1
        embeddings.append(np.mean(temp_embeddings, axis=0).tolist())
    return embeddings


def generate_support_data(graph, source, base_embeddings, hop, node_max_size, p_lambda):
    hop_num_list = []
    frontiers = {source}
    travel = [source]
    travel_set = {source}
    travel_hop = 1
    while travel_hop <= hop:
        nexts = set()
        node_size = node_max_size[travel_hop - 1]
        for frontier in frontiers:
            if len(graph[frontier]) > node_size:
                node_children = np.random.choice(graph[frontier], node_size, replace=False)
            else:
                node_children = graph[frontier]
            for current in node_children:
                if current not in travel_set:
                    travel.append(current)
                    nexts.add(current)
                    travel_set.add(current)
        frontiers = nexts
        hop_num_list.append(len(nexts))
        travel_hop += 1
    travel.remove(source)
    feature_embedding = cal_embedding(travel, hop_num_list, p_lambda, base_embeddings)
    return base_embeddings[source], feature_embedding


def generate_support_data_hin(graph, source, base_embeddings, node_type, class_num, hop, node_max_size):
    hop_num_list = []
    frontiers = {source}
    travel = [source]
    travel_dict = dict()
    travel_set = {source}
    travel_hop = 1
    while travel_hop <= hop:
        nexts = set()
        node_size = node_max_size[travel_hop - 1]
        for frontier in frontiers:
            if len(graph[frontier]) > node_size:
                node_children = np.random.choice(graph[frontier], node_size, replace=False)
            else:
                node_children = graph[frontier]
            for current in node_children:
                if current not in travel_set:
                    travel.append(current)
                    nexts.add(current)
                    travel_set.add(current)
                    if node_type[current] not in travel_dict:
                        travel_dict[node_type[current]] = set()
                    travel_dict[node_type[current]].add(current)
        frontiers = nexts
        hop_num_list.append(len(nexts))
        travel_hop += 1
    travel.remove(source)
    feature_embedding = cal_embedding_hin(travel_dict, base_embeddings, class_num)
    return base_embeddings[source], feature_embedding


def generate_query_data(graph, source, base_embeddings, s_n, hop, node_max_size, p_lambda):
    hop_num_list = []
    frontiers = {source}
    travel = [source]
    travel_set = {source}
    travel_hop = 1
    while travel_hop <= hop:
        nexts = set()
        node_size = node_max_size[travel_hop - 1]
        for frontier in frontiers:
            if travel_hop == 1:
                node_children = s_n
            else:
                if len(graph[frontier]) > node_size:
                    node_children = np.random.choice(list(graph[frontier]), node_size, replace=False)
                else:
                    node_children = graph[frontier]
            for current in node_children:
                if current not in travel_set:
                    travel.append(current)
                    nexts.add(current)
                    travel_set.add(current)
        frontiers = nexts
        hop_num_list.append(len(nexts))
        travel_hop += 1
    travel.remove(source)
    feature_embedding = cal_embedding(travel, hop_num_list, p_lambda, base_embeddings)
    return base_embeddings[source], feature_embedding


def generate_query_data_hin(graph, source, base_embeddings, node_type, s_n, class_num, hop, node_max_size):
    hop_num_list = []
    frontiers = {source}
    travel = [source]
    travel_dict = dict()
    travel_set = {source}
    travel_hop = 1
    while travel_hop <= hop:
        nexts = set()
        node_size = node_max_size[travel_hop - 1]
        for frontier in frontiers:
            if len(graph[frontier]) > node_size:
                node_children = np.random.choice(graph[frontier], node_size, replace=False)
            else:
                node_children = graph[frontier]
            for current in node_children:
                if current not in travel_set:
                    travel.append(current)
                    nexts.add(current)
                    travel_set.add(current)
                    if node_type[current] not in travel_dict:
                        travel_dict[node_type[current]] = list()
                    travel_dict[node_type[current]].append(current)
        frontiers = nexts
        hop_num_list.append(len(nexts))
        travel_hop += 1
    travel.remove(source)
    feature_embedding = cal_embedding_hin(travel_dict, base_embeddings, class_num)
    return base_embeddings[source], feature_embedding


def MaxMinNormalization(x, Max, Min):
    x = (x - Min) / (Max - Min)
    return x


def get_surround_info(n, g, node_type, class_num):
    surround_info = [0.] * class_num
    indicator = [0.] * class_num
    indicator[int(node_type[n])] = 1
    surround_set = set()
    for adj_1 in g[n]:
        if adj_1 not in surround_set:
            surround_info[int(node_type[adj_1])] += 1
            surround_set.add(adj_1)
        for adj_2 in g[adj_1]:
            if adj_2 not in surround_set:
                surround_info[int(node_type[adj_2])] += 1
                surround_set.add(adj_2)
    max_ = max(surround_info)
    min_ = min(surround_info)
    if max_ - min_ == 0:
        print('nonononononononononononononononononononononononono')
        sys.exit()
    surround_info_n = [MaxMinNormalization(x, max_, min_) for x in surround_info]
    return surround_info_n + indicator


def write_task_to_file(s_n, q_n, g, emb, node_type, class_num, hop, size):
    task_data = []
    blank_surround_info = [0.] * (class_num * 2)
    blank_row = [0.] * (2 + 128 * (class_num + 1) + class_num * 2)
    s_index = 0
    for n in s_n:
        oracle_embedding, embedding = generate_support_data_hin(g, n, emb, node_type, class_num, hop=hop,
                                                                node_max_size=size)
        temp = list(n.split()) + list(n.split()) + oracle_embedding
        for e in embedding:
            temp += e
        temp += get_surround_info(n, g, node_type, class_num)
        task_data.append(temp)
        s_index += 1
    while s_index < 5:
        task_data.append(blank_row)
        s_index += 1
    for n in q_n:
        oracle_embedding, embedding = generate_query_data_hin(g, n, emb, node_type, s_n, class_num, hop=hop,
                                                              node_max_size=size)
        if node_type[n] == '0':
            x = ['0']
        else:
            x = ['1']
        temp = x + list(n.split()) + oracle_embedding
        for e in embedding:
            temp += e
        temp += get_surround_info(n, g, node_type, class_num)
        task_data.append(temp)
    return task_data


class DataGenerator:
    def __init__(self, main_dir, dataset_name, kshot, meta_batchsz, class_num, total_batch_num=200):
        self.main_dir = main_dir
        self.kshot = kshot
        self.meta_batchsz = meta_batchsz
        self.total_batch_num = total_batch_num
        self.dataset_name = dataset_name
        self.class_num = class_num
        self.hop = 2
        self.size1 = 50
        self.size2 = 25
        self.p_lambda = 0

        self.metatrain_file = self.main_dir + dataset_name + '/train.csv'
        self.metatest_file = self.main_dir + dataset_name + '/test.csv'

        self.graph_dir = self.main_dir + dataset_name + '/graph.adjlist'
        self.type1_adjlist_dir = self.main_dir + dataset_name + '/type1.adjlist'
        self.type2_adjlist_dir = self.main_dir + dataset_name + '/type2.adjlist'
        self.emb_dir = self.main_dir + dataset_name + '/ml_mp2vec.txt'
        self.node_type_dir = self.main_dir + dataset_name + '/node_type.txt'

        self.graph = dict()
        with open(self.graph_dir, 'r') as fr:
            lines = fr.readlines()
            for line in lines:
                temp = list(line.strip('\n').split(' '))
                self.graph[temp[0]] = list()
                for n in range(1, len(temp)):
                    self.graph[temp[0]].append(temp[n])
        self.type1_adjlist = dict()
        with open(self.type1_adjlist_dir, 'r') as fr:
            lines = fr.readlines()
            for line in lines:
                temp = list(line.strip('\n').split(' '))
                self.type1_adjlist[temp[0]] = set()
                for n in range(1, len(temp)):
                    self.type1_adjlist[temp[0]].add(temp[n])
        self.type2_adjlist = dict()
        with open(self.type2_adjlist_dir, 'r') as fr:
            lines = fr.readlines()
            for line in lines:
                temp = list(line.strip('\n').split(' '))
                self.type2_adjlist[temp[0]] = set()
                for n in range(1, len(temp)):
                    self.type2_adjlist[temp[0]].add(temp[n])
        self.base_emb = dict()
        with open(self.emb_dir, 'r') as fr:
            lines = fr.readlines()
            for line in lines:
                temp = list(line.strip('\n').split(' '))
                self.base_emb[temp[0]] = temp[1:]
        self.node_type = dict()
        with open(self.node_type_dir, 'r') as fr:
            lines = fr.readlines()
            for line in lines:
                temp = list(line.strip('\n').split(' '))
                self.node_type[temp[0]] = str(temp[1])

    def make_data_tensor(self, training=True):
        num_total_batches = self.total_batch_num
        if training:
            file = self.metatrain_file
        else:
            file = self.metatest_file

        if training:
            if os.path.exists('./data/' + self.dataset_name + '/trainfile.csv'):
                pass
            else:
                all_data = []
                train_nodes = []
                with open(file, "r") as fr:
                    lines = fr.readlines()
                    for line in lines:
                        temp = list(line.strip('\n').split(','))
                        train_nodes.append(temp[0])
                for _ in tqdm.tqdm(range(num_total_batches), 'generating episodes'):
                    query_node = random.sample(train_nodes, 1)
                    print(query_node)
                    if self.node_type[query_node[0]] == '0':
                        support_node = random.sample(self.type1_adjlist[query_node[0]], self.kshot)
                    else:
                        support_node = random.sample(self.type2_adjlist[query_node[0]], self.kshot)
                    task_data = write_task_to_file(support_node, query_node, self.graph, self.base_emb, self.node_type,
                                                   self.class_num, self.hop, (self.size1, self.size2))
                    all_data.extend(task_data)

                with open('./data/' + self.dataset_name + '/trainfile.csv', 'w') as fw:
                    writer = csv.writer(fw)
                    writer.writerows(all_data)
                    print('save train file list to trainfile.csv')
        else:
            if os.path.exists('./data/' + self.dataset_name + '/testfile.csv'):
                pass
            else:
                all_data = []
                test_nodes = []
                other_nodes = []
                with open(file, "r") as fr:
                    lines = fr.readlines()
                    for line in lines:
                        temp = list(line.strip('\n').split(','))
                        test_nodes.append(temp[0])
                for n in tqdm.tqdm(test_nodes, 'generating test episodes'):
                    query_node = list()
                    query_node.append(n)
                    print(query_node)
                    # support_node = self.graph[query_node[0]]
                    # support_node = random.sample(self.author_adjlist[query_node[0]], self.kshot)
                    if self.node_type[query_node[0]] == '0':
                        if len(self.type1_adjlist[query_node[0]]) > self.kshot:
                            support_node = random.sample(self.type1_adjlist[query_node[0]], self.kshot)
                        else:
                            support_node = self.type1_adjlist[query_node[0]]
                    else:
                        if len(self.type2_adjlist[query_node[0]]) > self.kshot:
                            support_node = random.sample(self.type2_adjlist[query_node[0]], self.kshot)
                        else:
                            support_node = self.type2_adjlist[query_node[0]]
                    task_data = write_task_to_file(support_node, query_node, self.graph, self.base_emb, self.node_type,
                                                   self.class_num, self.hop, (self.size1, self.size2))
                    all_data.extend(task_data)
                with open('./data/' + self.dataset_name + '/testfile.csv', 'w') as fw:
                    writer = csv.writer(fw)
                    writer.writerows(all_data)
                    print('save test file list to testfile.csv')

        print('creating pipeline ops')
        if training:
            filename_queue = tf.train.string_input_producer(['./data/' + self.dataset_name + '/trainfile.csv'],
                                                            shuffle=False)
        else:
            filename_queue = tf.train.string_input_producer(['./data/' + self.dataset_name + '/testfile.csv'],
                                                            shuffle=False)
        reader = tf.TextLineReader()
        _, value = reader.read(filename_queue)
        record_defaults = [0.] * (2 + 128 * (self.class_num + 1) + self.class_num * 2)
        row = tf.decode_csv(value, record_defaults=record_defaults)
        feature_and_label = tf.stack(row)

        print('batching data')
        examples_per_batch = 1 + self.kshot
        batch_data_size = self.meta_batchsz * examples_per_batch
        features = tf.train.batch(
            [feature_and_label],
            batch_size=batch_data_size,
            num_threads=1,
            capacity=256,
        )
        all_node_id = []
        all_label_batch = []
        all_feature_batch = []
        for i in range(self.meta_batchsz):
            data_batch = features[i * examples_per_batch:(i + 1) * examples_per_batch]
            node_id, label_batch, feature_batch = tf.split(data_batch, [2, 128, (128 + 2) * self.class_num], axis=1)
            all_node_id.append(node_id)
            all_label_batch.append(label_batch)
            all_feature_batch.append(feature_batch)
        all_node_id = tf.stack(all_node_id)
        all_label_batch = tf.stack(all_label_batch)
        all_feature_batch = tf.stack(all_feature_batch)
        return all_node_id, all_label_batch, all_feature_batch
