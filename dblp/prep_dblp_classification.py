import random

dataset = 'dblp'


def load_graph_info():
    graph_dict = dict()
    node_type_dict = dict()
    node_id_dict = dict()
    node_id = 1
    # read nodes info
    with open('./dataset/' + dataset + '/author.txt', "r") as fr:
        lines = fr.readlines()
        for line in lines:
            temp = list(line.strip('\n').split('\t'))
            node_id_dict['a_' + temp[0]] = node_id
            node_type_dict[node_id] = 'author'
            if node_id not in graph_dict:
                graph_dict[node_id] = set()
            node_id += 1
    with open('./dataset/' + dataset + '/conf.txt', "r") as fr:
        lines = fr.readlines()
        for line in lines:
            temp = list(line.strip('\n').split('\t'))
            node_id_dict['c_' + temp[0]] = node_id
            node_type_dict[node_id] = 'conf'
            if node_id not in graph_dict:
                graph_dict[node_id] = set()
            node_id += 1
    with open('./dataset/' + dataset + '/paper.txt', "r", encoding="GBK") as fr:
        lines = fr.readlines()
        for line in lines:
            temp = list(line.strip('\n').split('\t'))
            node_id_dict['p_' + temp[0]] = node_id
            node_type_dict[node_id] = 'paper'
            if node_id not in graph_dict:
                graph_dict[node_id] = set()
            node_id += 1
    with open('./dataset/' + dataset + '/term.txt', "r") as fr:
        lines = fr.readlines()
        for line in lines:
            temp = list(line.strip('\n').split('\t'))
            node_id_dict['t_' + temp[0]] = node_id
            node_type_dict[node_id] = 'term'
            if node_id not in graph_dict:
                graph_dict[node_id] = set()
            node_id += 1
    # read edges info
    with open('./dataset/' + dataset + '/paper_author.txt', "r") as fr:
        lines = fr.readlines()
        for line in lines:
            temp = list(line.strip('\n').split('\t'))
            graph_dict[node_id_dict['p_' + temp[0]]].add(node_id_dict['a_' + temp[1]])
            graph_dict[node_id_dict['a_' + temp[1]]].add(node_id_dict['p_' + temp[0]])
    with open('./dataset/' + dataset + '/paper_conf.txt', "r") as fr:
        lines = fr.readlines()
        for line in lines:
            temp = list(line.strip('\n').split('\t'))
            graph_dict[node_id_dict['p_' + temp[0]]].add(node_id_dict['c_' + temp[1]])
            graph_dict[node_id_dict['c_' + temp[1]]].add(node_id_dict['p_' + temp[0]])
    with open('./dataset/' + dataset + '/paper_term.txt', "r") as fr:
        lines = fr.readlines()
        for line in lines:
            temp = list(line.strip('\n').split('\t'))
            graph_dict[node_id_dict['p_' + temp[0]]].add(node_id_dict['t_' + temp[1]])
            graph_dict[node_id_dict['t_' + temp[1]]].add(node_id_dict['p_' + temp[0]])
    # read label info
    labeled_author_dict = dict()
    with open('./dataset/' + dataset + '/author_label.txt', "r") as fr:
        lines = fr.readlines()
        for line in lines:
            temp = list(line.strip('\n').split('\t'))
            labeled_author_dict[node_id_dict['a_' + temp[0]]] = temp[1]
    labeled_conf_dict = dict()
    with open('./dataset/' + dataset + '/conf_label.txt', "r") as fr:
        lines = fr.readlines()
        for line in lines:
            temp = list(line.strip('\n').split('\t'))
            labeled_conf_dict[node_id_dict['c_' + temp[0]]] = temp[1]
    # labeled_paper_dict = dict()
    # with open('./dataset/' + dataset + '/paper_label.txt', "r") as fr:
    #     lines = fr.readlines()
    #     for line in lines:
    #         temp = list(line.strip('\n').split('\t'))
    #         l_p_d[node_id_dict['p_' + temp[0]]] = temp[1]
    return graph_dict, node_type_dict, labeled_author_dict, labeled_conf_dict


def graph_stat(graph_dict, node_type_dict, labeled_author_dict):
    node_num = 0
    tail_node_num = 0
    labeled_paper_dict = dict()
    tail_paper_set = set()
    tail_author_set = set()
    node_type_num_dict = {'author': 0, 'conf': 0, 'paper': 0, 'term': 0}
    tail_node_type_num_dict = {'author': 0, 'conf': 0, 'paper': 0, 'term': 0}
    p_set = set()
    for a in labeled_author_dict:
        for p in graph_dict[a]:
            p_set.add(p)
    print(len(p_set))
    for node in graph_dict:
        node_num += 1
        node_type_num_dict[node_type_dict[node]] += 1
        if node_type_dict[node] == 'paper':
            author_num = 0
            author_type_num_dict = {'0': 0, '1': 0, '2': 0, '3': 0}
            for adj in graph_dict[node]:
                if node_type_dict[adj] == 'author':
                    author_num += 1
                    if adj in labeled_author_dict:
                        author_type_num_dict[labeled_author_dict[adj]] += 1
            for k in author_type_num_dict:
                if author_type_num_dict[k] > author_num / 2:
                    labeled_paper_dict[node] = k
        if len(graph_dict[node]) <= 5:
            if node_type_dict[node] == 'author':
                tail_author_set.add(node)
            if node_type_dict[node] == 'paper':
                tail_paper_set.add(node)
            tail_node_num += 1
            tail_node_type_num_dict[node_type_dict[node]] += 1
    print('tail author num: %d' % len(tail_author_set))
    print('tail paper num: %d' % len(tail_paper_set))
    print('node type info: ')
    print(node_type_num_dict)
    print('tail node type info: ')
    print(tail_node_type_num_dict)
    print(len(labeled_paper_dict))
    return node_num, tail_node_num, labeled_paper_dict


def clear_dataset(graph_dict, node_type_dict, labeled_author_dict):
    print('clear_dataset')
    p_set = set()
    for a in labeled_author_dict:
        for p in graph_dict[a]:
            p_set.add(p)
    r_set = set()
    for n in graph_dict:
        if node_type_dict[n] == 'paper' and n not in p_set:
            for adj in graph_dict[n]:
                graph_dict[adj].remove(n)
            r_set.add(n)
        if node_type_dict[n] == 'author' and n not in labeled_author_dict:
            for adj in graph_dict[n]:
                graph_dict[adj].remove(n)
            r_set.add(n)
    for n in r_set:
        graph_dict.pop(n)
    r_set.clear()
    for n in graph_dict:
        if len(graph_dict[n]) == 0:
            r_set.add(n)
    for n in r_set:
        graph_dict.pop(n)
    a_num = 0
    p_num = 0
    t_num = 0
    c_num = 0
    for n in graph_dict:
        if node_type_dict[n] == 'author':
            a_num += 1
        if node_type_dict[n] == 'paper':
            p_num += 1
        if node_type_dict[n] == 'conf':
            c_num += 1
        if node_type_dict[n] == 'term':
            t_num += 1
    print(a_num)
    print(p_num)
    print(c_num)
    print(t_num)
    r_set.clear()
    for n in node_type_dict:
        if n not in graph_dict:
            r_set.add(n)
    for n in r_set:
        node_type_dict.pop(n)
    print(len(node_type_dict))
    edges = 0
    for n in graph_dict:
        edges+=len(graph_dict[n])
    print('edges')
    print(edges/2)
    return graph_dict, node_type_dict


def two_hop_graph_stat(g, n_d):
    a_num = 0
    t_a_num = 0
    t_a_set = set()
    n_5_10 = 0
    n_10_20 = 0
    for n in g:
        if n_d[n] == 'author':
            a_num += 1
            s = set()
            for adj in g[n]:
                for t_adj in g[adj]:
                    if n_d[t_adj] == 'author':
                        s.add(t_adj)
            if len(s) <= 5:
                t_a_num += 1
                t_a_set.add(n)
            if 5 < len(s) <= 10:
                n_5_10 += 1
            if 20 < len(s):
                n_10_20 += 1
    print(n_5_10)
    print(n_10_20)
    return a_num, t_a_num, t_a_set


def select_train_and_test_node(g, n_d):
    train_set_a = set()
    train_set_p = set()
    test_set = set()
    for n in g:
        if n_d[n] == 'author':
            s = set()
            for adj in g[n]:
                for t_adj in g[adj]:
                    if n_d[t_adj] == 'author':
                        s.add(t_adj)
            if len(g[n]) <= 5:
                test_set.add(n)
            else:
                if len(g[n]) > 5 and len(s) > 5:
                    train_set_a.add(n)
        if n_d[n] == 'paper':
            s = set()
            for adj in g[n]:
                for t_adj in g[adj]:
                    if n_d[t_adj] == 'paper':
                        s.add(t_adj)
            if len(g[n]) <= 5:
                test_set.add(n)
            else:
                if len(g[n]) > 5 and len(s) > 5:
                    train_set_p.add(n)
    print(len(train_set_a))
    print(len(train_set_p))
    train_list_a_selected = list(train_set_a)
    while len(train_list_a_selected) < len(train_set_p):
        train_list_a_selected.append(random.sample(train_set_a, 1)[0])
    train_list_selected = list(train_set_p) + train_list_a_selected
    random.shuffle(train_list_selected)
    return train_list_selected, test_set


def author_and_paper_adjlist(g, n_d):
    author_adjlist_dict = dict()
    paper_adjlist_dict = dict()
    for n in g:
        if n_d[n] == 'author':
            if n not in author_adjlist_dict:
                author_adjlist_dict[n] = set()
            for adj in g[n]:
                for t_adj in g[adj]:
                    if n_d[t_adj] == 'author' and t_adj != n:
                        author_adjlist_dict[n].add(t_adj)
        if n_d[n] == 'paper':
            if n not in paper_adjlist_dict:
                paper_adjlist_dict[n] = set()
            for adj in g[n]:
                for t_adj in g[adj]:
                    if n_d[t_adj] == 'paper' and t_adj != n:
                        paper_adjlist_dict[n].add(t_adj)
    return author_adjlist_dict, paper_adjlist_dict


def save_graph(graph_dict, graph_dir):
    with open(graph_dir, 'w') as fw:
        for key in graph_dict.keys():
            fw.write(str(key))
            for item in graph_dict[key]:
                fw.write(' ' + str(item))
            fw.write('\n')


def save_node(node_set, node_save_dir):
    with open(node_save_dir, 'w') as fw:
        for n in node_set:
            fw.write(str(n))
            fw.write('\n')


def save_node_type_info(node_type, save_dir):
    with open(save_dir, 'w') as fw:
        for key in node_type.keys():
            # temp = '0'
            if node_type[key] == 'author':
                temp = '0'
            elif node_type[key] == 'paper':
                temp = '1'
            elif node_type[key] == 'conf':
                temp = '2'
            else:
                temp = '3'
            fw.write(str(key) + ' ' + temp)
            fw.write('\n')


def save_gt_info(node_type, save_dir):
    with open(save_dir, 'w') as fw:
        for key in node_type.keys():
            fw.write(str(key) + ' ' + node_type[key])
            fw.write('\n')


# def test(g):
#     for n in g:
#         if len(g[n]) == 0:
#             print("yes")


if __name__ == '__main__':
    graph, node_type, labeled_author, labeled_conf = load_graph_info()
    graph, node_type = clear_dataset(graph, node_type, labeled_author)
    node_num, tail_node_num, labeled_paper = graph_stat(graph, node_type, labeled_author)
    train_node_set, test_node_set = select_train_and_test_node(graph, node_type)
    # save_graph(graph, './data/' + dataset + '/graph.adjlist')
    # save_node(train_node_set, './data/' + dataset + '/train.csv')
    # save_node(test_node_set, './data/' + dataset + '/test.csv')
    # author_adjlist, paper_adjlist = author_and_paper_adjlist(graph, node_type)
    # save_graph(author_adjlist, './data/' + dataset + '/type1.adjlist')
    # save_graph(paper_adjlist, './data/' + dataset + '/type2.adjlist')
    # save_node_type_info(node_type, './data/' + dataset + '/node_type.txt')
    # save_gt_info(labeled_author, './data/' + dataset + '/gt_a.txt')
    # save_gt_info(labeled_paper, './data/' + dataset + '/gt_p.txt')
