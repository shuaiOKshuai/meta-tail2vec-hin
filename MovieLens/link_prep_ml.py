import random
from sklearn.model_selection import train_test_split

dataset = 'ml'


def load_graph_info():
    graph_dict = dict()
    node_type_dict = dict()
    node_id_dict = dict()
    node_id = 1
    # read nodes info
    with open('./dataset/' + dataset + '/movies.dat', "r", encoding="UTF-8") as fr:
        lines = fr.readlines()
        for line in lines:
            temp = list(line.strip('\n').split('::'))
            node_id_dict['m_' + temp[0]] = node_id
            node_type_dict[node_id] = 'movie'
            if node_id not in graph_dict:
                graph_dict[node_id] = set()
            node_id += 1
    with open('./dataset/' + dataset + '/users.dat', "r", encoding="UTF-8") as fr:
        lines = fr.readlines()
        for line in lines:
            temp = list(line.strip('\n').split('::'))
            node_id_dict['u_' + temp[0]] = node_id
            node_type_dict[node_id] = 'user'
            if node_id not in graph_dict:
                graph_dict[node_id] = set()
            node_id += 1
    with open('./dataset/' + dataset + '/ratings.dat', "r") as fr:
        lines = fr.readlines()
        for line in lines:
            temp = list(line.strip('\n').split('::'))
            graph_dict[node_id_dict['u_' + temp[0]]].add(node_id_dict['m_' + temp[1]])
            graph_dict[node_id_dict['m_' + temp[1]]].add(node_id_dict['u_' + temp[0]])
    with open('./dataset/' + dataset + '/movies_extrainfos.dat', "r", encoding="UTF-8") as fr:
        lines = fr.readlines()
        for line in lines:
            temp = list(line.strip('\n').split('::'))
            if 'd_' + temp[6] not in node_id_dict:
                node_id_dict['d_' + temp[6]] = node_id
                node_type_dict[node_id] = 'director'
                node_id += 1
            if node_id_dict['d_' + temp[6]] not in graph_dict:
                graph_dict[node_id_dict['d_' + temp[6]]] = set()
            graph_dict[node_id_dict['m_' + temp[0]]].add(node_id_dict['d_' + temp[6]])
            graph_dict[node_id_dict['d_' + temp[6]]].add(node_id_dict['m_' + temp[0]])
            actors = list(temp[8].split(', '))
            for actor in actors:
                if 'a_' + actor not in node_id_dict:
                    node_id_dict['a_' + actor] = node_id
                    node_type_dict[node_id] = 'actor'
                    node_id += 1
                if node_id_dict['a_' + actor] not in graph_dict:
                    graph_dict[node_id_dict['a_' + actor]] = set()
                graph_dict[node_id_dict['m_' + temp[0]]].add(node_id_dict['a_' + actor])
                graph_dict[node_id_dict['a_' + actor]].add(node_id_dict['m_' + temp[0]])
    genre_dict = dict()
    genre_id = 0
    with open('./dataset/' + dataset + '/movies.dat', "r", encoding="UTF-8") as fr:
        lines = fr.readlines()
        for line in lines:
            temp = list(line.strip('\n').split('::'))
            genres = list(temp[2].split('|'))
            for genre in genres:
                if genre not in genre_dict:
                    genre_dict[genre] = genre_id
                    genre_id += 1
    labeled_movie_dict = dict()
    with open('./dataset/' + dataset + '/movies.dat', "r", encoding="UTF-8") as fr:
        lines = fr.readlines()
        for line in lines:
            temp = list(line.strip('\n').split('::'))
            labeled_movie_dict[node_id_dict['m_' + temp[0]]] = set()
            genres = list(temp[2].split('|'))
            for genre in genres:
                labeled_movie_dict[node_id_dict['m_' + temp[0]]].add(genre_dict[genre])
    return graph_dict, node_type_dict, labeled_movie_dict


def graph_stat(graph_dict, node_type_dict, labeled_movie_dict):
    node_num = 0
    tail_node_num = 0
    labeled_user_dict = dict()
    node_type_num_dict = {'user': 0, 'movie': 0, 'director': 0, 'actor': 0}
    tail_node_type_num_dict = {'user': 0, 'movie': 0, 'director': 0, 'actor': 0}
    for node in graph_dict:
        node_type_num_dict[node_type_dict[node]] += 1
        node_num += 1
        if node_type_dict[node] == 'user':
            movie_type_num_dict = dict()
            type_num = 0
            for adj in graph_dict[node]:
                for t in labeled_movie_dict[adj]:
                    type_num += 1
                    if t not in movie_type_num_dict.keys():
                        movie_type_num_dict[t] = 1
                    else:
                        movie_type_num_dict[t] += 1
            if node not in labeled_user_dict:
                labeled_user_dict[node] = set()
            for t in movie_type_num_dict.keys():
                if movie_type_num_dict[t] >= type_num / 1000:
                    # if node not in labeled_user_dict:
                    #     labeled_user_dict[node] = set()
                    labeled_user_dict[node].add(t)
            # labeled_user_dict[node] = str(max(zip(movie_type_num_dict.values(), movie_type_num_dict.keys()))[1])
        if len(graph_dict[node]) <= 10:
            tail_node_num += 1
            tail_node_type_num_dict[node_type_dict[node]] += 1
        if len(graph_dict[node]) <= 30 and node_type_dict[node] == 'user':
            tail_user_set.add(node)
        if len(graph_dict[node]) <= 10 and node_type_dict[node] == 'movie':
            tail_movie_set.add(node)
    print('node type info: ')
    print(node_type_num_dict)
    print('tail node type info: ')
    print(tail_node_type_num_dict)
    return node_num, tail_node_num, labeled_user_dict


def two_hop_graph_stat(g, n_d):
    a_num = 0
    t_a_num = 0
    t_a_set = set()
    n_5_10 = 0
    n_10_20 = 0
    for n in g:
        if n_d[n] == 'movie':
            a_num += 1
            s = set()
            for adj in g[n]:
                for t_adj in g[adj]:
                    if n_d[t_adj] == 'movie':
                        s.add(t_adj)
            if len(s) <= 5:
                t_a_num += 1
                t_a_set.add(n)
            if 5 < len(s) <= 10:
                n_5_10 += 1
            if 100 < len(s):
                n_10_20 += 1
    print(n_5_10)
    print(n_10_20)
    return a_num, t_a_num, t_a_set


def select_train_and_test_node(g, n_d):
    train_set_m = set()
    train_set_u = set()
    test_set = set()
    for n in g:
        if n_d[n] == 'movie':
            s = set()
            for adj in g[n]:
                for t_adj in g[adj]:
                    if n_d[t_adj] == 'movie':
                        s.add(t_adj)
            if len(g[n]) <= 10:
                test_set.add(n)
            else:
                if len(g[n]) <= 114 and len(s) > 5 and len(g[n]) > 10:
                    train_set_m.add(n)
        if n_d[n] == 'user':
            s = set()
            for adj in g[n]:
                for t_adj in g[adj]:
                    if n_d[t_adj] == 'user':
                        s.add(t_adj)
            if len(g[n]) < 30:
                test_set.add(n)
            else:
                if len(g[n]) <= 90 and len(s) > 5 and len(g[n]) >= 30:
                    train_set_u.add(n)
    print('movie train node num:')
    print(len(train_set_m))
    print('user train node num:')
    print(len(train_set_u))
    train_list_m_selected = list(train_set_m)
    while len(train_list_m_selected) < len(train_set_u):
        train_list_m_selected.append(random.sample(train_set_m, 1)[0])
    train_list_selected = list(train_set_u) + train_list_m_selected
    random.shuffle(train_list_selected)
    return train_list_selected, test_set


def movie_and_user_adjlist(g, n_d):
    movie_adjlist_dict = dict()
    user_adjlist_dict = dict()
    for n in g:
        if n_d[n] == 'movie':
            if n not in movie_adjlist_dict:
                movie_adjlist_dict[n] = set()
            for adj in g[n]:
                for t_adj in g[adj]:
                    if n_d[t_adj] == 'movie' and t_adj != n:
                        movie_adjlist_dict[n].add(t_adj)
        if n_d[n] == 'user':
            if n not in user_adjlist_dict:
                user_adjlist_dict[n] = set()
            for adj in g[n]:
                for t_adj in g[adj]:
                    if n_d[t_adj] == 'user' and t_adj != n:
                        user_adjlist_dict[n].add(t_adj)
    return movie_adjlist_dict, user_adjlist_dict


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
            if node_type[key] == 'user':
                temp = '0'
            elif node_type[key] == 'movie':
                temp = '1'
            elif node_type[key] == 'director':
                temp = '2'
            else:
                temp = '3'
            fw.write(str(key) + ' ' + temp)
            fw.write('\n')


def save_gt_info(node_type, save_dir):
    with open(save_dir, 'w') as fw:
        for key in node_type.keys():
            fw.write(str(key))
            for g in node_type[key]:
                fw.write(' ' + str(g))
            fw.write('\n')


def save_gt_info1(node_type, save_dir):
    with open(save_dir, 'w') as fw:
        for key in node_type.keys():
            fw.write(str(key) + ' ' + node_type[key])
            fw.write('\n')


def select_remove_edge(graph, test_node_set, node_type):
    remove_edge_set = set()
    for n in test_node_set:
        temp = []
        for adj in graph[n]:
            if node_type[adj] == 'user':
                temp.append(adj)
        if len(temp) > 1:
            author_node, _, _, _ = train_test_split(temp, range(len(temp)), train_size=1, random_state=19)
            remove_edge_set.add((n, author_node[0]))
    return remove_edge_set


def select_fake_edge(graph, test_node_set, node_type):
    author_node_list = []
    for n in graph:
        if node_type == 'user':
            author_node_list.append(n)
    fake_edge_set = set()
    for n in test_node_set:
        temp = []
        for adj in graph[n]:
            if node_type[adj] == 'user':
                temp.append(adj)
        if len(temp) > 1:
            for i in range(5):
                tail_node = random.sample(list(graph.keys()), 1)[0]
                while tail_node == n or tail_node in graph[n] or (n, tail_node) in fake_edge_set:
                    tail_node = random.sample(list(graph.keys()), 1)[0]
                fake_edge_set.add((n, tail_node))
    return fake_edge_set


def save_edge(edge_set, edge_save_dir):
    with open(edge_save_dir, 'w') as fw:
        for n in edge_set:
            fw.write(str(n[0]) + ' ' + str(n[1]) + '\n')


if __name__ == '__main__':
    tail_user_set = set()
    tail_movie_set = set()
    graph, node_type, labeled_movie = load_graph_info()
    node_num, tail_node_num, labeled_user = graph_stat(graph, node_type, labeled_movie)
    train_node_set, test_node_set = select_train_and_test_node(graph, node_type)
    true_edge = select_remove_edge(graph, test_node_set, node_type)
    fake_edge = select_fake_edge(graph, test_node_set, node_type)
    save_edge(true_edge, './link_prediction/' + '/true_edge.txt')
    save_edge(fake_edge, './link_prediction/' + '/fake_edge.txt')
    # save_graph(graph, './data/' + dataset + '/graph.adjlist')
    # save_node(train_node_set, './data/' + dataset + '/train.csv')
    # save_node(test_node_set, './data/' + dataset + '/test.csv')
    # movie_adjlist, user_adjlist = movie_and_user_adjlist(graph, node_type)
    # save_graph(user_adjlist, './data/' + dataset + '/type1.adjlist')
    # save_graph(movie_adjlist, './data/' + dataset + '/type2.adjlist')
    # save_node_type_info(node_type, './data/' + dataset + '/node_type.txt')
    # print(labeled_movie)
    # print(labeled_user)
    # print(len(labeled_movie))
    # print(len(labeled_user))
    # print(len(set(labeled_user.keys()) & set(tail_user_set)))
    # print(len(set(labeled_movie.keys()) & set(tail_movie_set)))
    # save_gt_info(labeled_movie, './data/' + dataset + '/gt_m.txt')
    # save_gt_info(labeled_user, './data/' + dataset + '/gt_u.txt')
