import numpy as np
import tensorflow as tf


class MAML:
    def __init__(self, d, meta_lr=0.001, train_lr=0.01):
        self.d = d
        self.meta_lr = meta_lr
        self.train_lr = train_lr
        # self.node_type_dir = './data/dblp/node_type.txt'
        #
        # self.node_type = dict()
        # with open(self.node_type_dir, 'r') as fr:
        #     lines = fr.readlines()
        #     for line in lines:
        #         temp = list(line.strip('\n').split(' '))
        #         self.node_type[temp[0]] = temp[1:]

        print('embedding shape:', self.d, 'meta-lr:', meta_lr, 'train-lr:', train_lr)

    def build(self, support_nb, support_xb, support_yb, query_nb, query_xb, query_yb, k, meta_batchsz, mode='train'):
        self.weights = self.conv_weights()
        training = True if mode is 'train' else False

        def meta_task(input):
            c = tf.constant(0.5, dtype=tf.float32)
            support_n, support_x, support_y, query_n, query_x, query_y = input
            support_x, s_surr_o, s_surr_p = tf.split(support_x, [128 * 4, 4, 4], axis=1)
            query_x, q_surr_o, q_surr_p = tf.split(query_x, [128 * 4, 4, 4], axis=1)
            s_surr_o = tf.expand_dims(s_surr_o, -1)
            s_surr_p = tf.expand_dims(s_surr_p, -1)
            q_surr_o = tf.expand_dims(q_surr_o, -1)
            q_surr_p = tf.expand_dims(q_surr_p, -1)

            support_n, support_n_1 = tf.split(support_n, [1] * 2, axis=1)
            query_c, query_n = tf.split(query_n, [1] * 2, axis=1)
            # wa = tf.cond(tf.less(tf.reduce_mean(query_c), c), lambda: self.weights['wa2a'],
                         # lambda: self.weights['wa2p'])
            # wp = tf.cond(tf.less(tf.reduce_mean(query_c), c), lambda: self.weights['wp2a'],
                         # lambda: self.weights['wp2p'])
            # wt = tf.cond(tf.less(tf.reduce_mean(query_c), c), lambda: self.weights['wt2a'],
                         # lambda: self.weights['wt2p'])
            # wc = tf.cond(tf.less(tf.reduce_mean(query_c), c), lambda: self.weights['wc2a'],
                         # lambda: self.weights['wc2p'])
            # self.weights['w1_gamma_1'] -> [128,4]; surr_o -> [5,4,1];
            # gamma_w1 -> [5,128,1]; self.weights['w1_s'] -> [5,128,1024]
            gamma_w1 = tf.nn.leaky_relu(tf.einsum('ij,ajl->ail', self.weights['w1_gamma_1'], s_surr_o) +
                                        tf.einsum('ij,ajl->ail', self.weights['w1_gamma_2'], s_surr_p))
            beta_w1 = tf.nn.leaky_relu(tf.einsum('ij,ajl->ail', self.weights['w1_beta_1'], s_surr_o) +
                                       tf.einsum('ij,ajl->ail', self.weights['w1_beta_2'], s_surr_p))
            w1_s = tf.multiply(self.weights['w1'], gamma_w1 + 1) + beta_w1
            # self.weights['out_w_gamma_1'] -> [1024,4]; surr_o -> [5,4,1];
            # gamma_out_w -> [5,1024,1]; self.weights['out_w_s'] -> [5,1024,128]
            gamma_out_w = tf.nn.leaky_relu(
                tf.einsum('ij,ajl->ail', self.weights['out_w_gamma_1'], s_surr_o) +
                tf.einsum('ij,ajl->ail', self.weights['out_w_gamma_2'], s_surr_p))
            beta_out_w = tf.nn.leaky_relu(tf.einsum('ij,ajl->ail', self.weights['out_w_beta_1'], s_surr_o) +
                                          tf.einsum('ij,ajl->ail', self.weights['out_w_beta_2'], s_surr_p))
            out_w_s = tf.multiply(self.weights['out_w'], gamma_out_w + 1) + beta_out_w
            # self.weights['b1_gamma_1'] -> [1024,4]; surr_o -> [5,4,1];
            # gamma_b1 -> [5,1024]; self.weights['b1_s'] -> [5,1024]
            gamma_b1 = tf.squeeze(tf.nn.leaky_relu(tf.einsum('ij,ajl->ail', self.weights['b1_gamma_1'], s_surr_o) +
                                                   tf.einsum('ij,ajl->ail', self.weights['b1_gamma_2'], s_surr_p)))
            beta_b1 = tf.squeeze(tf.nn.leaky_relu(tf.einsum('ij,ajl->ail', self.weights['b1_beta_1'], s_surr_o) +
                                                  tf.einsum('ij,ajl->ail', self.weights['b1_beta_2'], s_surr_p)))
            b1_s = tf.multiply(self.weights['b1'], gamma_b1 + 1) + beta_b1
            # self.weights['out_b_gamma_1'] -> [128,4]; surr_o -> [5,4,1];
            # gamma_out_b -> [5,128]; self.weights['out_b_s'] -> [5,128]
            gamma_out_b = tf.squeeze(
                tf.nn.leaky_relu(tf.einsum('ij,ajl->ail', self.weights['out_b_gamma_1'], s_surr_o) +
                                 tf.einsum('ij,ajl->ail', self.weights['out_b_gamma_2'], s_surr_p)))
            beta_out_b = tf.squeeze(
                tf.nn.leaky_relu(tf.einsum('ij,ajl->ail', self.weights['out_b_beta_1'], s_surr_o) +
                                 tf.einsum('ij,ajl->ail', self.weights['out_b_beta_2'], s_surr_p)))
            out_b_s = tf.multiply(self.weights['out_b'], gamma_out_b + 1) + beta_out_b

            sx1, sx2, sx3, sx4 = tf.split(support_x, [128] * 4, axis=1)
            qx1, qx2, qx3, qx4 = tf.split(query_x, [128] * 4, axis=1)
            # sx1 = tf.matmul(sx1, wa)
            # sx2 = tf.matmul(sx2, wp)
            # sx3 = tf.matmul(sx3, wt)
            # sx4 = tf.matmul(sx4, wc)
            support_x = tf.divide(tf.add_n([sx1, sx2, sx3, sx4]), 4)
            # support_x = tf.div((sx1 + sx2 + sx3 + sx4), 4)
            # qx1 = tf.matmul(qx1, wa)
            # qx2 = tf.matmul(qx2, wp)
            # qx3 = tf.matmul(qx3, wt)
            # qx4 = tf.matmul(qx4, wc)
            query_x = tf.divide(tf.add_n([qx1, qx2, qx3, qx4]), 4)

            query_preds, query_losses, query_nodes = [], [], []
            support_pred = self.forward(support_x, w1_s, b1_s, out_w_s, out_b_s)
            if training:
                support_loss = tf.losses.mean_squared_error(support_y, support_pred) + 0.1 * (
                        tf.nn.l2_loss(self.weights['w1_gamma_1']) + tf.nn.l2_loss(self.weights['w1_gamma_2']) +
                        tf.nn.l2_loss(self.weights['w1_beta_1']) + tf.nn.l2_loss(self.weights['w1_beta_2']) +
                        tf.nn.l2_loss(self.weights['out_w_gamma_1']) + tf.nn.l2_loss(self.weights['out_w_gamma_2']) +
                        tf.nn.l2_loss(self.weights['out_w_beta_1']) + tf.nn.l2_loss(self.weights['out_w_beta_2']) +
                        tf.nn.l2_loss(self.weights['b1_gamma_1']) + tf.nn.l2_loss(self.weights['b1_gamma_2']) +
                        tf.nn.l2_loss(self.weights['b1_beta_1']) + tf.nn.l2_loss(self.weights['b1_beta_2']) +
                        tf.nn.l2_loss(self.weights['out_b_gamma_1']) + tf.nn.l2_loss(self.weights['out_b_gamma_2']) +
                        tf.nn.l2_loss(self.weights['out_b_beta_1']) + tf.nn.l2_loss(self.weights['out_b_beta_2'])) / 16
                # support_loss = tf.losses.mean_squared_error(support_y, support_pred) + 1 * (
                #         tf.nn.l2_loss(self.weights['w1']) + tf.nn.l2_loss(self.weights['out_w'])) / 2
            else:
                idx = tf.reshape(tf.where(tf.reshape(support_n[0], [-1]) > 0), [-1])
                support_loss = tf.losses.mean_squared_error(tf.gather(support_y, idx),
                                                            tf.gather(support_pred, idx)) + 0.1 * (
                        tf.nn.l2_loss(self.weights['w1_gamma_1']) + tf.nn.l2_loss(self.weights['w1_gamma_2']) +
                        tf.nn.l2_loss(self.weights['w1_beta_1']) + tf.nn.l2_loss(self.weights['w1_beta_2']) +
                        tf.nn.l2_loss(self.weights['out_w_gamma_1']) + tf.nn.l2_loss(self.weights['out_w_gamma_2']) +
                        tf.nn.l2_loss(self.weights['out_w_beta_1']) + tf.nn.l2_loss(self.weights['out_w_beta_2']) +
                        tf.nn.l2_loss(self.weights['b1_gamma_1']) + tf.nn.l2_loss(self.weights['b1_gamma_2']) +
                        tf.nn.l2_loss(self.weights['b1_beta_1']) + tf.nn.l2_loss(self.weights['b1_beta_2']) +
                        tf.nn.l2_loss(self.weights['out_b_gamma_1']) + tf.nn.l2_loss(self.weights['out_b_gamma_2']) +
                        tf.nn.l2_loss(self.weights['out_b_beta_1']) + tf.nn.l2_loss(self.weights['out_b_beta_2'])) / 16

            grads = tf.gradients(support_loss, list(self.weights.values()))
            gvs = dict(zip(self.weights.keys(), grads))
            for var in gvs:
                gvs[var] = tf.clip_by_norm(gvs[var], 10)
            # gvs = [(tf.clip_by_norm(grad, 10), var) for grad, var in gvs]

            fast_weights = dict(
                zip(self.weights.keys(), [self.weights[key] - self.train_lr * gvs[key] for key in self.weights.keys()]))

            # fast_weights_1 = dict(
            #     zip(self.weights.keys(), [self.weights[key] - self.train_lr * gvs[key] for
            #     key in self.weights.keys()]))
            # fast_weights_2 = dict(
            #     zip(self.s_weights.keys(),
            #         [self.s_weights[key] - self.train_lr * gvs[key] for key in self.s_weights.keys()]))
            # fast_weights = dict()
            # fast_weights.update(fast_weights_1)
            # fast_weights.update(fast_weights_2)

            # self.weights['w1_gamma_1'] -> [128,4]; surr_o -> [1,4,1];
            # gamma_w1 -> [1,128,1]; self.weights['w1_q'] -> [1,128,1024]
            gamma_w1 = tf.nn.leaky_relu(tf.einsum('ij,ajl->ail', fast_weights['w1_gamma_1'], q_surr_o) +
                                        tf.einsum('ij,ajl->ail', fast_weights['w1_gamma_2'], q_surr_p))
            beta_w1 = tf.nn.leaky_relu(tf.einsum('ij,ajl->ail', fast_weights['w1_beta_1'], q_surr_o) +
                                       tf.einsum('ij,ajl->ail', fast_weights['w1_beta_2'], q_surr_p))
            w1_q = tf.multiply(fast_weights['w1'], gamma_w1 + 1) + beta_w1
            # self.weights['out_w_gamma_1'] -> [1024,4]; surr_o -> [1,4,1];
            # gamma_out_w -> [1,1024,1]; self.weights['out_w_q'] -> [1,1024,128]
            gamma_out_w = tf.nn.leaky_relu(
                tf.einsum('ij,ajl->ail', fast_weights['out_w_gamma_1'], q_surr_o) +
                tf.einsum('ij,ajl->ail', fast_weights['out_w_gamma_2'], q_surr_p))
            beta_out_w = tf.nn.leaky_relu(tf.einsum('ij,ajl->ail', fast_weights['out_w_beta_1'], q_surr_o) +
                                          tf.einsum('ij,ajl->ail', fast_weights['out_w_beta_2'], q_surr_p))
            out_w_q = tf.multiply(fast_weights['out_w'], gamma_out_w + 1) + beta_out_w
            # self.weights['b1_gamma_1'] -> [1024,4]; surr_o -> [1,4,1];
            # gamma_b1 -> [1,1024]; self.weights['b1_q'] -> [1,1024]
            gamma_b1 = tf.squeeze(tf.nn.leaky_relu(tf.einsum('ij,ajl->ail', fast_weights['b1_gamma_1'], q_surr_o) +
                                                   tf.einsum('ij,ajl->ail', fast_weights['b1_gamma_2'], q_surr_p)))
            beta_b1 = tf.squeeze(tf.nn.leaky_relu(tf.einsum('ij,ajl->ail', fast_weights['b1_beta_1'], q_surr_o) +
                                                  tf.einsum('ij,ajl->ail', fast_weights['b1_beta_2'], q_surr_p)))
            b1_q = tf.multiply(fast_weights['b1'], gamma_b1 + 1) + beta_b1
            # self.weights['out_b_gamma_1'] -> [128,4]; surr_o -> [1,4,1];
            # gamma_out_b -> [1,128]; self.weights['out_b_q'] -> [1,128]
            gamma_out_b = tf.squeeze(
                tf.nn.leaky_relu(tf.einsum('ij,ajl->ail', fast_weights['out_b_gamma_1'], q_surr_o) +
                                 tf.einsum('ij,ajl->ail', fast_weights['out_b_gamma_2'], q_surr_p)))
            beta_out_b = tf.squeeze(
                tf.nn.leaky_relu(tf.einsum('ij,ajl->ail', fast_weights['out_b_beta_1'], q_surr_o) +
                                 tf.einsum('ij,ajl->ail', fast_weights['out_b_beta_2'], q_surr_p)))
            out_b_q = tf.multiply(fast_weights['out_b'], gamma_out_b + 1) + beta_out_b

            query_pred = self.forward(query_x, w1_q, b1_q, out_w_q, out_b_q)
            query_loss = tf.losses.mean_squared_error(query_y, query_pred)
            query_pred = tf.reshape(query_pred, [-1])
            query_n = tf.reshape(query_n, [-1])
            query_preds.append(query_pred)
            query_nodes.append(query_n)
            query_losses.append(query_loss)
            fast_weights_k = dict()
            fast_weights_k['w1'] = fast_weights['w1']
            fast_weights_k['b1'] = fast_weights['b1']
            fast_weights_k['out_w'] = fast_weights['out_w']
            fast_weights_k['out_b'] = fast_weights['out_b']
            fast_weights_k['w1_gamma_1'] = fast_weights['w1_gamma_1']
            fast_weights_k['w1_gamma_2'] = fast_weights['w1_gamma_2']
            fast_weights_k['w1_beta_1'] = fast_weights['w1_beta_1']
            fast_weights_k['w1_beta_2'] = fast_weights['w1_beta_2']
            fast_weights_k['out_w_gamma_1'] = fast_weights['out_w_gamma_1']
            fast_weights_k['out_w_gamma_2'] = fast_weights['out_w_gamma_2']
            fast_weights_k['out_w_beta_1'] = fast_weights['out_w_beta_1']
            fast_weights_k['out_w_beta_2'] = fast_weights['out_w_beta_2']
            fast_weights_k['b1_gamma_1'] = fast_weights['b1_gamma_1']
            fast_weights_k['b1_gamma_2'] = fast_weights['b1_gamma_2']
            fast_weights_k['b1_beta_1'] = fast_weights['b1_beta_1']
            fast_weights_k['b1_beta_2'] = fast_weights['b1_beta_2']
            fast_weights_k['out_b_gamma_1'] = fast_weights['out_b_gamma_1']
            fast_weights_k['out_b_gamma_2'] = fast_weights['out_b_gamma_2']
            fast_weights_k['out_b_beta_1'] = fast_weights['out_b_beta_1']
            fast_weights_k['out_b_beta_2'] = fast_weights['out_b_beta_2']

            for _ in range(1, k):
                # self.weights['w1_gamma_1'] -> [128,4]; surr_o -> [5,4,1];
                # gamma_w1 -> [5,128,1]; self.weights['w1_s'] -> [5,128,1024]
                gamma_w1 = tf.nn.leaky_relu(tf.einsum('ij,ajl->ail', fast_weights_k['w1_gamma_1'], s_surr_o) +
                                            tf.einsum('ij,ajl->ail', fast_weights_k['w1_gamma_2'], s_surr_p))
                beta_w1 = tf.nn.leaky_relu(tf.einsum('ij,ajl->ail', fast_weights_k['w1_beta_1'], s_surr_o) +
                                           tf.einsum('ij,ajl->ail', fast_weights_k['w1_beta_2'], s_surr_p))
                w1_s = tf.multiply(fast_weights_k['w1'], gamma_w1 + 1) + beta_w1
                # self.weights['out_w_gamma_1'] -> [1024,4]; surr_o -> [5,4,1];
                # gamma_out_w -> [5,1024,1]; self.weights['out_w_s'] -> [5,1024,128]
                gamma_out_w = tf.nn.leaky_relu(
                    tf.einsum('ij,ajl->ail', fast_weights_k['out_w_gamma_1'], s_surr_o) +
                    tf.einsum('ij,ajl->ail', fast_weights_k['out_w_gamma_2'], s_surr_p))
                beta_out_w = tf.nn.leaky_relu(tf.einsum('ij,ajl->ail', fast_weights_k['out_w_beta_1'], s_surr_o) +
                                              tf.einsum('ij,ajl->ail', fast_weights_k['out_w_beta_2'], s_surr_p))
                out_w_s = tf.multiply(fast_weights_k['out_w'], gamma_out_w + 1) + beta_out_w
                # self.weights['b1_gamma_1'] -> [1024,4]; surr_o -> [5,4,1];
                # gamma_b1 -> [5,1024]; self.weights['b1_s'] -> [5,1024]
                gamma_b1 = tf.squeeze(
                    tf.nn.leaky_relu(tf.einsum('ij,ajl->ail', fast_weights_k['b1_gamma_1'], s_surr_o) +
                                     tf.einsum('ij,ajl->ail', fast_weights_k['b1_gamma_2'], s_surr_p)))
                beta_b1 = tf.squeeze(tf.nn.leaky_relu(tf.einsum('ij,ajl->ail', fast_weights_k['b1_beta_1'], s_surr_o) +
                                                      tf.einsum('ij,ajl->ail', fast_weights_k['b1_beta_2'], s_surr_p)))
                b1_s = tf.multiply(fast_weights_k['b1'], gamma_b1 + 1) + beta_b1
                # self.weights['out_b_gamma_1'] -> [128,4]; surr_o -> [5,4,1];
                # gamma_out_b -> [5,128]; self.weights['out_b_s'] -> [5,128]
                gamma_out_b = tf.squeeze(
                    tf.nn.leaky_relu(tf.einsum('ij,ajl->ail', fast_weights_k['out_b_gamma_1'], s_surr_o) +
                                     tf.einsum('ij,ajl->ail', fast_weights_k['out_b_gamma_2'], s_surr_p)))
                beta_out_b = tf.squeeze(
                    tf.nn.leaky_relu(tf.einsum('ij,ajl->ail', fast_weights_k['out_b_beta_1'], s_surr_o) +
                                     tf.einsum('ij,ajl->ail', fast_weights_k['out_b_beta_2'], s_surr_p)))
                out_b_s = tf.multiply(fast_weights_k['out_b'], gamma_out_b + 1) + beta_out_b
                if training:
                    loss = tf.losses.mean_squared_error(support_y,
                                                        self.forward(support_x, w1_s, b1_s, out_w_s, out_b_s)) + 0.1 * (
                        tf.nn.l2_loss(self.weights['w1_gamma_1']) + tf.nn.l2_loss(self.weights['w1_gamma_2']) +
                        tf.nn.l2_loss(self.weights['w1_beta_1']) + tf.nn.l2_loss(self.weights['w1_beta_2']) +
                        tf.nn.l2_loss(self.weights['out_w_gamma_1']) + tf.nn.l2_loss(self.weights['out_w_gamma_2']) +
                        tf.nn.l2_loss(self.weights['out_w_beta_1']) + tf.nn.l2_loss(self.weights['out_w_beta_2']) +
                        tf.nn.l2_loss(self.weights['b1_gamma_1']) + tf.nn.l2_loss(self.weights['b1_gamma_2']) +
                        tf.nn.l2_loss(self.weights['b1_beta_1']) + tf.nn.l2_loss(self.weights['b1_beta_2']) +
                        tf.nn.l2_loss(self.weights['out_b_gamma_1']) + tf.nn.l2_loss(self.weights['out_b_gamma_2']) +
                        tf.nn.l2_loss(self.weights['out_b_beta_1']) + tf.nn.l2_loss(self.weights['out_b_beta_2'])) / 16
                else:
                    idx = tf.reshape(tf.where(tf.reshape(support_n[0], [-1]) > 0), [-1])
                    loss = tf.losses.mean_squared_error(tf.gather(support_y, idx), tf.gather(
                        self.forward(support_x, w1_s, b1_s, out_w_s, out_b_s), idx)) + 0.1 * (
                        tf.nn.l2_loss(self.weights['w1_gamma_1']) + tf.nn.l2_loss(self.weights['w1_gamma_2']) +
                        tf.nn.l2_loss(self.weights['w1_beta_1']) + tf.nn.l2_loss(self.weights['w1_beta_2']) +
                        tf.nn.l2_loss(self.weights['out_w_gamma_1']) + tf.nn.l2_loss(self.weights['out_w_gamma_2']) +
                        tf.nn.l2_loss(self.weights['out_w_beta_1']) + tf.nn.l2_loss(self.weights['out_w_beta_2']) +
                        tf.nn.l2_loss(self.weights['b1_gamma_1']) + tf.nn.l2_loss(self.weights['b1_gamma_2']) +
                        tf.nn.l2_loss(self.weights['b1_beta_1']) + tf.nn.l2_loss(self.weights['b1_beta_2']) +
                        tf.nn.l2_loss(self.weights['out_b_gamma_1']) + tf.nn.l2_loss(self.weights['out_b_gamma_2']) +
                        tf.nn.l2_loss(self.weights['out_b_beta_1']) + tf.nn.l2_loss(self.weights['out_b_beta_2'])) / 16
                grads = tf.gradients(loss, list(fast_weights_k.values()))
                gvs = dict(zip(fast_weights_k.keys(), grads))
                for var in gvs:
                    gvs[var] = tf.clip_by_norm(gvs[var], 10)
                fast_weights_k = dict(zip(fast_weights_k.keys(), [fast_weights_k[key] - self.train_lr * gvs[key]
                                                                  for key in fast_weights_k.keys()]))

                # self.weights['w1_gamma_1'] -> [128,4]; surr_o -> [1,4,1];
                # gamma_w1 -> [1,128,1]; self.weights['w1_q'] -> [1,128,1024]
                gamma_w1 = tf.nn.leaky_relu(tf.einsum('ij,ajl->ail', fast_weights_k['w1_gamma_1'], q_surr_o) +
                                            tf.einsum('ij,ajl->ail', fast_weights_k['w1_gamma_2'], q_surr_p))
                beta_w1 = tf.nn.leaky_relu(tf.einsum('ij,ajl->ail', fast_weights_k['w1_beta_1'], q_surr_o) +
                                           tf.einsum('ij,ajl->ail', fast_weights_k['w1_beta_2'], q_surr_p))
                w1_q = tf.multiply(fast_weights_k['w1'], gamma_w1 + 1) + beta_w1
                # self.weights['out_w_gamma_1'] -> [1024,4]; surr_o -> [1,4,1];
                # gamma_out_w -> [1,1024,1]; self.weights['out_w_q'] -> [1,1024,128]
                gamma_out_w = tf.nn.leaky_relu(
                    tf.einsum('ij,ajl->ail', fast_weights_k['out_w_gamma_1'], q_surr_o) +
                    tf.einsum('ij,ajl->ail', fast_weights_k['out_w_gamma_2'], q_surr_p))
                beta_out_w = tf.nn.leaky_relu(tf.einsum('ij,ajl->ail', fast_weights_k['out_w_beta_1'], q_surr_o) +
                                              tf.einsum('ij,ajl->ail', fast_weights_k['out_w_beta_2'], q_surr_p))
                out_w_q = tf.multiply(fast_weights_k['out_w'], gamma_out_w + 1) + beta_out_w
                # self.weights['b1_gamma_1'] -> [1024,4]; surr_o -> [1,4,1];
                # gamma_b1 -> [1,1024]; self.weights['b1_q'] -> [1,1024]
                gamma_b1 = tf.squeeze(
                    tf.nn.leaky_relu(tf.einsum('ij,ajl->ail', fast_weights_k['b1_gamma_1'], q_surr_o) +
                                     tf.einsum('ij,ajl->ail', fast_weights_k['b1_gamma_2'], q_surr_p)))
                beta_b1 = tf.squeeze(tf.nn.leaky_relu(tf.einsum('ij,ajl->ail', fast_weights_k['b1_beta_1'], q_surr_o) +
                                                      tf.einsum('ij,ajl->ail', fast_weights_k['b1_beta_2'], q_surr_p)))
                b1_q = tf.multiply(fast_weights_k['b1'], gamma_b1 + 1) + beta_b1
                # self.weights['out_b_gamma_1'] -> [128,4]; surr_o -> [1,4,1];
                # gamma_out_b -> [1,128]; self.weights['out_b_q'] -> [1,128]
                gamma_out_b = tf.squeeze(
                    tf.nn.leaky_relu(tf.einsum('ij,ajl->ail', fast_weights_k['out_b_gamma_1'], q_surr_o) +
                                     tf.einsum('ij,ajl->ail', fast_weights_k['out_b_gamma_2'], q_surr_p)))
                beta_out_b = tf.squeeze(
                    tf.nn.leaky_relu(tf.einsum('ij,ajl->ail', fast_weights_k['out_b_beta_1'], q_surr_o) +
                                     tf.einsum('ij,ajl->ail', fast_weights_k['out_b_beta_2'], q_surr_p)))
                out_b_q = tf.multiply(fast_weights_k['out_b'], gamma_out_b + 1) + beta_out_b

                query_pred = self.forward(query_x, w1_q, b1_q, out_w_q, out_b_q)
                query_loss = tf.losses.mean_squared_error(query_y, query_pred)
                query_pred = tf.reshape(query_pred, [-1])
                query_n = tf.reshape(query_n, [-1])
                query_preds.append(query_pred)
                query_nodes.append(query_n)
                query_losses.append(query_loss)

            result = [support_pred, support_loss, query_preds, query_losses, query_nodes]
            return result

        out_dtype = [tf.float32, tf.float32, [tf.float32] * k, [tf.float32] * k, [tf.float32] * k]
        result = tf.map_fn(meta_task, elems=(support_nb, support_xb, support_yb, query_nb, query_xb, query_yb),
                           dtype=out_dtype, name='map_fn')
        support_pred_tasks, support_loss_tasks, query_preds_tasks, query_losses_tasks, query_nodes = result

        if mode is 'train':
            self.support_loss = support_loss = tf.reduce_sum(support_loss_tasks) / meta_batchsz
            self.query_losses = query_losses = [tf.reduce_sum(query_losses_tasks[j]) / meta_batchsz
                                                for j in range(k)]

            optimizer = tf.train.AdamOptimizer(self.meta_lr, name='meta_optim')
            gvs = optimizer.compute_gradients(self.query_losses[-1])
            gvs = [(tf.clip_by_norm(grad, 10), var) for grad, var in gvs]
            self.meta_op = optimizer.apply_gradients(gvs)

        else:
            self.test_support_loss = support_loss = tf.reduce_sum(support_loss_tasks) / meta_batchsz
            self.test_query_losses = query_losses = [tf.reduce_sum(query_losses_tasks[j]) / meta_batchsz
                                                     for j in range(k)]
            self.test_query_preds = query_preds_tasks
            self.query_nodes = query_nodes

        tf.summary.scalar(mode + ':support loss', support_loss)
        for j in range(k):
            tf.summary.scalar(mode + ':query loss, step ' + str(j + 1), query_losses[j])

    # def conv_weights(self):
    #     weights = {}
    #     fc_initializer = tf.contrib.layers.xavier_initializer()
    #     with tf.variable_scope('MAML', reuse=tf.AUTO_REUSE):
    #         weights['w1'] = tf.get_variable('w1', [128, 1024], initializer=fc_initializer)
    #         weights['b1'] = tf.get_variable('b1', initializer=tf.zeros([1024]))
    #         weights['out_w'] = tf.get_variable('out_w', [1024, 128], initializer=fc_initializer)
    #         weights['out_b'] = tf.get_variable('out_b', initializer=tf.zeros([128]))
    #     return weights

    # weights func for dblp dataset
    def conv_weights(self):
        weights = {}
        a_weights = {}
        fc_initializer = tf.contrib.layers.xavier_initializer()
        with tf.variable_scope('MAML', reuse=tf.AUTO_REUSE):
            weights['w1'] = tf.get_variable('w1', [128, 1024], initializer=fc_initializer)
            tf.summary.histogram('w1', weights['w1'])
            weights['b1'] = tf.get_variable('b1', initializer=tf.zeros([1024]))
            tf.summary.histogram('b1', weights['b1'])
            weights['out_w'] = tf.get_variable('out_w', [1024, 128], initializer=fc_initializer)
            tf.summary.histogram('out_w', weights['out_w'])
            weights['out_b'] = tf.get_variable('out_b', initializer=tf.zeros([128]))
            tf.summary.histogram('out_b', weights['out_b'])
            # weights['wa'] = tf.get_variable('wa', [128, 128], initializer=fc_initializer)
            # tf.summary.histogram('wa', weights['wa'])
            # weights['wp'] = tf.get_variable('wp', [128, 128], initializer=fc_initializer)
            # tf.summary.histogram('wp', weights['wp'])
            # weights['wt'] = tf.get_variable('wt', [128, 128], initializer=fc_initializer)
            # tf.summary.histogram('wt', weights['wt'])
            # weights['wc'] = tf.get_variable('wc', [128, 128], initializer=fc_initializer)
            # tf.summary.histogram('wc', weights['wc'])
            # weights['wa2a'] = tf.get_variable('wa2a', [128, 128], initializer=fc_initializer)
            # tf.summary.histogram('wa2a', weights['wa2a'])
            # weights['wp2a'] = tf.get_variable('wp2a', [128, 128], initializer=fc_initializer)
            # tf.summary.histogram('wp2a', weights['wp2a'])
            # weights['wt2a'] = tf.get_variable('wt2a', [128, 128], initializer=fc_initializer)
            # tf.summary.histogram('wt2a', weights['wt2a'])
            # weights['wc2a'] = tf.get_variable('wc2a', [128, 128], initializer=fc_initializer)
            # tf.summary.histogram('wc2a', weights['wc2a'])
            # weights['wa2p'] = tf.get_variable('wa2p', [128, 128], initializer=fc_initializer)
            # tf.summary.histogram('wa2p', weights['wa2p'])
            # weights['wp2p'] = tf.get_variable('wp2p', [128, 128], initializer=fc_initializer)
            # tf.summary.histogram('wp2p', weights['wp2p'])
            # weights['wt2p'] = tf.get_variable('wt2p', [128, 128], initializer=fc_initializer)
            # tf.summary.histogram('wt2p', weights['wt2p'])
            # weights['wc2p'] = tf.get_variable('wc2p', [128, 128], initializer=fc_initializer)
            # tf.summary.histogram('wc2p', weights['wc2p'])

            weights['w1_gamma_1'] = tf.get_variable('w1_gamma_1', [128, 4], initializer=fc_initializer)
            tf.summary.histogram('w1_gamma_1', weights['w1_gamma_1'])
            weights['w1_gamma_2'] = tf.get_variable('w1_gamma_2', [128, 4], initializer=fc_initializer)
            tf.summary.histogram('w1_gamma_2', weights['w1_gamma_2'])
            weights['w1_beta_1'] = tf.get_variable('w1_beta_1', [128, 4], initializer=fc_initializer)
            tf.summary.histogram('w1_beta_1', weights['w1_beta_1'])
            weights['w1_beta_2'] = tf.get_variable('w1_beta_2', [128, 4], initializer=fc_initializer)
            tf.summary.histogram('w1_beta_2', weights['w1_beta_2'])
            weights['out_w_gamma_1'] = tf.get_variable('out_w_gamma_1', [1024, 4], initializer=fc_initializer)
            tf.summary.histogram('out_w_gamma_1', weights['out_w_gamma_1'])
            weights['out_w_gamma_2'] = tf.get_variable('out_w_gamma_2', [1024, 4], initializer=fc_initializer)
            tf.summary.histogram('out_w_gamma_2', weights['out_w_gamma_2'])
            weights['out_w_beta_1'] = tf.get_variable('out_w_beta_1', [1024, 4], initializer=fc_initializer)
            tf.summary.histogram('out_w_beta_1', weights['out_w_beta_1'])
            weights['out_w_beta_2'] = tf.get_variable('out_w_beta_2', [1024, 4], initializer=fc_initializer)
            tf.summary.histogram('out_w_beta_2', weights['out_w_beta_2'])
            weights['b1_gamma_1'] = tf.get_variable('b1_gamma_1', [1024, 4], initializer=fc_initializer)
            tf.summary.histogram('b1_gamma_1', weights['b1_gamma_1'])
            weights['b1_gamma_2'] = tf.get_variable('b1_gamma_2', [1024, 4], initializer=fc_initializer)
            tf.summary.histogram('b1_gamma_2', weights['b1_gamma_2'])
            weights['b1_beta_1'] = tf.get_variable('b1_beta_1', [1024, 4], initializer=fc_initializer)
            tf.summary.histogram('b1_beta_1', weights['b1_beta_1'])
            weights['b1_beta_2'] = tf.get_variable('b1_beta_2', [1024, 4], initializer=fc_initializer)
            tf.summary.histogram('b1_beta_2', weights['b1_beta_2'])
            weights['out_b_gamma_1'] = tf.get_variable('out_b_gamma_1', [128, 4], initializer=fc_initializer)
            tf.summary.histogram('out_b_gamma_1', weights['out_b_gamma_1'])
            weights['out_b_gamma_2'] = tf.get_variable('out_b_gamma_2', [128, 4], initializer=fc_initializer)
            tf.summary.histogram('out_b_gamma_2', weights['out_b_gamma_2'])
            weights['out_b_beta_1'] = tf.get_variable('out_b_beta_1', [128, 4], initializer=fc_initializer)
            tf.summary.histogram('out_b_beta_1', weights['out_b_beta_1'])
            weights['out_b_beta_2'] = tf.get_variable('out_b_beta_2', [128, 4], initializer=fc_initializer)
            tf.summary.histogram('out_b_beta_2', weights['out_b_beta_2'])

            # s_weights['w1_s'] = tf.get_variable('w1_s', [5, 128, 1024], initializer=fc_initializer)
            # s_weights['b1_s'] = tf.get_variable('b1_s', initializer=tf.zeros([5, 1024]))
            # s_weights['out_w_s'] = tf.get_variable('out_w_s', [5, 1024, 128], initializer=fc_initializer)
            # s_weights['out_b_s'] = tf.get_variable('out_b_s', initializer=tf.zeros([5, 128]))
            # q_weights['w1_q'] = tf.get_variable('w1_q', [1, 128, 1024], initializer=fc_initializer)
            # q_weights['b1_q'] = tf.get_variable('b1_q', initializer=tf.zeros([1, 1024]))
            # q_weights['out_w_q'] = tf.get_variable('out_w_q', [1, 1024, 128], initializer=fc_initializer)
            # q_weights['out_b_q'] = tf.get_variable('out_b_q', initializer=tf.zeros([1, 128]))
        return weights

    # def forward(self, x, weights):
    #     layer_1 = tf.nn.relu(tf.add(tf.matmul(x, weights['w1']), weights['b1']))
    #     out_layer = tf.add(tf.matmul(layer_1, weights['out_w']), weights['out_b'])
    #     return out_layer

    # forward func for dblp dataset
    # def forward(self, x, weights):
    #     layer_1 = tf.nn.relu(tf.add(tf.matmul(x, weights['w1']), weights['b1']))
    #     out_layer = tf.add(tf.matmul(layer_1, weights['out_w']), weights['out_b'])
    #     return out_layer

    def forward(self, x, w1, b1, out_w, out_b):
        layer_1 = tf.nn.relu(tf.add(tf.einsum('aj,ajl->al', x, w1), b1))  # [5,1024]
        out_layer = tf.add(tf.einsum('aj,ajl->al', layer_1, out_w), out_b)
        return out_layer
    #
    # def q_forward(self, x, weights):
    #     layer_1 = tf.nn.relu(tf.add(tf.einsum('aj,ajl->al', x, weights['w1_q']), weights['b1_q']))  # [5,1024]
    #     out_layer = tf.add(tf.einsum('aj,ajl->al', layer_1, weights['out_w_q']), weights['out_b_q'])
    #     return out_layer
