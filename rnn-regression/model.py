# encoding=utf-8
import tensorflow as tf


class RNN_Regression(object):
    def __init__(self, batch_size, rnn_size, num_layers, basic_cell, learning_rate, seq_len, dropout):
        self._batch_size = batch_size
        self._rnn_size = rnn_size
        self._num_layers = num_layers
        self._basic_cell = basic_cell
        self._learning_rate = learning_rate
        self._seq_len = seq_len
        self._dropout = dropout

    def inference(self, input_data):
        '''
        计算输入的logits
        '''
        cell = self.get_basic_cell()
        final_state = self.run_rnn(cell, input_data)
        logits = self.get_logits(final_state)
        return logits

    def loss(self, logits, labels):
        '''
        计算损失
        '''
        return tf.losses.mean_squared_error(labels, logits)

    def train(self, loss):
        '''
        训练模型
        '''
        optimizer = tf.train.AdamOptimizer(self._learning_rate)
        gradients = optimizer.compute_gradients(loss)
        capped_gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients if grad is not None]
        train_op = optimizer.apply_gradients(capped_gradients)
        return train_op

    def eval(self):
        pass

    def get_basic_cell(self):
        '''
        获得基础的rnn单元
        '''
        if self._basic_cell == 'gru':
            cell = tf.nn.rnn_cell.GRUCell
        elif self._basic_cell == 'lstm':
            cell = tf.nn.rnn_cell.BasicLSTMCell
        elif self._basic_cell == 'rnn':
            cell = tf.nn.rnn_cell.BasicRNNCell
        else:
            raise '不支持的RNN类型 %s' % self._basic_cell

        def bulid(rnn_size):
            return cell(rnn_size, kernel_initializer=tf.contrib.layers.xavier_initializer())

        cell = tf.nn.rnn_cell.MultiRNNCell([bulid(self._rnn_size) for _ in range(self._num_layers)])
        cell = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=self._dropout)
        return cell

    def run_rnn(self, cell, input_data):
        '''
        运行rnn单元
        '''
        _, final_state = tf.nn.dynamic_rnn(cell, input_data, dtype=tf.float32)
        if self._basic_cell == 'lstm':
            final_state = final_state[-1]
        return final_state[0]

    def get_logits(self, final_state):
        '''
        由于是回归,因此直接转换成大小为1的logits
        '''
        return tf.layers.dense(final_state, 1)
