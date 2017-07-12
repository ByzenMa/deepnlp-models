# encoding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


# 电影脚本生成模型
class tv_script_generation(object):
    # 初始化模型参数
    def __init__(self, basic_cell, rnn_size, rnn_layer, embed_dim, seq_length, vocab_size, batch_size, grad_clip):
        self.rnn_size = rnn_size
        self.embed_dim = embed_dim
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        self.basic_cell = basic_cell
        self.rnn_layer = rnn_layer
        self.batch_size = batch_size
        self.grad_clip = grad_clip

    # 模型的推断过程
    def inference(self, input_text):
        # 获得经过embed层的输入
        embed_input = self.get_embed(input_text)
        # 获得rnn单元
        cell, initial_state = self.get_mutli_layer_cell()
        # 建立rnn模型
        outputs, final_state = self.build_rnn(cell, embed_input)
        # 建立softmax层，获得logits，probs
        logits, probs = self.build_nn(outputs)
        return logits, initial_state, final_state, probs

    # 计算模型的损失
    def loss(self, logits, labels, input_shape):
        cost = tf.contrib.seq2seq.sequence_loss(
            logits,
            labels,
            tf.ones([input_shape[0], input_shape[1]]))
        return cost

    # 模型训练
    def train(self, loss, learning_rate):
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), self.grad_clip)
        train_op = tf.train.AdamOptimizer(learning_rate)
        optimizer = train_op.apply_gradients(zip(grads, tvars))
        return optimizer

    # 获得embed层
    def get_embed(self, input_text):
        return tf.contrib.layers.embed_sequence(input_text, self.vocab_size, self.embed_dim)

    # 获得rnn结构
    def get_mutli_layer_cell(self):
        if self.basic_cell == 'rnn':
            cell_fn = tf.nn.rnn_cell.BasicRNNCell
        elif self.basic_cell == 'gru':
            cell_fn = tf.nn.rnn_cell.GRUCell
        elif self.basic_cell == 'lstm':
            cell_fn = tf.nn.rnn_cell.BasicLSTMCell
        else:
            raise Exception('model type not supported: {}'.format(self.basic_cell))

        # 建立单层神经元
        def build_cell(lstm_size):
            cell = cell_fn(lstm_size)
            return cell

        # 建立多层rnn结构
        cell = tf.nn.rnn_cell.MultiRNNCell([build_cell(self.rnn_size) for _ in range(self.rnn_layer)])
        initial_state = cell.zero_state(self.batch_size, tf.float32)
        initial_state = tf.identity(initial_state, name="initial_state")
        return cell, initial_state

    # 获得rnn运行结果
    def build_rnn(self, cell, embed_input):
        outputs, final_state = tf.nn.dynamic_rnn(cell, embed_input, dtype=tf.float32)
        final_state = tf.identity(final_state, name='final_state')
        return outputs, final_state

    # 获得nn层，返回logits和probs
    def build_nn(self, outputs):
        logits = tf.layers.dense(outputs, self.vocab_size)
        probs = tf.nn.softmax(logits)
        return logits, probs
