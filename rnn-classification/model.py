# coding = utf-8
import tensorflow as tf


class RNN_Classification(object):
    def __init__(self, basic_cell, rnn_size, num_layers, num_labels):
        self._basic_cell = basic_cell
        self._rnn_size = rnn_size
        self._num_layers = num_layers
        self._num_labels = num_labels

    def inference(self, inputs, keep_prob):
        """
        Inference
        """
        cell = self.get_basic_cell(keep_prob)
        final_state = self.run_rnn(cell, inputs)
        logits = self.transform(final_state)
        return logits

    def loss(self, logtis, labels):
        """
        Compute loss
        """
        cost = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logtis)
        return tf.reduce_mean(cost)

    def train(self, loss, learning_rate):
        """
        Clip the grad and return train_op
        """
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        gradients = optimizer.compute_gradients(loss)
        cliped_grads = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients if grad is not None]
        train_op = optimizer.apply_gradients(cliped_grads)
        return train_op

    def eval(self, logits, labels):
        """
        Evaluate prediction
        """
        labels = tf.cast(labels, tf.int64)
        correct = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, axis=1), labels), tf.float32), axis=0)
        return correct

    def get_basic_cell(self, keep_prob):
        """
        Get basic rnn cell
        """
        if self._basic_cell == 'rnn':
            cell = tf.nn.rnn_cell.BasicRNNCell
        elif self._basic_cell == 'lstm':
            cell = tf.nn.rnn_cell.BasicLSTMCell
        elif self._basic_cell == 'gru':
            cell = tf.nn.rnn_cell.GRUCell
        else:
            raise Exception('Unsupported rnn cell: %s' % self._basic_cell)

        def build(rnn_size):
            return cell(rnn_size, kernel_initializer=tf.contrib.layers.xavier_initializer())

        cell = tf.nn.rnn_cell.MultiRNNCell([build(self._rnn_size) for _ in range(self._num_layers)])
        cell = tf.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob=keep_prob)
        return cell

    def run_rnn(self, cell, inputs):
        """
        Run rnn
        """
        outputs, final_state = tf.nn.dynamic_rnn(cell, inputs, dtype=tf.float32)
        if self._basic_cell == 'lstm':
            final_state = final_state[-1]
        return final_state[0]

    def transform(self, final_state):
        """
        Get outputs
        """
        with tf.name_scope('outputs'):
            return tf.layers.dense(final_state, self._num_labels)
