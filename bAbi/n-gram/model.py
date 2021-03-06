# encoding=utf-8
import tensorflow as tf


class LR(object):
    '''
    采用tensorflow实现Logistic Regression
    '''

    def __init__(self, init_learning_rate, vocab_size):
        self.init_learning_rate = init_learning_rate
        self.vocab_size = vocab_size

    def inference(self, x):
        logits = tf.layers.dense(x, self.vocab_size, kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
        return logits

    def loss(self, logits, labels):
        return tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))

    def train(self, loss):
        train_op = tf.train.GradientDescentOptimizer(self.init_learning_rate)
        return train_op.minimize(loss)

    def eval(self, logits, labels):
        correct_prediction = tf.equal(tf.argmax(labels, 1), tf.argmax(logits, 1))
        return tf.reduce_sum(tf.cast(correct_prediction, 'int32'))


class NN(object):
    '''
    实现NN下模型的效果
    '''

    def __init__(self, init_learning_rate, vocab_size):
        self.init_learning_rate = init_learning_rate
        self.vocab_size = vocab_size

    def inference(self, x):
        layer1 = tf.layers.dense(x, 1000, activation=tf.nn.relu,
                                 kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
        layer2 = tf.layers.dense(layer1, 1000, activation=tf.nn.relu,
                                 kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
        logits = tf.layers.dense(layer2, self.vocab_size,
                                 kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
        return logits

    def loss(self, logits, labels):
        return tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))

    def train(self, loss):
        train_op = tf.train.GradientDescentOptimizer(self.init_learning_rate)
        return train_op.minimize(loss)

    def eval(self, logits, labels):
        correct_prediction = tf.equal(tf.argmax(labels, 1), tf.argmax(logits, 1))
        return tf.reduce_sum(tf.cast(correct_prediction, 'int32'))
