# encoding = utf-8
import tensorflow as tf
import model
import data_helper as helper
import pickle
import numpy as np
import os

tf.flags.DEFINE_integer('rnn_size', 100, 'rnn size')
tf.flags.DEFINE_integer('num_layers', 1, 'num layers')
tf.flags.DEFINE_integer('num_labels', 2, 'num labels')
tf.flags.DEFINE_string('basic_cell', 'gru', 'basic cell')
tf.flags.DEFINE_float('keep_prob', 0.8, 'keep prob')
tf.flags.DEFINE_float('learning_rate', 0.001, 'learning rate')
tf.flags.DEFINE_integer('batch_size', 64, 'batch size')
tf.flags.DEFINE_integer('num_epochs', 100, 'num epochs')
tf.flags.DEFINE_float('gpu_fraction', 0.9, 'gpu fraction')
tf.flags.DEFINE_float('decay_factor', 1.0, 'learning rate decay factor')
tf.flags.DEFINE_integer('decay_step', 5, 'decay step')

FLAGS = tf.flags.FLAGS
DATA_PAHT = os.path.join(helper.DATA_DIR, helper.PICKLE_NAME)


def load_data():
    """
    Load data
    """
    with open(DATA_PAHT, 'rb') as file:
        data_dict = pickle.load(file)
    print 'X_train shape:', data_dict['X_train'].shape
    print 'y_train shape', data_dict['y_train'].shape
    print 'X_test shape', data_dict['X_test'].shape
    print 'y_test shape', data_dict['y_test'].shape

    return data_dict['X_train'], data_dict['y_train'], data_dict['X_test'], data_dict['y_test']


def get_placeholder():
    """
    Get placeholder
    """
    X_pl = tf.placeholder(tf.float32, shape=[None, helper.MONTH_NUMS, helper.FEATURE_DIM], name='X_placeholder')
    y_pl = tf.placeholder(tf.int32, shape=(None,), name='y_placeholder')
    lr_pl = tf.placeholder(tf.float32, name='learning_rate_placeholder')
    keep_prob_pl = tf.placeholder(tf.float32, name='keep_prob')
    return X_pl, y_pl, lr_pl, keep_prob_pl


def run_train():
    """
    Run training
    """
    X_train, y_train, X_test, y_test = load_data()
    with tf.Graph().as_default() as graph:
        X_pl, y_pl, lr_pl, keep_prob_pl = get_placeholder()
        m = model.RNN_Classification(FLAGS.basic_cell, FLAGS.rnn_size, FLAGS.num_layers, FLAGS.num_labels)
        logits = m.inference(X_pl, keep_prob_pl)
        loss = m.loss(logtis=logits, labels=y_pl)
        eval = m.eval(logits=logits, labels=y_pl)
        train_op = m.train(loss, lr_pl)
        init = tf.global_variables_initializer()
        gpu_op = tf.GPUOptions(per_process_gpu_memory_fraction=FLAGS.gpu_fraction)
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_op), graph=graph) as sess:
            sess.run(init)
            max_test_pr = 0.0
            accumulate_step = 0
            current_lr = FLAGS.learning_rate
            for i in range(FLAGS.num_epochs):

                # If reach decay step, then decay the learning rate
                if accumulate_step >= FLAGS.decay_step:
                    current_lr *= FLAGS.decay_factor

                # Train process
                train_avg_cost = list()
                train_avg_pr = list()
                for X_train_batch, y_train_batch in helper.gen_batch(FLAGS.batch_size, X_train, y_train):
                    train_cost, _, train_pr = sess.run([loss, train_op, eval],
                                                       feed_dict={X_pl: X_train_batch, y_pl: y_train_batch,
                                                                  lr_pl: current_lr,
                                                                  keep_prob_pl: FLAGS.keep_prob})
                    train_avg_cost.append(train_cost)
                    train_avg_pr.append(train_pr)
                _train_cost = np.mean(train_avg_cost)
                _train_pr = np.mean(train_avg_pr)

                # Test process
                test_avg_cost = list()
                test_avg_pr = list()
                for X_test_batch, y_test_batch in helper.gen_batch(FLAGS.batch_size, X_test, y_test):
                    test_cost, test_pr = sess.run([loss, eval],
                                                  feed_dict={X_pl: X_test_batch, y_pl: y_test_batch,
                                                             keep_prob_pl: 1.0})
                    test_avg_cost.append(test_cost)
                    test_avg_pr.append(test_pr)
                _test_cost = np.mean(test_avg_cost)
                _test_pr = np.mean(test_avg_pr)

                # Refresh accumulate step and precision
                if max_test_pr < _test_pr:
                    max_test_pr = _test_pr
                    accumulate_step = 0
                else:
                    accumulate_step += 1

                # Print information
                print 'Epoch=%d lr=%.6f tr_l= %.3f tr_p=%.3f te_l=%.3f te_p=%.3f max_te_p=%.3f' % (
                    i, current_lr, _train_cost, _train_pr, _test_cost, _test_pr, max_test_pr)


if __name__ == '__main__':
    run_train()
