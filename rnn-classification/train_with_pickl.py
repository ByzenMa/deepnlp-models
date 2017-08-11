# encoding = utf-8
import tensorflow as tf
import model
import pickle as pickl
import time

tf.flags.DEFINE_integer('batch_size', 64, 'batch size')
tf.flags.DEFINE_integer('num_epochs', 100, 'num epochs')
tf.flags.DEFINE_integer('rnn_size', 100, 'rnn size')
tf.flags.DEFINE_integer('num_layers', 1, 'num layers')
tf.flags.DEFINE_float('keep_prob', 0.8, 'keep prob')
tf.flags.DEFINE_integer('num_labels', 2, 'num labels')
tf.flags.DEFINE_float('init_learning_rate', 0.001, 'init learning rate')
tf.flags.DEFINE_string('basic_cell', 'gru', 'basic cell')
FLAGS = tf.flags.FLAGS

PICKLE_DATA = 'data/data.pckl'


def load_data():
    """
    Load from .pickle data file
    """
    with open(PICKLE_DATA, 'rb') as file:
        data_dict = pickl.load(file)
    print 'X_train shape = ', data_dict['X_train'].shape
    print 'y_train shape = ', data_dict['y_train'].shape
    return data_dict['X_train'], data_dict['y_train']


def inputs(batch_size, num_epochs):
    """
    Get inputs
    """
    X, y = load_data()
    comp, label = tf.train.slice_input_producer([X, y], num_epochs=num_epochs, shuffle=True)
    comp_batch, label_batch = tf.train.batch([comp, label], batch_size)
    return comp_batch, label_batch


def get_placeholder():
    """
    Create a placeholder
    """
    lr_pl = tf.placeholder(tf.float32, name='learning_rate')
    keep_prob_pl = tf.placeholder(tf.float32, name='keep_prob')
    return lr_pl, keep_prob_pl


def run_train():
    """
    Train model
    """
    with tf.Graph().as_default():
        X, y = inputs(FLAGS.batch_size, FLAGS.num_epochs)
        lr_pl, keep_prob_pl = get_placeholder()
        m = model.RNN_Classification(FLAGS.basic_cell, FLAGS.rnn_size, FLAGS.num_layers, FLAGS.num_labels)
        logits = m.inference(X, keep_prob_pl)
        loss = m.loss(logits, y)
        train_op = m.train(loss, lr_pl)
        eval = m.eval(logits, y)
        init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess = tf.Session()
        sess.run(init)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        try:
            step = 0
            while not coord.should_stop():
                start_time = time.time()
                _, cost, correct = sess.run([train_op, loss, eval],
                                            feed_dict={lr_pl: FLAGS.init_learning_rate, keep_prob_pl: FLAGS.keep_prob})
                duration = time.time() - start_time

                if step % 100 == 0:
                    print 'Step: %d    train loss: %.3f(%.3f sec)   precision: %.3f' % (step, cost, duration, correct)
                step += 1
        except tf.errors.OutOfRangeError:
            print 'Done'
        finally:
            coord.request_stop()
        coord.join(threads)
        sess.close()


if __name__ == '__main__':
    run_train()
