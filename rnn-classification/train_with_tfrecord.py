# encoding = utf-8
import tensorflow as tf
import model
import data_helper as helper
import time

tf.flags.DEFINE_integer('rnn_size', 100, 'rnn size')
tf.flags.DEFINE_integer('num_layers', 1, 'num layers')
tf.flags.DEFINE_float('keep_prob', 0.8, 'keep prob')
tf.flags.DEFINE_integer('num_labels', 2, 'num labels')
tf.flags.DEFINE_string('basic_cell', 'gru', 'basic cell')
tf.flags.DEFINE_float('lr_decay_factor', 0.99, 'decay factor')
tf.flags.DEFINE_float('decay_steps', 20, 'decay steps')
tf.flags.DEFINE_float('learning_rate', 0.001, 'learning rate')
tf.flags.DEFINE_integer('batch_size', 64, 'batch size')
tf.flags.DEFINE_integer('num_epochs', 200, 'num epochs')

FLAGS = tf.flags.FLAGS
DATA_PATH = 'data/train.tfrecord'


def get_pl():
    """
    Get learning rate placeholder
    """
    lr_pl = tf.placeholder(tf.float32, name='learning_rate')
    return lr_pl


def read_and_decode(filename):
    """
    Read file and decode file
    """
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename)
    features = tf.parse_single_example(serialized_example, features={
        'X': tf.FixedLenFeature([], tf.string),
        'y': tf.FixedLenFeature([], tf.int64)

    })
    X = tf.decode_raw(features['X'], tf.float32)
    X.set_shape([helper.MONTH_NUMS * helper.FEATURE_DIM])
    X_reshaped = tf.reshape(X, [helper.MONTH_NUMS, helper.FEATURE_DIM])
    y = tf.cast(features['y'], tf.int32)
    return X_reshaped, y


def get_inputs():
    """
    Get inputs
    """
    filename = tf.train.string_input_producer([DATA_PATH])
    X, y = read_and_decode(filename)
    return tf.train.shuffle_batch([X, y], FLAGS.batch_size, capacity=1 + FLAGS.batch_size, min_after_dequeue=1,
                                  num_threads=1)


def train():
    """
    Run training
    """
    with tf.Graph().as_default():
        lr_pl = get_pl()
        X, y = get_inputs()
        m = model.RNN_Classification(FLAGS.basic_cell, FLAGS.rnn_size, FLAGS.num_layers, FLAGS.keep_prob,
                                     FLAGS.num_labels)
        logits = m.inference(X)
        loss = m.loss(logits, y)
        train_op = m.train(loss, lr_pl)
        init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        coord = tf.train.Coordinator()
        # Begin sess
        with tf.Session() as sess:
            sess.run(init)
            threads = tf.train.start_queue_runners(sess, coord)
            try:
                step = 0
                while coord.should_stop():
                    start_time = time.time()
                    cost, _ = sess.run([loss, train_op], feed_dict={lr_pl: FLAGS.learning_rate})
                    duration = time.time() - start_time

                    if step % 100 == 0:
                        print 'Step %d: loss = %.3f (%.3f sec)' % (step, cost, duration)
                    step += 1
            except tf.errors.OutOfRangeError:
                print 'Done training for %d epochs, %d steps' % {FLAGS.num_epochs, step}
            finally:
                coord.request_stop()
            coord.join(threads)
            sess.close()


if __name__ == '__main__':
    train()
