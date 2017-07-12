# encoding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from distutils.version import LooseVersion
import warnings
from model import tv_script_generation as gen
import data_helper as helper

# 模型超参
tf.flags.DEFINE_integer("rnn_size", 200, "rnn size (default: 128)")
tf.flags.DEFINE_string("basic_cell", "gru", "basic model (default: gru)")
tf.flags.DEFINE_integer("rnn_layer", 2, "num layers (default: 3)")
tf.flags.DEFINE_integer("embed_dim", 300, "num layers (default: 3)")
tf.flags.DEFINE_float("grad_clip", 1, "num layers (default: 3)")
tf.flags.DEFINE_float("learning_rate", 0.002, "learning rate (default: 0.001)")
# 训练参数
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 300, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_string("save_dir", "./save", "Model saving directory")
tf.flags.DEFINE_integer("show_every_n_batches", 89, "Show every n batches")
FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()


# 打印参数
def print_param():
    print("\nParameters:")
    for attr, value in sorted(FLAGS.__flags.items()):
        print("{}={}".format(attr.upper(), value))
    print("")


# 检测系统环境
def check_env():
    assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer'
    print('TensorFlow Version: {}'.format(tf.__version__))

    if not tf.test.gpu_device_name():
        warnings.warn('No GPU found. Please use a GPU to train your neural network.')
    else:
        print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


# 生成本地
def get_inputs():
    input = tf.placeholder(tf.int32, shape=[None, None], name='input')
    targets = tf.placeholder(tf.int32, shape=[None, None], name='targets')
    lr = tf.placeholder(tf.float32, name='learning_rate')
    return input, targets, lr


# 模型训练
def train():
    batches = helper.get_batches(FLAGS.batch_size)
    train_graph = tf.Graph()
    with train_graph.as_default():
        input, targets, lr = get_inputs()
        vocab_size, seq_length = helper.get_text_info()
        model = gen(FLAGS.basic_cell, FLAGS.rnn_size, FLAGS.rnn_layer, FLAGS.embed_dim, seq_length, vocab_size,
                    FLAGS.batch_size, FLAGS.grad_clip)
        logits, initial_state, final_state, _ = model.inference(input)
        loss = model.loss(logits, targets, tf.shape(input))
        train_op = model.train(loss, lr)
        with tf.Session(graph=train_graph) as sess:
            sess.run(tf.global_variables_initializer())
            for epoch_i in range(FLAGS.num_epochs):
                state = sess.run(initial_state, {input: batches[0][0]})
                # 开始迭代循环
                for batch_i, (x, y) in enumerate(batches):
                    feed = {
                        input: x,
                        targets: y,
                        initial_state: state,
                        lr: FLAGS.learning_rate}
                    train_loss, state, _ = sess.run([loss, final_state, train_op], feed)

                    # 模型日志打印
                    if (epoch_i * len(batches) + batch_i) % FLAGS.show_every_n_batches == 0:
                        print('Epoch {:>3} Batch {:>4}/{}   train_loss = {:.3f}'.format(
                            epoch_i,
                            batch_i,
                            len(batches),
                            train_loss))
            # 模型保存
            saver = tf.train.Saver()
            saver.save(sess, FLAGS.save_dir)
            print('Model Trained and Saved')


if __name__ == "__main__":
    print_param()
    check_env()
    train()
