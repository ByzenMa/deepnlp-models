# encoding=utf-8
import tensorflow as tf
import model
import pickle as pckl
import data_helper as helper

# 模型参数
tf.flags.DEFINE_integer('rnn_size', 100, 'rnn size')
tf.flags.DEFINE_integer('num_layers', 2, 'rnn layers number')
tf.flags.DEFINE_integer('batch_size', 64, 'batch size')
tf.flags.DEFINE_integer('num_epochs', 1000, 'epochs number')
tf.flags.DEFINE_string('basic_cell', 'gru', 'basic rnn cell')
tf.flags.DEFINE_float('learning_rate_decay', 0.99, 'learning rate decay rate')
tf.flags.DEFINE_float('init_learning_rate', 0.001, 'learning rate decay rate')
tf.flags.DEFINE_integer('decay_step', 20, 'decay step')
tf.flags.DEFINE_float('gpu_fraction', 0.9, 'gpu fraction')
tf.flags.DEFINE_float('dropout', 0.5, 'dropout')
FLAGS = tf.flags.FLAGS
seq_len = helper.MONTHS_NUM
data_path = helper.DATA_PAHT


def get_data():
    '''
    加载数据
    '''
    if not tf.gfile.Exists(data_path):
        print 'pickle文件不存在，处理数据并存储在%s' % data_path
        features, labels = helper.load_data()
        helper.pickle_data(features, labels)
    with open(data_path, 'rb') as file:
        data_dict = pckl.load(file)
        print 'X_train shape:', data_dict.get('X_train').shape
        print 'y_train shape:', data_dict.get('y_train').shape
        print 'X_test shape:', data_dict.get('X_test').shape
        print 'y_test shape:', data_dict.get('y_test').shape
    return data_dict.get('X_train'), data_dict.get('y_train'), data_dict.get('X_test'), data_dict.get('y_test')


def get_placeholder():
    '''
    建立placeholder
    '''
    x_pl = tf.placeholder(tf.float32, shape=(None, helper.MONTHS_NUM, helper.FEATURE_DIM), name='input_placeholer')
    y_pl = tf.placeholder(tf.float32, shape=(None, 1), name='label_placeholder')
    lr_pl = tf.placeholder(tf.float32, name='learning_rate_placeholder')
    dropout_pl = tf.placeholder(tf.float32, name='dropout')
    return x_pl, y_pl, lr_pl, dropout_pl


def train_and_test():
    '''
    模型的训练和测试
    '''
    X_train, y_train, X_test, y_test = get_data()
    with tf.Graph().as_default() as graph:
        x_pl, y_pl, lr_pl, dropout_pl = get_placeholder()
        m = model.RNN_Regression(FLAGS.batch_size, FLAGS.rnn_size, FLAGS.num_layers, FLAGS.basic_cell, lr_pl, seq_len,
                                 dropout_pl)
        logits = m.inference(x_pl)
        loss = m.loss(logits, y_pl)
        train_op = m.train(loss)
        init = tf.global_variables_initializer()
        gpu_op = tf.GPUOptions(per_process_gpu_memory_fraction=FLAGS.gpu_fraction)
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_op), graph=graph) as sess:
            # 初始化所有变量
            sess.run(init)
            current_learning_rate = FLAGS.init_learning_rate
            min_test_loss = float('inf')
            decay_step = 0
            for epoch in range(FLAGS.num_epochs):
                # 训练过程
                train_gen = helper.generate_batch(FLAGS.batch_size, X_train, y_train)
                if decay_step > FLAGS.decay_step:
                    current_learning_rate = current_learning_rate * FLAGS.learning_rate_decay

                train_total_loss = list()
                for X_train_batch, y_train_batch in train_gen:
                    feed_dict = {x_pl: X_train_batch, y_pl: y_train_batch, lr_pl: current_learning_rate,
                                 dropout_pl: FLAGS.dropout}
                    _train_loss, _ = sess.run([loss, train_op], feed_dict=feed_dict)
                    train_total_loss.append(_train_loss)

                # 测试过程
                test_total_loss = list()
                test_gen = helper.generate_batch(FLAGS.batch_size, X_test, y_test)
                for X_test_batch, y_test_batch in test_gen:
                    feed_dict = {x_pl: X_test_batch, y_pl: y_test_batch, dropout_pl: 1}
                    _loss = sess.run([loss], feed_dict=feed_dict)
                    test_total_loss.append(_loss)
                train_avg_loss = tf.reduce_mean(tf.convert_to_tensor(train_total_loss)).eval()
                test_avg_loss = tf.reduce_mean(tf.convert_to_tensor(test_total_loss)).eval()

                # 累计衰减步数
                if test_avg_loss < min_test_loss:
                    decay_step = 0
                    min_test_loss = test_avg_loss
                else:
                    decay_step += 1

                print 'Eopch{:>3}   lr={:.6f}   train_loss={:.3f}   test_loss={:.3f}    min_test_loss={:.3f}'.format(
                    epoch,
                    current_learning_rate,
                    train_avg_loss,
                    test_avg_loss,
                    min_test_loss)


if __name__ == '__main__':
    train_and_test()
