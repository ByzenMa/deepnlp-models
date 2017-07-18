# encoding=utf-8
import tensorflow as tf
import model
import data_helper as helper

# 模型超参
# tf.flags.DEFINE_string("rnn_cell", "rnn", "rnn cell")
tf.flags.DEFINE_integer("rnn_size", 100, "rnn size")
tf.flags.DEFINE_integer("embed_dim", 50, "embeding size")
tf.flags.DEFINE_float("init_learning_rate", 0.001, "initial learning rate")
tf.flags.DEFINE_float("dropout", 0.3, "dropout")
tf.flags.DEFINE_float("gpu_fraction", 0.99, "gpu fraction")
# 训练参数
tf.flags.DEFINE_integer("batch_size", 32, "batch size")
tf.flags.DEFINE_integer("num_epochs", 40, "number epochs")
tf.flags.DEFINE_integer("show_every_n_batches", 100, "show every n batches")
FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()


def print_param():
    '''
    打印超参
    :return:
    '''
    print("\nParameters")
    for attr, value in sorted(FLAGS.__flags.items()):
        print("{}={}".format(attr.upper(), value))
    print("")


def get_placeholder(vocab_size, story_maxlen, query_maxlen):
    '''
    创建placeholder
    :param story_maxlen:
    :param query_maxlen:
    :return:
    '''
    story_pl = tf.placeholder(tf.int32, shape=[None, story_maxlen], name='story_placeholder')
    question_pl = tf.placeholder(tf.int32, shape=[None, query_maxlen], name='question_placeholder')
    answer_pl = tf.placeholder(tf.int32, shape=[None, vocab_size], name='answer_placeholder')
    dropout_pl = tf.placeholder(tf.float32)
    return story_pl, question_pl, answer_pl, dropout_pl


def train_and_test(challenge, rnn_cell):
    '''
    训练模型
    :return:
    '''
    train, test = helper.extract_file(challenge)
    vocab, word_idx, story_maxlen, query_maxlen = helper.get_vocab(train, test)
    vocab_size = len(vocab) + 1  # Reserve 0 for masking via pad_sequences
    x, xq, y = helper.vectorize_stories(train, word_idx, story_maxlen, query_maxlen)
    tx, txq, ty = helper.vectorize_stories(test, word_idx, story_maxlen, query_maxlen)
    with tf.Graph().as_default() as graph:
        story_pl, question_pl, answer_pl, dropout_pl = get_placeholder(vocab_size, story_maxlen, query_maxlen)
        rnn = model.RNN(rnn_cell, FLAGS.embed_dim, FLAGS.rnn_size, vocab_size)
        logits = rnn.inference(story_pl, question_pl, dropout_pl)
        loss = rnn.loss(logits, answer_pl)
        train_op = rnn.train(loss, FLAGS.init_learning_rate)
        correct = rnn.eval(logits, answer_pl)
        init = tf.global_variables_initializer()
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=FLAGS.gpu_fraction)
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options), graph=graph) as sess:
            # 初始化所有变量
            sess.run(init)
            max_test_acc = 0
            for i in range(FLAGS.num_epochs):
                batch_id = 1
                train_gen = helper.generate_data(FLAGS.batch_size, x, xq, y)
                for x_batch, xq_batch, y_batch in train_gen:
                    feed_dict = {story_pl: x_batch, question_pl: xq_batch, answer_pl: y_batch,
                                 dropout_pl: FLAGS.dropout}
                    cost, _ = sess.run([loss, train_op], feed_dict=feed_dict)
                    # 每固定批次
                    # if batch_id % FLAGS.show_every_n_batches == 0:
                    #     print ('Epoch {:>3} Batch {:>4}   train_loss = {:.3f}'.format(i, batch_id, cost))
                    batch_id += 1
                # 每个epoch后，测试一次
                test_gen = helper.generate_data(FLAGS.batch_size, tx, txq, ty)
                total_correct = 0
                total = len(tx)
                for tx_batch, txq_batch, ty_batch in test_gen:
                    feed_dict = {story_pl: tx_batch, question_pl: txq_batch, answer_pl: ty_batch,
                                 dropout_pl: 1.0}
                    cor = sess.run(correct, feed_dict=feed_dict)
                    total_correct += int(cor)
                acc = total_correct * 1.0 / total
                # 获得max test accuary
                if acc > max_test_acc:
                    max_test_acc = acc
                print (
                    'Epoch{:>3}   train_loss = {:.3f}   accuary = {:.3f}   max_text_acc = {:.3f}'.format(i, cost, acc,
                                                                                                         max_test_acc))
            return max_test_acc


def train_process():
    '''
    训练过程
    :return:
    '''
    print_param()
    rnn_cells = ['rnn', 'lstm', 'gru']
    prefixs = ['en', 'en-10k']
    tasks = [
        'qa1_single-supporting-fact'
    ]
    suffix = '_{}.txt'
    with open('result.file', 'w') as result:
        for rnn_cell in rnn_cells:
            for prefix in prefixs:
                for task in tasks:
                    challenge = 'tasks_1-20_v1-2/' + prefix + '/' + task + suffix
                    max_test_acc = train_and_test(challenge, rnn_cell)
                    result.write(rnn_cell + '\t' + prefix + '\t' + task + '\t' + str(max_test_acc) + '\n')
                    result.flush()
        result.close


if __name__ == "__main__":
    train_process()
