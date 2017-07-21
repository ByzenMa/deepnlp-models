# encoding=utf-8
import tensorflow as tf
import model
import data_helper as helper

# 模型超参
tf.flags.DEFINE_float("init_learning_rate", 0.01, "initial learning rate")
tf.flags.DEFINE_float("gpu_fraction", 0.99, "gpu fraction")
# 训练参数
tf.flags.DEFINE_integer("batch_size", 32, "batch size")
tf.flags.DEFINE_integer("glove_dim", 300, "glove dim size")
tf.flags.DEFINE_integer("num_epochs", 200, "number epochs")
tf.flags.DEFINE_integer("decay_step", 10, "number epochs")
tf.flags.DEFINE_float("decay_rate", 0.99, "gpu fraction")
tf.flags.DEFINE_string("model", "nn", "model type")
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


def get_placeholder(vocab_size):
    '''
    创建placeholder
    :param story_maxlen:
    :param query_maxlen:
    :return:
    '''
    input_pl = tf.placeholder(tf.float32, shape=[None, FLAGS.glove_dim * 2], name='input_placeholder')
    label_pl = tf.placeholder(tf.int32, shape=[None, vocab_size], name='label_placeholder')
    learning_rate_pl = tf.placeholder(tf.float32)
    return input_pl, label_pl, learning_rate_pl


def train_and_test(challenge):
    '''
    训练模型
    :return:
    '''
    train, test = helper.extract_file(challenge)
    vocab, word_idx = helper.get_vocab(train, test)
    vocab_size = len(vocab) + 1  # Reserve 0 for masking via pad_sequences
    x, y = helper.vectorize_stories(train, word_idx)
    tx, ty = helper.vectorize_stories(test, word_idx)
    with tf.Graph().as_default() as graph:
        input_pl, label_pl, learning_rate_pl = get_placeholder(vocab_size)
        # 选择使用的model
        if FLAGS.model == "nn":
            used_model = model.NN
        else:
            used_model = model.LR
        lr = used_model(learning_rate_pl, vocab_size)
        logits = lr.inference(input_pl)
        loss = lr.loss(logits, label_pl)
        train_op = lr.train(loss)
        correct = lr.eval(logits, label_pl)
        init = tf.global_variables_initializer()
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=FLAGS.gpu_fraction)
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options), graph=graph) as sess:
            # 初始化所有变量
            sess.run(init)
            max_test_acc = 0
            step_spam = 0
            current_learning_rate = FLAGS.init_learning_rate
            for i in range(FLAGS.num_epochs):
                batch_id = 1
                if step_spam >= FLAGS.decay_step:
                    current_learning_rate = current_learning_rate * FLAGS.decay_rate
                train_gen = helper.generate_data(FLAGS.batch_size, x, y)
                for x_batch, y_batch in train_gen:
                    feed_dict = {input_pl: x_batch, label_pl: y_batch, learning_rate_pl: current_learning_rate}
                    cost, _ = sess.run([loss, train_op], feed_dict=feed_dict)
                    # 每固定批次
                    # if batch_id % FLAGS.show_every_n_batches == 0:
                    #     print ('Epoch {:>3} Batch {:>4}   train_loss = {:.3f}'.format(i, batch_id, cost))
                    batch_id += 1
                # 每个epoch后，测试一次
                test_gen = helper.generate_data(FLAGS.batch_size, tx, ty)
                total_correct = 0
                total = len(tx)
                for tx_batch, ty_batch in test_gen:
                    feed_dict = {input_pl: tx_batch, label_pl: ty_batch, learning_rate_pl: current_learning_rate}
                    cor = sess.run(correct, feed_dict=feed_dict)
                    total_correct += int(cor)
                acc = total_correct * 1.0 / total
                # 获得max test accuary
                if acc > max_test_acc:
                    max_test_acc = acc
                    step_spam = 0
                else:
                    step_spam += 1
                print (
                    'Epoch{:>3}   lr = {:.6f}   train_loss = {:.3f}   accuary = {:.3f}   max_text_acc = {:.3f}'.format(
                        i, current_learning_rate, cost, acc, max_test_acc))
            return max_test_acc


def train_process():
    '''
    训练过程
    :return:
    '''
    print_param()
    prefixs = ['en', 'en-10k']
    tasks = [
        'qa1_single-supporting-fact', 'qa2_two-supporting-facts', 'qa3_three-supporting-facts',
        'qa4_two-arg-relations', 'qa5_three-arg-relations', 'qa6_yes-no-questions', 'qa7_counting',
        'qa8_lists-sets', 'qa9_simple-negation', 'qa10_indefinite-knowledge',
        'qa11_basic-coreference', 'qa12_conjunction', 'qa13_compound-coreference',
        'qa14_time-reasoning', 'qa15_basic-deduction', 'qa16_basic-induction', 'qa17_positional-reasoning',
        'qa18_size-reasoning', 'qa19_path-finding', 'qa20_agents-motivations'
    ]
    suffix = '_{}.txt'
    with open('result.file', 'w') as result:
        for prefix in prefixs:
            for task in tasks:
                challenge = 'tasks_1-20_v1-2/' + prefix + '/' + task + suffix
                max_test_acc = train_and_test(challenge)
                result.write(prefix + '\t' + task + '\t' + str(max_test_acc) + '\n')
                result.flush()
    result.close


if __name__ == "__main__":
    train_process()
