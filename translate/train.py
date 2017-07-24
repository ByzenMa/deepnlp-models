# encoding=utf-8
import tensorflow as tf
import data_helper as helper
import model
import numpy as np

# 模型超参
tf.flags.DEFINE_integer("rnn_size", 128, "rnn size")
tf.flags.DEFINE_integer("num_layers", 1, "num layers")
tf.flags.DEFINE_integer("encoding_embedding_size", 100, "encoding embedding size")
tf.flags.DEFINE_integer("decoding_embedding_size", 100, "decoding embedding size")
tf.flags.DEFINE_float("init_learning_rate", 0.001, "initial learning rate")
tf.flags.DEFINE_float("gpu_fraction", 0.99, "gpu fraction")
tf.flags.DEFINE_float("dropout", 0.5, "dropout")

# 训练参数
tf.flags.DEFINE_integer("batch_size", 64, "batch size")
tf.flags.DEFINE_integer("num_epochs", 10, "number epochs")
tf.flags.DEFINE_integer("decay_step", 10, "number epochs")
tf.flags.DEFINE_integer("display_step", 200, "display step")
tf.flags.DEFINE_float("decay_rate", 0.99, "gpu fraction")
FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()


def pad_sentence_batch(sentence_batch, pad_int):
    """Pad sentences with <PAD> so that each sentence of a batch has the same length"""
    max_sentence = max([len(sentence) for sentence in sentence_batch])
    return [sentence + [pad_int] * (max_sentence - len(sentence)) for sentence in sentence_batch]


def get_batches(sources, targets, batch_size, source_pad_int, target_pad_int):
    """Batch targets, sources, and the lengths of their sentences together"""
    for batch_i in range(0, len(sources) // batch_size):
        start_i = batch_i * batch_size

        # Slice the right amount for the batch
        sources_batch = sources[start_i:start_i + batch_size]
        targets_batch = targets[start_i:start_i + batch_size]

        # Pad
        pad_sources_batch = np.array(pad_sentence_batch(sources_batch, source_pad_int))
        pad_targets_batch = np.array(pad_sentence_batch(targets_batch, target_pad_int))

        # Need the lengths for the _lengths parameters
        pad_targets_lengths = []
        for target in pad_targets_batch:
            pad_targets_lengths.append(len(target))

        pad_source_lengths = []
        for source in pad_sources_batch:
            pad_source_lengths.append(len(source))

        yield pad_sources_batch, pad_targets_batch, pad_source_lengths, pad_targets_lengths


def model_inputs():
    input_pl = tf.placeholder(tf.int32, shape=[None, None], name='input')
    targets_pl = tf.placeholder(tf.int32, shape=[None, None])
    lr_pl = tf.placeholder(tf.float32)
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    target_sequence_length = tf.placeholder(tf.int32, shape=(None,), name='target_sequence_length')
    max_target_len = tf.reduce_max(target_sequence_length, name='max_target_len')
    source_sequence_length = tf.placeholder(tf.int32, shape=(None,), name='source_sequence_length')
    return input_pl, targets_pl, lr_pl, keep_prob, target_sequence_length, max_target_len, source_sequence_length


def train():
    save_path = 'checkpoints/dev'
    (source_int_text, target_int_text), (source_vocab_to_int, target_vocab_to_int), _ = helper.load_preprocess()
    max_target_sentence_length = max([len(sentence) for sentence in source_int_text])
    train_graph = tf.Graph()
    with train_graph.as_default():
        input_data, targets, lr, keep_prob, target_sequence_length, \
        max_target_sequence_length, source_sequence_length = model_inputs()
        m = model.S2S(tf.reverse(input_data, [-1]),  # 采用reverse来训练S2S
                      targets,
                      FLAGS.rnn_size,
                      FLAGS.num_layers,
                      keep_prob,
                      FLAGS.batch_size,
                      source_sequence_length,
                      target_sequence_length,
                      max_target_sequence_length,
                      len(source_vocab_to_int),
                      len(target_vocab_to_int),
                      FLAGS.encoding_embedding_size,
                      FLAGS.decoding_embedding_size,
                      target_vocab_to_int)
        # 获取测试过程的解码和推断过程的解码
        training_decoder_output, inference_decoder_output = m.seq2seq_model()
        # 获得损失和
        cost, inference_logits = m.loss(training_decoder_output, inference_decoder_output)
        train_op = m.train(cost, lr)
        accuary = m.accuary(inference_logits)
        # Split data to training and validation sets

    batch_size = FLAGS.batch_size
    train_source = source_int_text[batch_size:]
    train_target = target_int_text[batch_size:]
    valid_source = source_int_text[:batch_size]
    valid_target = target_int_text[:batch_size]
    (valid_sources_batch, valid_targets_batch, valid_sources_lengths, valid_targets_lengths) = next(
        get_batches(valid_source,
                    valid_target,
                    batch_size,
                    source_vocab_to_int['<PAD>'],
                    target_vocab_to_int['<PAD>']))
    with tf.Session(graph=train_graph) as sess:
        sess.run(tf.global_variables_initializer())

        for epoch_i in range(FLAGS.epochs):
            for batch_i, (source_batch, target_batch, sources_lengths, targets_lengths) in enumerate(
                    get_batches(train_source, train_target, batch_size,
                                source_vocab_to_int['<PAD>'],
                                target_vocab_to_int['<PAD>'])):

                _, loss = sess.run(
                    [train_op, cost],
                    {input_data: source_batch,
                     targets: target_batch,
                     lr: FLAGS.init_learning_rate,
                     target_sequence_length: targets_lengths,
                     source_sequence_length: sources_lengths,
                     keep_prob: FLAGS.dropout})

                if batch_i % FLAGS.display_step == 0 and batch_i > 0:
                    batch_train_logits = sess.run(
                        inference_logits,
                        {input_data: source_batch,
                         source_sequence_length: sources_lengths,
                         target_sequence_length: targets_lengths,
                         keep_prob: 1.0})

                    batch_valid_logits = sess.run(
                        inference_logits,
                        {input_data: valid_sources_batch,
                         source_sequence_length: valid_sources_lengths,
                         target_sequence_length: valid_targets_lengths,
                         keep_prob: 1.0})

                    train_acc = m.accuary(target_batch, batch_train_logits)

                valid_acc = m.accuracy(valid_targets_batch, batch_valid_logits)

                print(
                    'Epoch {:>3} Batch {:>4}/{} - Train Accuracy: {:>6.4f}, Validation Accuracy: {:>6.4f}, Loss: {:>6.4f}'
                        .format(epoch_i, batch_i, len(source_int_text) // batch_size, train_acc, valid_acc, loss))

    # Save Model
    saver = tf.train.Saver()
    saver.save(sess, save_path)
    print('Model Trained and Saved')


if __name__ == '__main__':
    train()
