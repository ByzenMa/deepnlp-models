# encoding=utf-8
import tensorflow as tf
import numpy as np
import train
import data_helper as helper

FLAGS = train.FLAGS
load_path = FLAGS.save_path


def sentence_to_seq(sentence, vocab_to_int):
    '''
    将测试句子转换成输入id
    '''
    sent = sentence.lower()
    sent_to_id = [vocab_to_int.get(word, vocab_to_int['<UNK>']) for word in sent.split(' ')]
    return sent_to_id


def test(translate_sentence):
    _, (source_vocab_to_int, target_vocab_to_int), (source_int_to_vocab, target_int_to_vocab) = helper.load_preprocess()
    translate_sentence = sentence_to_seq(translate_sentence, source_vocab_to_int)
    loaded_graph = tf.Graph()
    with tf.Session(graph=loaded_graph) as sess:
        # Load saved model
        loader = tf.train.import_meta_graph(load_path + '.meta')
        loader.restore(sess, load_path)

        input_data = loaded_graph.get_tensor_by_name('input:0')
        logits = loaded_graph.get_tensor_by_name('predictions:0')
        target_sequence_length = loaded_graph.get_tensor_by_name('target_sequence_length:0')
        source_sequence_length = loaded_graph.get_tensor_by_name('source_sequence_length:0')
        keep_prob = loaded_graph.get_tensor_by_name('keep_prob:0')

        translate_logits = sess.run(logits, {input_data: [translate_sentence] * FLAGS.batch_size,
                                             target_sequence_length: [len(translate_sentence) * 2] * FLAGS.batch_size,
                                             source_sequence_length: [len(translate_sentence)] * FLAGS.batch_size,
                                             keep_prob: 1.0})[0]

    print('Input')
    print('  Word Ids:      {}'.format([i for i in translate_sentence]))
    print('  English Words: {}'.format([source_int_to_vocab[i] for i in translate_sentence]))

    print('\nPrediction')
    print('  Word Ids:      {}'.format([i for i in translate_logits]))
    print('  French Words: {}'.format(" ".join([target_int_to_vocab[i] for i in translate_logits])))


if __name__ == '__main__':
    translate_sentence = 'he saw a old yellow truck .'
    test(translate_sentence)
