# encoding=utf-8
import os
import pickle
import copy
import numpy as np

PREPROCESS_DATA = 'data/preprocess.p'

CODES = {'<PAD>': 0, '<EOS>': 1, '<UNK>': 2, '<GO>': 3}


def load_data(path):
    '''
    读取文件
    :param path:
    :return:
    '''
    input_file = os.path.join(path)
    with open(input_file, 'r') as f:
        return f.read()


def text_to_ids(source_text, target_text, source_vocab_to_int, target_vocab_to_int):
    '''
    将输入文本转换成id文本，其中解码的文本需要后缀添加'<EOS>'
    :param source_text:
    :param target_text:
    :param source_vocab_to_int:
    :param target_vocab_to_int:
    :return:
    '''

    def vocab_to_int(vocab_to_int_dict, text, target=False):
        if target:
            sents = [sent + ' <EOS>' for sent in text.split('\n')]
        else:
            sents = text.split('\n')
        vocab_to_int = []
        for sent in sents:
            ids = [vocab_to_int_dict[word] for word in sent.split(' ') if len(word.strip()) > 0]
            vocab_to_int.append(ids)
        return vocab_to_int

    source_id_text = vocab_to_int(source_vocab_to_int, source_text)
    target_id_text = vocab_to_int(target_vocab_to_int, target_text, target=True)
    return source_id_text, target_id_text


def preprocess_and_save_data(source_path, target_path):
    source_text = load_data(source_path)
    target_text = load_data(target_path)
    source_text = source_text.lower()
    target_text = target_text.lower()
    source_vocab_to_int, source_int_to_vocab = create_lookup_tables(source_text)
    target_vocab_to_int, target_int_to_vocab = create_lookup_tables(target_text)
    source_text, target_text = text_to_ids(source_text, target_text, source_vocab_to_int, target_vocab_to_int)
    with open(PREPROCESS_DATA, 'wb') as out_file:
        pickle.dump((
            (source_text, target_text),
            (source_vocab_to_int, target_vocab_to_int),
            (source_int_to_vocab, target_int_to_vocab)), out_file)


def load_preprocess():
    with open(PREPROCESS_DATA, mode='rb') as in_file:
        return pickle.load(in_file)


def create_lookup_tables(text):
    vocab = set(text.split())
    vocab_to_int = copy.copy(CODES)
    for v_i, v in enumerate(vocab, len(CODES)):
        vocab_to_int[v] = v_i
    int_to_vocab = {v_i: v for v, v_i in vocab_to_int.items()}
    print "vocab size: %d" % len(vocab_to_int)
    return vocab_to_int, int_to_vocab


def save_params(params):
    with open('params.p', 'wb') as out_file:
        pickle.dump(params, out_file)


def load_params():
    with open('params.p', mode='rb') as in_file:
        return pickle.load(in_file)


def batch_data(source, target, batch_size):
    for batch_i in range(0, len(source) // batch_size):
        start_i = batch_i * batch_size
        source_batch = source[start_i:start_i + batch_size]
        target_batch = target[start_i:start_i + batch_size]
        yield np.array(pad_sentence_batch(source_batch)), np.array(pad_sentence_batch(target_batch))


def pad_sentence_batch(sentence_batch):
    max_sentence = max([len(sentence) for sentence in sentence_batch])
    return [sentence + [CODES['<PAD>']] * (max_sentence - len(sentence))
            for sentence in sentence_batch]
