# encoding=utf-8
import numpy as np
import os
import pickle
from pathlib2 import Path

data_dir = './data/simpsons/moes_tavern_lines.txt'
save_dir = './save'


# 加载数据
def load_data(path):
    input_file = os.path.join(path)
    with open(input_file, "r") as f:
        data = f.read()
    return data


def token_lookup():
    tokenize_dict = {'.': "||Period||", ',': "||Comma||", '"': "||Quotation_mark||", ';': "||Semicolon||",
                     '!': "||Exclamation_mark||", '?': "||Question_mark||", '(': "||Left_parentheses||",
                     ')': "||Right_parentheses||", '--': "||Dash||", '\n': "||Return||"}
    return tokenize_dict


def create_lookup_tables(text):
    vocab_to_int = {}
    for sentence in text:
        for word in sentence.split(' '):
            if word not in vocab_to_int:
                vocab_to_int[word] = len(vocab_to_int)
    int_to_vocab = dict((v, k) for k, v in vocab_to_int.iteritems())
    return vocab_to_int, int_to_vocab


# 将数据写入pickle文件，方便以后访问
def preprocess_and_save_data():
    text = load_data(data_dir)
    # Ignore notice, since we don't use it for analysing the data
    text = text[81:]
    token_dict = token_lookup()
    for key, token in token_dict.items():
        text = text.replace(key, ' {} '.format(token))
    text = text.lower()
    text = text.split()
    vocab_to_int, int_to_vocab = create_lookup_tables(text)
    int_text = [vocab_to_int[word] for word in text]
    pickle.dump((int_text, vocab_to_int, int_to_vocab, token_dict), open('data/preprocess.p', 'wb'))


# 获取批次
def get_batches(batch_size):
    int_text, _, _, _ = load_preprocess()
    text_len = len(int_text)
    _, seq_length = get_text_info()
    batchs = text_len / (batch_size * seq_length)
    if text_len < batchs * batch_size * seq_length + 1:
        batchs -= 1
    if batchs < 1:
        return []
    input_len = batchs * batch_size * seq_length
    input_text = np.array(int_text[:input_len]).reshape([batch_size, -1])
    target_text = np.array(int_text[1:input_len + 1]).reshape([batch_size, -1])
    batchs_data = np.ndarray((batchs, 2, batch_size, seq_length), dtype=np.int32)
    batchs_data[:, 0, :, :] = np.split(input_text, batchs, axis=-1)
    batchs_data[:, 1, :, :] = np.split(target_text, batchs, axis=-1)
    return batchs_data


# 加载预处理数据
def load_preprocess():
    my_file = Path("data/preprocess.p")
    if not my_file.exists():
        print("Pickle file is not exist, preprocess and save data.")
        preprocess_and_save_data()
    return pickle.load(open('data/preprocess.p', mode='rb'))


# 加载参数
def get_params():
    _, seq_length = get_text_info()
    return seq_length, save_dir


# 获得文本信息
def get_text_info():
    text = load_data(data_dir)
    text = text[81:]
    scenes = text.split('\n\n')
    sentences = [sentence for scene in scenes for sentence in scene.split('\n')]
    word_count_sentence = [len(sentence.split()) for sentence in sentences]
    vocab_size = len({word: None for word in text.split()})
    seq_length = int(np.average(word_count_sentence)) + 1
    return vocab_size, seq_length
