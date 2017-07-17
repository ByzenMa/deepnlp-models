# encoding=utf-8
import re
import tensorflow as tf
import tarfile
import numpy as np

TAR_GZ = './data/babi_tasks_1-20_v1-2.tar.gz'


def tokenize(sent):
    '''
    对句子进行分词
    :param sent:要被解析的句子
    :return:
    '''
    return [x.strip() for x in re.split('(\W+)?', sent) if x.strip()]


def parse_stories(lines, only_supporting=False):
    '''
    解析文本
    :param lines:
    :param only_supporting:
    :return:
    '''
    data = []
    story = []
    for line in lines:
        line = line.decode('utf-8').strip()
        nid, line = line.split(' ', 1)
        nid = int(nid)
        if nid == 1:
            story = []
        if '\t' in line:
            q, a, supporting = line.split('\t')
            q = tokenize(q)
            substory = None
            if only_supporting:
                # Only select the related substory
                supporting = map(int, supporting.split())
                substory = [story[i - 1] for i in supporting]
            else:
                # Provide all the substories
                substory = [x for x in story if x]
            data.append((substory, q, a))
            story.append('')
        else:
            sent = tokenize(line)
            story.append(sent)
    return data


def get_stories(f, only_supporting=False, max_length=None):
    '''Given a file name, read the file, retrieve the stories,
    and then convert the sentences into a single story.
    If max_length is supplied,
    any stories longer than max_length tokens will be discarded.
    '''
    data = parse_stories(f.readlines(), only_supporting=only_supporting)
    flatten = lambda data: reduce(lambda x, y: x + y, data)
    data = [(flatten(story), q, answer) for story, q, answer in data if
            not max_length or len(flatten(story)) < max_length]
    return data


def extract_file(challenge):
    tar = tarfile.open(TAR_GZ)
    train = get_stories(tar.extractfile(challenge.format('train')))
    test = get_stories(tar.extractfile(challenge.format('test')))
    return train, test


def get_vocab(train, test):
    vocab = set()
    for story, q, answer in train + test:
        vocab |= set(story + q + [answer])  # 对集合采用并操作('|')，与之对应，采用交操作('&')
    vocab = sorted(vocab)
    word_idx = dict((c, i + 1) for i, c in enumerate(vocab))
    story_maxlen = max(map(len, (x for x, _, _ in train + test)))
    query_maxlen = max(map(len, (x for _, x, _ in train + test)))
    return vocab, word_idx, story_maxlen, query_maxlen


def vectorize_stories(data, word_idx, story_maxlen, query_maxlen):
    xs = []
    xqs = []
    ys = []
    for story, query, answer in data:
        x = [word_idx[w] for w in story]
        xq = [word_idx[w] for w in query]
        # let's not forget that index 0 is reserved
        y = np.zeros(len(word_idx) + 1)
        y[word_idx[answer]] = 1
        xs.append(x)
        xqs.append(xq)
        ys.append(y)
    return pad_sequences(xs, maxlen=story_maxlen), pad_sequences(xqs, maxlen=query_maxlen), np.array(ys)


def pad_sequences(sequences, maxlen=None):
    '''
    调用tensorflow中的keras包
    :param sequences:
    :param maxlen:
    :return:
    '''
    return tf.contrib.keras.preprocessing.sequence.pad_sequences(sequences, maxlen)


if __name__ == "__main__":
    s = "1 John travelled to the hallway."
    print tokenize(s)

    lines = ['1 John travelled to the hallway.', '2 Mary journeyed to the bathroom.',
             '3 Where is John? 	hallway	1', '4 Daniel went back to the bathroom.', '5 John moved to the bedroom.',
             '6 Where is Mary? 	bathroom	2']
    print parse_stories(lines)

    challenge = 'tasks_1-20_v1-2/en/qa2_two-supporting-facts_{}.txt'
    train, test = extract_file(challenge)
    print train[0:2]
