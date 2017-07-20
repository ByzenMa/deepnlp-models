# encoding=utf-8
import re
import tarfile
import spacy
import numpy as np

SEED = 2017
TAR_GZ = '../data/babi_tasks_1-20_v1-2.tar.gz'
nlp = spacy.load('en')


def tokenize(sent):
    '''
    对句子进行分词，并过滤掉停用词
    :param sent:要被解析的句子
    :return:
    '''
    split = [x.strip() for x in re.split('(\W+)?', sent) if x.strip() not in spacy.en.language_data.STOP_WORDS]
    return [word for word in split if len(word) > 0]


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


def filter_story(data):
    '''
    根据n-gram的要求，将不含有question的词语的sentence过滤掉
    :param data:
    :return:
    '''
    data_filter = []
    for story, query, answer in data:
        new_story = []
        for sent in story:
            if set(sent) & set(query):
                new_story.append(sent)
        data_filter.append((new_story, query, answer))
    return data_filter


def get_stories(f, only_supporting=False, max_length=None):
    '''Given a file name, read the file, retrieve the stories,
    and then convert the sentences into a single story.
    If max_length is supplied,
    any stories longer than max_length tokens will be discarded.
    '''
    data = parse_stories(f.readlines(), only_supporting=only_supporting)
    data = filter_story(data)
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
    '''
    获得词集合以及词到id的映射
    :param train:
    :param test:
    :return:
    '''
    vocab = set()
    for story, q, answer in train + test:
        vocab |= set(story + q + [answer])  # 对集合采用并操作('|')，与之对应，采用交操作('&')
    vocab = sorted(vocab)
    word_idx = dict((c, i + 1) for i, c in enumerate(vocab))  # 对id=0进行了保留
    return vocab, word_idx


def vectorize_stories(data, word_idx):
    '''
    采用gloVe将story词向量求和，question词向量求和，两者连接起来作为输入；预测值作为输出
    :param data:
    :param word_idx:
    :param story_maxlen:
    :param query_maxlen:
    :return:
    '''
    xs = []
    ys = []
    for story, query, answer in data:
        x = nlp(unicode(' '.join(story))).vector
        xq = nlp(unicode(' '.join(query))).vector
        join = np.concatenate((x, xq))
        # let's not forget that index 0 is reserved
        y = np.zeros(len(word_idx) + 1)
        y[word_idx[answer]] = 1
        xs.append(join)
        ys.append(y)
    return xs, ys


def generate_data(batch_size, input, label, shuffle=True):
    '''
    生成迭代器
    :param batch_size:
    :param input:
    :param label:
    :param shuffle:
    :return:
    '''
    assert len(input) == len(label), \
        "story和answer的长度不相等，请检查数据是否正确"
    length = len(input)
    input_arry = np.array(input)
    label_arry = np.array(label)
    if shuffle:
        np.random.seed(SEED)
        idx = np.random.permutation(length)
        input_arry = input_arry[idx]
        label_arry = label_arry[idx]
    total_batch = int(length / batch_size)
    input_arry_list = np.split(input_arry[:total_batch * batch_size], total_batch, 0)
    label_arry_list = np.split(label_arry[:total_batch * batch_size], total_batch, 0)
    for i in range(total_batch):
        yield input_arry_list[i], label_arry_list[i]


if __name__ == "__main__":
    s = "1 John travelled to the hallway."
    print tokenize(s)

    lines = ['1 John travelled to the hallway.', '2 Mary journeyed to the bathroom.',
             '3 Where is John? 	hallway	1', '4 Daniel went back to the bathroom.', '5 John moved to the bedroom.',
             '6 Where is Mary? 	bathroom	2']
    print parse_stories(lines)

    challenge = 'tasks_1-20_v1-2/en/qa1_single-supporting-fact_{}.txt'
    train, test = extract_file(challenge)
    _, word_idx = get_vocab(train, test)
    xs, ys = vectorize_stories(train, word_idx)
    print len(xs), len(ys)
