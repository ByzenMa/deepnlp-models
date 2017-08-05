# encoding=utf-8
import tarfile
import re
import spacy
import numpy as np

DATA_GZ = "../data/babi_tasks_1-20_v1-2.tar.gz"
nlp = spacy.load('en')


def parse_stories(lines, only_supporting=False):
    """
    Parse file
    """
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
                supporting = map(int, supporting.split())
                substory = [story[i - 1] for i in supporting]
            else:
                substory = [x for x in story if x]
            data.append((substory, q, a))
            story.append('')
        else:
            sent = tokenize(line)
            story.append(sent)
    return data


def filter_story(data):
    """
    Filter story
    """
    data_filter = []
    for story, query, answer in data:
        new_story = []
        for sent in story:
            if set(sent) & set(query):
                new_story.append(sent)
        data_filter.append((new_story, query, answer))
    return data_filter


def get_stories(f, only_supporting=False, max_length=None):
    """
    Get stories
    """
    data = parse_stories(f.readlines(), only_supporting=only_supporting)
    data = filter_story(data)
    flatten = lambda data: reduce(lambda x, y: x + y, data)
    data = [(flatten(story), query, answer) for story, query, answer in data if
            not max_length or len(flatten(story)) < max_length]
    return data


def extract_file(challenge):
    """
    Extract tar file
    """
    tar = tarfile.open(DATA_GZ)
    train = get_stories(tar.extractfile(challenge.format('train')))
    test = get_stories(tar.extractfile(challenge.format('test')))
    return train, test


def get_vocab(train, test):
    """
    Get vocab
    """
    vocab = set()
    for story, q, answer in train + test:
        vocab |= set(story + q + [answer])
    vocab = sorted(vocab)
    word_idx = dict((c, i + 1) for i, c in enumerate(vocab))
    return vocab, word_idx


def vectorize_stories(data, word_idx):
    """
    Vectorize data
    """
    xs = []
    ys = []
    for story, query, answer in data:
        x = nlp(unicode(' '.join(story))).vector
        xq = nlp(unicode(' '.join(query))).vector
        join = np.concatenate((x, xq))
        y = np.zeros(len(word_idx) + 1)
        y[word_idx[answer]] = 1
        xs.append(join)
        ys.append(y)
    return xs, ys


def tokenize(sent):
    """
    Tokenize
    """
    split = [x.strip() for x in re.split('(\W+)?', sent) if
             x.strip() is not None and x.strip() not in spacy.en.language_data.STOP_WORDS]
    return [word.lower() for word in split if len(word) > 0]


if __name__ == '__main__':
    challenge = 'tasks_1-20_v1-2/en/qa1_single-supporting-fact_{}.txt'
    train, test = extract_file(challenge)
