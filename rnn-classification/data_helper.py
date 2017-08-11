# encoding = utf-8
"""
This file contains:
(1) load data form findata_reg2.csv,
(2) and then preprocess data;
(3) and then convert to tfrecord and save to data directory
"""
import tensorflow as tf
import pandas as pd
import numpy as np
import sklearn.model_selection as sk
import os
import pickle as pckl

DATA_PATH = 'data/findata_reg2.csv'
MONTH_NUMS = 12
FEATURE_DIM = 92
DATA_DIR = 'data/'
PICKLE_NAME = 'data.pckl'


def load_data():
    """
    Load data form data/findata_reg2.csv
    """
    assert tf.gfile.Exists(DATA_PATH), \
        "Data file %s is not exists" % DATA_PATH

    data = pd.read_csv(DATA_PATH)
    if data.shape[0] % MONTH_NUMS != 0:
        print "Data first dimension is not a multiple of %d." % MONTH_NUMS

    # As the company is recognised by org_id and accountbook. We find out all companies.
    comp_ids = data[['org_id', 'accountbook']].drop_duplicates()

    # Filter out companies that does not have MONTH_NUMS months data
    features = list()
    labels = list()
    feature_dim = data.shape[1] - 4
    for row in comp_ids.itertuples():
        comp_data = data[(data.org_id == row[1]) & (data.accountbook == row[2])]
        label_data = comp_data['zf_1'].drop_duplicates()
        if comp_data.shape[0] == MONTH_NUMS and label_data.shape[0] == 1:
            feature = np.reshape(comp_data.ix[:, 4:].copy().as_matrix(), [MONTH_NUMS, feature_dim])
            label = 0 if np.reshape(label_data.copy().as_matrix(), [1]) <= 20000 else 1
            if label == 0:
                features.append(feature)
                features.append(feature)
                labels.append(label)
                labels.append(label)
            else:
                features.append(feature)
                labels.append(label)
    labels = np.array(labels, dtype=np.int32)
    labels = np.reshape(labels, (labels.shape[0],))
    print 'labels contains %d class 1, %d class 0' % (np.sum(labels, axis=0), labels.shape[0] - np.sum(labels, axis=0))
    return np.array(features, dtype=np.float32), labels


def _byte_feature(x_row):
    """
    Change to byte feature
    """
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[x_row]))


def _int64_feature(y_row):
    """
    Change to int feature
    """
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[y_row]))


def convert_to_tfrecord(X, y, name):
    """
    Convert data into tfrecord
    """
    num_exampls = X.shape[0]
    assert num_exampls == y.shape[0], \
        "Expected same amount: X has %d examples, while y has %d examples" % (X.shape[0], y.shape[0])
    data_path = os.path.join(DATA_DIR + name + '.tfrecord')
    if tf.gfile.Exists(data_path):
        print "%s.tfrecord file already exists in %s" % (name, data_path)
        return

    print 'Start to convert to %s ' % data_path
    writer = tf.python_io.TFRecordWriter(path=data_path)
    for row_id in range(num_exampls):
        x_row = X[row_id].tostring()
        y_row = y[row_id]
        features_dict = {
            'X': _byte_feature(x_row),
            'y': _int64_feature(y_row)
        }
        example = tf.train.Example(features=tf.train.Features(feature=features_dict))
        writer.write(example.SerializeToString())
    writer.close()
    print 'Converting done'


def split_data(features, labels, test_split=0.1):
    """
    Split dataset into train and test parts
    """
    X_train, X_test, y_train, y_test = sk.train_test_split(features, labels, test_size=test_split)
    print 'X_train shape: ', X_train.shape
    print 'y_train shape: ', y_train.shape
    print 'X_test shape: ', X_test.shape
    print 'y_test shape: ', y_test.shape
    return X_train, y_train, X_test, y_test


def gen_batch(batch_size, features, labels):
    """
    Generate batch
    """
    assert features.shape[0] == labels.shape[0], \
        "Features dim %d, not same as labels dim %d" % (features.shape[0], labels.shape[0])
    num_data = features.shape[0]
    batchs = num_data / batch_size
    for i in range(batchs):
        features_batch = features[batch_size * i:batch_size * (i + 1)]
        labels_batch = labels[batch_size * i:batch_size * (i + 1)]
        yield features_batch, labels_batch


def convert():
    """
    Convert train and test to tfrecord format
    """
    features, labels = load_data()
    X_train, y_train, X_test, y_test = split_data(features, labels, 0.1)
    convert_to_tfrecord(X_train, y_train, 'train')
    convert_to_tfrecord(X_test, y_test, 'test')
    # Write to .pickle file
    data_dict = {
        'X_train': X_train,
        'y_train': y_train,
        'X_test': X_test,
        'y_test': y_test
    }
    pickle_file = os.path.join(DATA_DIR, PICKLE_NAME)
    with open(pickle_file, 'w') as file:
        print 'Start to dump .pickle file to %s' % pickle_file
        pckl.dump(data_dict, file)
        print 'Dump done'


if __name__ == '__main__':
    convert()
