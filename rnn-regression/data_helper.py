# encoding=utf-8
from sklearn.model_selection import train_test_split
import pickle as pckl
import pandas as pd
import numpy as np
import tensorflow as tf

PATH = 'data/findata_reg2.csv'
DATA_PAHT = 'data/credit.pckl'
MONTHS_NUM = 12
FEATURE_DIM = 92
SEED = 2017


def load_data():
    '''
    加载数据
    '''
    data = pd.read_csv(PATH)
    companies = data[['org_id', 'accountbook']].drop_duplicates()
    print '共有数据%d条' % data.shape[0]

    if companies.shape[0] * MONTHS_NUM != data.shape[0]:
        print '存在不包含12个月的数据，进行数据清洗'

    feature_dim = data.shape[1] - 4  # 第一列是org_id, 第二列是accountbook, 第三列是period，以及第四列是审批的金额
    assert feature_dim == FEATURE_DIM, \
        '数据维度为%d，期望维度为%d' % (feature_dim, FEATURE_DIM)

    # 去除不满12个月的数据
    for row in companies.itertuples():
        comp_data = data[(data.org_id == row[1]) & (data.accountbook == row[2])]
        if comp_data.shape[0] != MONTHS_NUM or comp_data.iloc[0]['zf_1'] <= 20000:
            data = data.drop(data[(data.org_id == row[1]) & (data.accountbook == row[2])].index)

    labels = np.reshape(data[['org_id', 'accountbook', 'zf_1']].drop_duplicates()[['zf_1']].as_matrix(),
                        newshape=(-1, 1))
    # 将标签缩小10000，否则loss过大不易训练
    labels = np.divide(labels, 10000)
    features = np.reshape(data.ix[:, 4:].copy().as_matrix(), newshape=(-1, MONTHS_NUM, feature_dim))
    return features, labels


def pickle_data(features, labels, test_size=0.1):
    # 采用sklearn划分数据并保存成.pickle文件便于以后使用
    if tf.gfile.Exists(DATA_PAHT):
        print 'pickle文件已经存在'
    else:
        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=test_size, random_state=0)
        data_dict = {'X_train': X_train, 'X_test': X_test, 'y_train': y_train, 'y_test': y_test}
        with open(DATA_PAHT, 'w') as data:
            pckl.dump(data_dict, data)


def generate_batch(batch_size, X_data, y_data):
    np.random.seed(SEED)
    data_len = X_data.shape[0]
    assert data_len == y_data.shape[0], \
        '特征和标签的维度不一致'
    batchs = data_len / batch_size
    perm = np.random.permutation(data_len)
    # Shuffle数据并截断成batch_size的整数倍
    X = np.split(X_data[perm][:batchs * batch_size], batchs, axis=0)
    y = np.split(y_data[perm][:batchs * batch_size], batchs, axis=0)
    for X_batch, y_batch in zip(X, y):
        yield X_batch, y_batch


if __name__ == '__main__':
    # features, labels = load_data()
    # print features.shape, features.shape[0] * 12
    # print max(labels), min(labels)
    # pickle_data(features, labels)
    pass
