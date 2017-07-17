import tensorflow as tf


class RNN(object):
    def __init__(self, rnn_cell, embed_hidden_size, sent_hidden_size, query_hidden_size, vocab_size):
        self.basic_cell = rnn_cell
        self.embed_hidden_size = embed_hidden_size
        self.sent_hidden_size = sent_hidden_size
        self.query_hidden_size = query_hidden_size
        self.vocab_size = vocab_size

    def inference(self, x, xq, dropout):
        # 获得story的嵌入
        x_embed = self.get_embed(x, dropout)
        # 获得question的嵌入
        xq_embed = self.get_embde(xq, dropout)
        xq_encode = self.get_rnn_encode(xq_embed)
        # 合并xq_encode和x_embed，将每个x_embed的输入加上question的编码信息，该机制类似attention机制
        merged = self.get_merged(x_embed, xq_encode)
        merged_encode = self.get_rnn_encode(merged)
        # 获得预测输入logits，注意这里采用最后的隐层信息来进行预测
        logits = self.get_logits(merged_encode)
        return logits

    def loss(self, logits, labels):
        '''
        由于该模型直接采用隐层输出进行预测，类似于一般的NN，因此可以简单的采用交叉熵求解
        :param logits:
        :param labels:
        :return:
        '''
        loss = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
        return tf.reduce_mean(loss)

    def train(self):
        pass

    def eval(self):
        pass

    def get_embed(self, input, dropout):
        '''
        用于将输入转换成word-embedding并添加dropout层
        :param input:
        :param dropout:
        :return:
        '''
        embed = tf.contrib.layers.embed_sequence(input, self.vocab_size, self.embed_hidden_size)
        dropout = tf.layers.dropout(embed, dropout)
        return dropout

    def get_rnn_encode(self, embed):
        '''
        进行rnn编码，最后返回隐层信息final_state
        :param embed:
        :return:
        '''
        cell = self.get_rnn_cell()
        _, final_states = self.cell_run(cell, embed)
        return final_states

    def get_rnn_cell(self):
        '''
        获得rnn基本单元
        :return:
        '''
        if self.basic_cell == 'rnn':
            cell_fn = tf.nn.rnn_cell.BasicRNNCell
        elif self.basic_cell == 'gru':
            cell_fn = tf.nn.rnn_cell.GRUCell
        elif self.basic_cell == 'lstm':
            cell_fn = tf.nn.rnn_cell.BasicLSTMCell
        else:
            raise Exception('model type not supported: {}'.format(self.basic_cell))

        # 建立单层神经元
        def build_cell(lstm_size):
            cell = cell_fn(lstm_size)
            return cell

        # 建立多层rnn结构
        cell = tf.nn.rnn_cell.MultiRNNCell([build_cell(self.rnn_size) for _ in range(self.rnn_layer)])
        return cell

    def cell_run(self, cell, embed):
        '''
        运行rnn
        :param cell:
        :param embed:
        :return:
        '''
        outputs, final_state = tf.nn.dynamic_rnn(cell, embed,
                                                 dtype=tf.float32)  # 这里只要声明了dtype，那么就可以不用声明初始state，默认为zero_state
        final_state = tf.identity(final_state, name='final_state')
        return outputs, final_state

    def get_merged(self, x_embed, xq_encode):
        '''
        将story和编码后的question整合在一块。不同于一般的cancat，这个整合是将xq_encode和所有的x_embed按时间步长挨个相加，可以看成一种新的rnn整合方式
        :param x_embed: [batch_size, seq_len, embed_hidden_size]
        :param xq_encode:[batch_size, embed_hidden_size]
        :return:
        '''
        seq_len = x_embed.shape().to_list()[1]
        xq_encode_repeat = tf.contrib.keras.layers.RepeatVector(seq_len)(xq_encode)
        assert x_embed.shape().to_list() == xq_encode_repeat.shape().to_list(), \
            'x_embed and xq_encode_repeat doesn\'t have same shape'
        merged = tf.add(x_embed, xq_encode_repeat)
        return merged

    def get_logits(self, merged_encode):
        '''
        根据隐层输出生成分类结果
        :param merged_encode:
        :return:
        '''
        return tf.layers.dense(merged_encode, self.vocab_size)
