# encoding=utf-8
import tensorflow as tf
from tensorflow.python.layers.core import Dense
import numpy as np


class S2S(object):
    def __init__(self, input_data, target_data, rnn_size, num_layers, keep_prob, batch_size, source_sequence_length,
                 target_sequence_length, max_target_sentence_length, source_vocab_size, target_vocab_size,
                 enc_embedding_size, dec_embedding_size, target_vocab_to_int):
        self.input_data = input_data
        self.target_data = target_data
        self.rnn_size = rnn_size
        self.num_layers = num_layers
        self.keep_prob = keep_prob
        self.batch_size = batch_size
        self.source_sequence_length = source_sequence_length
        self.target_sequence_length = target_sequence_length
        self.max_target_sentence_length = max_target_sentence_length
        self.source_vocab_size = source_vocab_size
        self.target_vocab_size = target_vocab_size
        self.enc_embedding_size = enc_embedding_size
        self.dec_embedding_size = dec_embedding_size
        self.target_vocab_to_int = target_vocab_to_int

    def seq2seq_model(self):
        '''
        构建s2s模型，得到解码的训练输出和测试输出
        '''
        _, final_state = self.encoding_layer()
        target_data_processed = self.process_decoder_input()
        training_decoder_output, inference_decoder_output = self.decoding_layer(target_data_processed, final_state)
        return training_decoder_output, inference_decoder_output

    def encoding_layer(self):
        '''
        编码过程，采用LSTM进行编码，并将输出结果和隐层信息返回
        '''
        embed_input = tf.contrib.layers.embed_sequence(self.input_data, self.source_vocab_size,
                                                       self.enc_embedding_size)

        def build_cell(lstm_size):
            cell = tf.contrib.rnn.LSTMCell(lstm_size, initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
            return cell

        cell = tf.nn.rnn_cell.MultiRNNCell([build_cell(self.rnn_size) for _ in range(self.num_layers)])
        cell = tf.contrib.rnn.DropoutWrapper(cell, self.keep_prob)
        output, final_state = tf.nn.dynamic_rnn(cell, embed_input, sequence_length=self.source_sequence_length,
                                                dtype=tf.float32)
        return output, final_state

    def process_decoder_input(self):
        """
        对于解码输入数据，将解码数据变换成以'<GO>'开始，并去除末尾的'<EOS>'
        """
        go_idx = self.target_vocab_to_int['<GO>']
        tiled_go_idx = tf.cast(
            tf.reshape(tf.tile(tf.constant([go_idx], dtype=tf.int32), [self.batch_size]), shape=[self.batch_size, -1]),
            tf.int32)
        return tf.concat([tiled_go_idx, self.target_data], axis=1)[:, :-1]

    def decoding_layer(self, dec_input, encoder_state):
        '''
        解码过程，在训练和测试阶段采用不同的方式
        '''
        dec_embeddings = tf.Variable(tf.random_uniform([self.target_vocab_size, self.dec_embedding_size]))
        dec_embed_input = tf.nn.embedding_lookup(dec_embeddings, dec_input)

        def make_cell(rnn_size):
            dec_cell = tf.contrib.rnn.LSTMCell(rnn_size,
                                               initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
            return dec_cell

        dec_cell = tf.contrib.rnn.MultiRNNCell([make_cell(self.rnn_size) for _ in range(self.num_layers)])
        output_layer = Dense(self.target_vocab_size,
                             kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))
        with tf.variable_scope("decode") as decoding_scope:
            training_decoder_output = self.decoding_layer_train(encoder_state, dec_cell, dec_embed_input, output_layer)
            decoding_scope.reuse_variables()
            start_of_sequence_id = self.target_vocab_to_int['<GO>']
            end_of_sequence_id = self.target_vocab_to_int['<EOS>']
            inference_decoder_output = self.decoding_layer_infer(encoder_state, dec_cell, dec_embeddings,
                                                                 start_of_sequence_id,
                                                                 end_of_sequence_id,
                                                                 output_layer)
        return training_decoder_output, inference_decoder_output

    def decoding_layer_train(self, encoder_state, dec_cell, dec_embed_input, output_layer):
        '''
        创建decode的训练过程，由于s2s的训练过程和测试过程
        '''
        training_helper = tf.contrib.seq2seq.TrainingHelper(inputs=dec_embed_input,
                                                            sequence_length=self.target_sequence_length,
                                                            time_major=False)
        training_decoder = tf.contrib.seq2seq.BasicDecoder(dec_cell,
                                                           training_helper,
                                                           encoder_state,
                                                           output_layer)
        training_decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(training_decoder,
                                                                          impute_finished=True,
                                                                          maximum_iterations=self.max_target_sentence_length)
        return training_decoder_output

    def decoding_layer_infer(self, encoder_state, dec_cell, dec_embeddings,
                             start_of_sequence_id,
                             end_of_sequence_id,
                             output_layer):
        '''
        创建decode的推理过程，由于跟训练过程不一样，因此需要单独建立
        '''
        start_tokens = tf.tile(tf.constant([start_of_sequence_id], dtype=tf.int32), [self.batch_size],
                               name='start_tokens')
        inference_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(dec_embeddings, start_tokens,
                                                                    end_of_sequence_id)

        inference_decoder = tf.contrib.seq2seq.BasicDecoder(dec_cell, inference_helper, encoder_state, output_layer)

        inference_decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(inference_decoder, impute_finished=True,
                                                                           maximum_iterations=self.max_target_sentence_length)
        return inference_decoder_output

    def loss(self, training_decoder_output, inference_decoder_output):
        '''
        计算损失
        '''
        training_logits = tf.identity(training_decoder_output.rnn_output, name='logits')
        inference_logits = tf.identity(inference_decoder_output.sample_id, name='predictions')
        masks = tf.sequence_mask(self.target_sequence_length, self.max_target_sentence_length, dtype=tf.float32,
                                 name='masks')
        cost = tf.contrib.seq2seq.sequence_loss(training_logits, self.target_data, masks)
        return cost, inference_logits

    def train(self, cost, learning_rate):
        # 采用Adam优化方式进行优化
        optimizer = tf.train.AdamOptimizer(learning_rate)

        # 梯度裁剪
        gradients = optimizer.compute_gradients(cost)
        capped_gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients if grad is not None]
        train_op = optimizer.apply_gradients(capped_gradients)
        return train_op

    def accuary(self, target, logits):
        """
        计算准确率
        """
        max_seq = max(target.shape[1], logits.shape[1])
        if max_seq - target.shape[1]:
            target = np.pad(
                target,
                [(0, 0), (0, max_seq - target.shape[1])],
                'constant')
        if max_seq - logits.shape[1]:
            logits = np.pad(
                logits,
                [(0, 0), (0, max_seq - logits.shape[1])],
                'constant')
        return np.mean(np.equal(target, logits))
