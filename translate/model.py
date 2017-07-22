import tensorflow as tf
from tensorflow.python.layers.core import Dense


class S2S(object):
    def __init__(self):
        pass

    def seq2seq_model(self, input_data, target_data, keep_prob, batch_size,
                      source_sequence_length, target_sequence_length,
                      max_target_sentence_length,
                      source_vocab_size, target_vocab_size,
                      enc_embedding_size, dec_embedding_size,
                      rnn_size, num_layers, target_vocab_to_int):
        output, final_state = self.encoding_layer(input_data, rnn_size, num_layers, keep_prob, source_sequence_length,
                                                  source_vocab_size, enc_embedding_size)
        target_data_processed = self.process_decoder_input(target_data, target_vocab_to_int, batch_size)
        training_decoder_output, inference_decoder_output = self.decoding_layer(target_data_processed, final_state,
                                                                                target_sequence_length,
                                                                                max_target_sentence_length, rnn_size,
                                                                                num_layers,
                                                                                target_vocab_to_int, target_vocab_size,
                                                                                batch_size, keep_prob,
                                                                                dec_embedding_size)
        return training_decoder_output, inference_decoder_output

    def encoding_layer(self, rnn_inputs, rnn_size, num_layers, keep_prob,
                       source_sequence_length, source_vocab_size,
                       encoding_embedding_size):
        '''
        编码过程，采用LSTM进行编码，并将输出结果和隐层信息返回
        '''
        embed_input = tf.contrib.layers.embed_sequence(rnn_inputs, source_vocab_size, encoding_embedding_size)

        def build_cell(lstm_size):
            cell = tf.contrib.rnn.LSTMCell(lstm_size, initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
            return cell

        cell = tf.nn.rnn_cell.MultiRNNCell([build_cell(rnn_size) for _ in range(num_layers)])
        cell = tf.contrib.rnn.DropoutWrapper(cell, keep_prob)
        output, final_state = tf.nn.dynamic_rnn(cell, embed_input, sequence_length=source_sequence_length,
                                                dtype=tf.float32)
        return output, final_state

    def process_decoder_input(self, target_data, target_vocab_to_int, batch_size):
        """
        对于解码输入数据，将解码数据变换成以'<GO>'开始，并去除末尾的'<EOS>'
        """
        go_idx = target_vocab_to_int['<GO>']
        tiled_go_idx = tf.cast(
            tf.reshape(tf.tile(tf.constant([go_idx], dtype=tf.int32), [batch_size]), shape=[batch_size, -1]), tf.int32)
        return tf.concat([tiled_go_idx, target_data], axis=1)[:, :-1]

    def decoding_layer(self, dec_input, encoder_state,
                       target_sequence_length, max_target_sequence_length,
                       rnn_size,
                       num_layers, target_vocab_to_int, target_vocab_size,
                       batch_size, keep_prob, decoding_embedding_size):
        '''
        解码过程，在训练和测试阶段采用不同的方式
        '''
        dec_embeddings = tf.Variable(tf.random_uniform([target_vocab_size, decoding_embedding_size]))
        dec_embed_input = tf.nn.embedding_lookup(dec_embeddings, dec_input)

        def make_cell(rnn_size):
            dec_cell = tf.contrib.rnn.LSTMCell(rnn_size,
                                               initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
            return dec_cell

        dec_cell = tf.contrib.rnn.MultiRNNCell([make_cell(rnn_size) for _ in range(num_layers)])
        output_layer = Dense(target_vocab_size,
                             kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))
        with tf.variable_scope("decode") as decoding_scope:
            training_decoder_output = self.decoding_layer_train(encoder_state, dec_cell, dec_embed_input,
                                                                target_sequence_length, max_target_sequence_length,
                                                                output_layer, keep_prob)
            decoding_scope.reuse_variables()
            start_of_sequence_id = target_vocab_to_int['<GO>']
            end_of_sequence_id = target_vocab_to_int['<EOS>']
            inference_decoder_output = self.decoding_layer_infer(encoder_state, dec_cell, dec_embeddings,
                                                                 start_of_sequence_id,
                                                                 end_of_sequence_id, max_target_sequence_length,
                                                                 target_vocab_size, output_layer, batch_size, keep_prob)
        return training_decoder_output, inference_decoder_output

    def decoding_layer_train(self, encoder_state, dec_cell, dec_embed_input,
                             target_sequence_length, max_summary_length,
                             output_layer):
        '''
        创建decode的训练过程，由于s2s的训练过程和测试过程
        '''
        training_helper = tf.contrib.seq2seq.TrainingHelper(inputs=dec_embed_input,
                                                            sequence_length=target_sequence_length,
                                                            time_major=False)
        training_decoder = tf.contrib.seq2seq.BasicDecoder(dec_cell,
                                                           training_helper,
                                                           encoder_state,
                                                           output_layer)
        training_decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(training_decoder,
                                                                          impute_finished=True,
                                                                          maximum_iterations=max_summary_length)
        return training_decoder_output

    def decoding_layer_infer(self, encoder_state, dec_cell,
                             dec_embeddings, start_of_sequence_id,
                             end_of_sequence_id, max_target_sequence_length,
                             output_layer, batch_size):
        '''
        创建decode的推理过程，由于跟训练过程不一样，因此需要单独建立
        '''
        start_tokens = tf.tile(tf.constant([start_of_sequence_id], dtype=tf.int32), [batch_size], name='start_tokens')
        inference_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(dec_embeddings, start_tokens,
                                                                    end_of_sequence_id)

        inference_decoder = tf.contrib.seq2seq.BasicDecoder(dec_cell, inference_helper, encoder_state, output_layer)

        inference_decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(inference_decoder, impute_finished=True,
                                                                           maximum_iterations=max_target_sequence_length)
        return inference_decoder_output
