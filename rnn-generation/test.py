import tensorflow as tf
import numpy as np

def get_tensor_by_name():
    inputTensor = tf.Graph.get_tensor_by_name('')
    np.array([1,3]).shape


def get_batches(int_text, batch_size, seq_length):
    """
    Return batches of input and target
    :param int_text: Text with the words replaced by their ids
    :param batch_size: The size of batch
    :param seq_length: The length of sequence
    :return: Batches as a Numpy array
    """
    text_len = len(int_text)
    batchs = text_len / (batch_size * seq_length)
    if text_len < batchs * batch_size * seq_length + 1:
        batchs -= 1
    if batchs < 1:
        return []
    input_len = batchs * batch_size * seq_length
    input_text = np.array(int_text[:input_len]).reshape([batch_size, -1])
    target_text = np.array(int_text[1:input_len + 1]).reshape([batch_size, -1])
    batchs_data = np.ndarray((batchs, 2, batch_size, seq_length), dtype=np.int32)
    for i in range(batchs):
        input_batch = input_text[:, i * seq_length:(i + 1) * seq_length]
        target_batch = target_text[:, i * seq_length:(i + 1) * seq_length]
        batchs_data[i, 0, :, :] = input_batch
        batchs_data[i, 1, :, :] = target_batch
    return batchs_data


def create_lookup_tables(text):
    """
    Create lookup tables for vocabulary
    :param text: The text of tv scripts split into words
    :return: A tuple of dicts (vocab_to_int, int_to_vocab)
    """
    vocab_to_int = {}
    for word in text.split(' '):
        if word not in vocab_to_int:
            vocab_to_int[word] = len(vocab_to_int)
    int_to_vocab = dict((v, k) for k, v in vocab_to_int.iteritems())
    return vocab_to_int, int_to_vocab


if __name__ == '__main__':
    int_text = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    batch_size = 2
    seq_length = 3
    print get_batches(int_text, batch_size, seq_length)
