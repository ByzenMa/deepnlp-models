import tensorflow as tf

# 模型超参
tf.flags.DEFINE_integer("rnn_size", 128, "rnn size")
tf.flags.DEFINE_integer("num_layers", 1, "num layers")
tf.flags.DEFINE_integer("encoding_embedding_size", 100, "encoding embedding size")
tf.flags.DEFINE_integer("decoding_embedding_size", 100, "decoding embedding size")
tf.flags.DEFINE_float("init_learning_rate", 0.001, "initial learning rate")
tf.flags.DEFINE_float("gpu_fraction", 0.99, "gpu fraction")
tf.flags.DEFINE_float("dropout", 0.5, "dropout")

# 训练参数
tf.flags.DEFINE_integer("batch_size", 64, "batch size")
tf.flags.DEFINE_integer("num_epochs", 10, "number epochs")
tf.flags.DEFINE_integer("decay_step", 10, "number epochs")
tf.flags.DEFINE_float("decay_rate", 0.99, "gpu fraction")
FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()


