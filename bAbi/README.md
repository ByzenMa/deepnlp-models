1.内容介绍

用于复现bAbi模型，[参考文献](https://arxiv.org/abs/1502.05698v1)。代码参考keras对bAbi的rnn[实现](https://github.com/fchollet/keras/blob/master/examples/babi_rnn.py)

2.实验结果

- en(小数据集)

sk Number                  | FB LSTM Baseline | My LSTM | My GRU | My RNN
---                          | ---              | ---   | ---   | ---
QA1 - Single Supporting Fact | 50               | 52.8  |52.2   |57.8
QA2 - Two Supporting Facts   | 20               | 30.6  |35.4   |27.8
QA3 - Three Supporting Facts | 20               | 23.6  |24.2   |24.4
QA4 - Two Arg. Relations     | 61               | 70    |75.3   |66
QA5 - Three Arg. Relations   | 70               | 56.9  |80.3   |55
QA6 - yes/No Questions       | 48               | 52.5  |56.3   |53
QA7 - Counting               | 49               | 77.5  |79     |77.6
QA8 - Lists/Sets             | 45               | 73    |75.7   |64.5
QA9 - Simple Negation        | 64               | 64.6  |64     |66.6
QA10 - Indefinite Knowledge  | 44               | 48    |48.8   |58.6
QA11 - Basic Coreference     | 72               | 76.4  |81.5   |77.4
QA12 - Conjunction           | 74               | 75.9  |77     |76.5
QA13 - Compound Coreference  | 94               | 93.6  |93.6   |93.6
QA14 - Time Reasoning        | 27               | 22.9  |37.2   |26.3
QA15 - Basic Deduction       | 21               | 21.1  |43.9   |24.5
QA16 - Basic Induction       | 23               | 50.2  |49.9   |48.4
QA17 - Positional Reasoning  | 51               | 51.4  |49.8   |54.7
QA18 - Size Reasoning        | 52               | 91.5  |91.3   |91.2
QA19 - Path Finding          | 8                | 9.4   |10.5   |10.7
QA20 - Agent's Motivations   | 91               | 91.1  |96.9   |91

- en-10k(大数据集)

sk Number                  | My LSTM | My GRU | My RNN
---                        | ---   | ---   | ---
QA1 - Single Supporting Fact | 52.8  |99.2   |98.8
QA2 - Two Supporting Facts   | 30.6  |43.9   |34
QA3 - Three Supporting Facts | 23.6  |52.9   |21.9
QA4 - Two Arg. Relations     | 70    |99.2   |99.2
QA5 - Three Arg. Relations   | 56.9  |98.8   |89.1
QA6 - yes/No Questions       | 52.5  |90.2   |79.7
QA7 - Counting               | 77.5  |99     |89.1
QA8 - Lists/Sets             | 73    |99.2   |77.4
QA9 - Simple Negation        | 64.6  |99.2     |82.9
QA10 - Indefinite Knowledge  | 48    |99.1   |75.9
QA11 - Basic Coreference     | 76.4  |99.2   |81.2
QA12 - Conjunction           | 75.9  |99.2     |99
QA13 - Compound Coreference  | 93.6  |95.1   |93.6
QA14 - Time Reasoning        | 22.9  |81.4   |33.2
QA15 - Basic Deduction       | 21.1  |88.7   |33
QA16 - Basic Induction       | 50.2  |49.8   |51.1
QA17 - Positional Reasoning  | 51.4  |62.3   |52.7
QA18 - Size Reasoning        | 91.5  |97.2   |91.7
QA19 - Path Finding          | 9.4   |42.1   |11.1
QA20 - Agent's Motivations   | 91.1  |99.2   |92.3