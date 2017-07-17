####1.内容介绍

用于复现bAbi模型，[参考文献](https://arxiv.org/abs/1502.05698v1)。代码参考keras对bAbi的rnn[实现](https://github.com/fchollet/keras/blob/master/examples/babi_rnn.py)

####2.实验结果
sk Number                  | FB LSTM Baseline | My LSTM | My GRU
---                          | ---              | ---   | ---   
QA1 - Single Supporting Fact | 50               | 52.3  |
QA2 - Two Supporting Facts   | 20               | -     |
QA3 - Three Supporting Facts | 20               | -     |
QA4 - Two Arg. Relations     | 61               | -     |
QA5 - Three Arg. Relations   | 70               | -     |
QA6 - yes/No Questions       | 48               | -     |
QA7 - Counting               | 49               | -     |
QA8 - Lists/Sets             | 45               | -     |
QA9 - Simple Negation        | 64               | -     |
QA10 - Indefinite Knowledge  | 44               | -     |
QA11 - Basic Coreference     | 72               | -     |
QA12 - Conjunction           | 74               | -     |
QA13 - Compound Coreference  | 94               | -     |
QA14 - Time Reasoning        | 27               | -     |
QA15 - Basic Deduction       | 21               | -     |
QA16 - Basic Induction       | 23               | -     |
QA17 - Positional Reasoning  | 51               | -     |
QA18 - Size Reasoning        | 52               | -     |
QA19 - Path Finding          | 8                | -     |
QA20 - Agent's Motivations   | 91               | -     |