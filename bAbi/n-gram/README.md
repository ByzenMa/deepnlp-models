1.内容介绍

用于复现bAbi模型在n-gram模型下的效果，[参考文献](https://arxiv.org/abs/1502.05698v1)。代码使用tensorflow实现线性分类器。

2.实验结果

n-gram + LR

sk Number                  | FB n-gram Baseline | en | en-10k 
---                          | ---              | ---   | ---  
QA1 - Single Supporting Fact | 36               | 67.7  |64.2  
QA2 - Two Supporting Facts   | 2               | 16.6  |14.4  
QA3 - Three Supporting Facts | 7               | 18.1 |22.2  
QA4 - Two Arg. Relations     | 50               | 67.1    |66.4 
QA5 - Three Arg. Relations   | 20               | 56.4 |55.4 
QA6 - yes/No Questions       | 49               | 48.5  |51.7  
QA7 - Counting               | 52               | 71.6  |71.1   
QA8 - Lists/Sets             | 40               | 79.7    |88.1  
QA9 - Simple Negation        | 62               | 63.5  |52.4    
QA10 - Indefinite Knowledge  | 45               | 43.1    |44.1 
QA11 - Basic Coreference     | 29               | 20.3  |20.9
QA12 - Conjunction           | 9               | 54.1 |54    
QA13 - Compound Coreference  | 26               | 18.9  |19.2  
QA14 - Time Reasoning        | 19               | 74.1  |73.1 
QA15 - Basic Deduction       | 20               | 64.5  |67.2 
QA16 - Basic Induction       | 43               | 24.9  |23.2  
QA17 - Positional Reasoning  | 46               | 51.8  |47.8 
QA18 - Size Reasoning        | 52               | 52.5  |46.7  
QA19 - Path Finding          | 0               | 11.1  |12.3 
QA20 - Agent's Motivations   | 76               | 99.2  |99.2