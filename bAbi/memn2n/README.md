1.内容介绍

用于复现bAbi模型在MemN2N模型下的效果，[参考文献](https://arxiv.org/abs/1503.08895)。代码参考domluna的[tensorflow实现](https://github.com/domluna/memn2n)

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


n-gram + NN

sk Number                  | FB n-gram Baseline | en | en-10k 
---                          | ---              | ---   | ---  
QA1 - Single Supporting Fact | 36               | 66.3  |62.9  
QA2 - Two Supporting Facts   | 2               | 19.9  |17.8  
QA3 - Three Supporting Facts | 7               | 21.4 |24.7  
QA4 - Two Arg. Relations     | 50               | 66.2    |68.4 
QA5 - Three Arg. Relations   | 20               | 56.8 |59.4 
QA6 - yes/No Questions       | 49               | 78.2  |78  
QA7 - Counting               | 52               | 83.5  |84   
QA8 - Lists/Sets             | 40               | 91.6    |96.8 
QA9 - Simple Negation        | 62               | 70.9  |71.7    
QA10 - Indefinite Knowledge  | 45               | 68.4    |70.6 
QA11 - Basic Coreference     | 29               | 21.7  |22.1
QA12 - Conjunction           | 9               | 55.4 |55.3    
QA13 - Compound Coreference  | 26               | 22.5  |20.5  
QA14 - Time Reasoning        | 19               | 73.1 |74.1 
QA15 - Basic Deduction       | 20               | 63.7  |69 
QA16 - Basic Induction       | 43               | 24.9  |25.5  
QA17 - Positional Reasoning  | 46               | 48.8  |62.2 
QA18 - Size Reasoning        | 52               | 54.2  |54.9  
QA19 - Path Finding          | 0               | 12  |14.2 
QA20 - Agent's Motivations   | 76               | 99.2  |99.2