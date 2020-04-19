# RNN的其它应用

## many to one  
输入 vector sequence，输出：one vector  
例如：  
文件的情绪分析，28'32''  
文本的关键词，30'00''  

## many to many，output is shorter

输入输出都是vector sequence，输出的sequence更短。  
例如：语音辨识，输入声音信号，输出字符序列    
1. 将声音信号切成小段，每段不超过0.01s  
2. 把每一小段声音转成一个向量。  
3. 根据每个向量输出一个字符。  
4. 通常好个向量输出同一个字符，需要把重复的字符去掉。例如33'50''
由于输入和输出的长度不同，不管训练还是预测，都会遇到一个问题。  
训练时，多个向量输出同一个字符，如果把多余的字符去掉，但不会误伤叠词。  
解决方法：增加一个为空的字符，只把为空的字符去掉。  
预测时，某个向量的输出是真实字符还是空字符？  
解决方法：CTC training，即穷举所有正确的可能。 35'54''

## many to many, no limitation

输入和输出都是序列，不知道序列的长短。  
应用1：语言翻译  
训练方法：
1. 每输入一个英文单词，输出一个中文字符。  
2. 英文序列停止了，但中文序列继续输出。  
3. 在中文字符中增加一个停止字符。  
4. 当输出为停止字符时，停止输出。39'11''，40'52''

应用2：语言A的声音信号 -> 语言B的文字  
不需要经过A的文字作为中间转换。  
用于语言A没有对应的文字的场景。  

## Beyond Sequence

应用：句法分析树  
输入：序列  
输出：语言结构树  
使用seq2seq，而不是struct learning  
44'28''

## sequence to sequence, Auto-encoder

1. 用在文字上  
普通的encoder算法会忽略单词的顺序，因而影响句子的理解。  
Auto-encoder算法会考虑单词的顺序。  
单层模型：47'24''  
多层模型：48'28''  
2. 用在语音上  
把长度不固定的语音片段转成长度固定的向量。  
用处：语音搜索  
问：怎样把语音变成向量？  
答：encoder技术  
问：怎样比较？  
答：53'02''  
通常把encoder和decoder一起训练，51'30''

# Attention based mode

只读模型，57'36''  
可读可写模型，58'21''，又叫Neural Turing Machine  

## 阅读理解

### 文本QA
1. 通过Semantic分析，将每个句子转成一个向量  
2. 输入一个问题，由reading head controller决定，哪个句子与问题相关  
3. 把相关的句子读进模型，产生output
59'59''

### 视觉QA

1. 模型读入一张图  
2. 通过CNN，将每个小区域转成一个向量  
3. 输入一个问题，由reading head controller决定，哪个区域与问题相关  
4. 读入相关的区域，产生output
1:02'36''

### 语音QA

例如托福听力测试  
1:04'17''  
将语音识别、句义分析、attention的综合运用

# Deep Learing VS Structured Learning

Deep Learning是指RNN、LSTM、DNN等技术  
Structured Learning是指HMM、CRF、SVM、感知机等技术  

|Deep Learning|Structured Learning|
|---|---|
|无向的RNN不考虑整个序列|使用viterbi[?]，需要考虑整个序列|
|难以考虑label dependency|可以结合label dependency|
|cost不能反映error|cost能反应error|
|Deep|Linear|  

DL效果更好，但SL也很重要，常常将两者结合，能得到比较好的效果。  

DL与SL结合的例子：
1. 语音识别：CNN/LSTM/DNN + HMM  
HMM公式：  
$$
P(x, y) = P(y_1|start)\prod_{l=1}^{L-1}P(y|l+1|y_l)P(end|y_L)\prod_{l=1}^LP(x_l|y_l)
$$

将HMM应用于语音识别，那么x代表声音信号，y代表语音辨识的结果。  
公式中第一个连乘代表transition部分。这一部分由HMM训练。  
第二个连乘代表initial部分。这一部分由DL提供。  
$$
P(x_l|y_l) = \frac{P(y_l|x_l)P(x_l)}{P(y_l)}
$$

公式中，$P(y_l|x_l)$由DL训练，P(x_l)可以忽略，因为$x_l$代表输入,$P(y_l)$通过统计可得。  
2. semantic tagging：双向LSTM + CRF/Structured SVM
先用RNN找出feature  
由这些feature定义$\phi(x, y)$  
$\phi(x, y)$用于CRF/SVM

# is structured learning practical

1:22'12''，这部分没听懂
