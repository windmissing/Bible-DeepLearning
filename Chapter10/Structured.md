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
