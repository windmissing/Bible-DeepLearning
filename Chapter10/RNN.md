# 什么是RNN

CNN的局限性：
某个输入对应的输出，不止取决于输入本身，还取决于之前的输入。  
单纯针对当前输入做训练难以得到好的效果。  5'39''。  
解决方法：把上一次的信息也作为这一次的输入。  

补充：1 of N encoding，把单词转成向量。  
（1）图（2'12''左），再加一个order  
(2)word washing，图（3'34''右）

RNN训练方法：  
上一次输入得到的a(有的地方记作h)存在memory中，供下一次输入的计算过程使用。6'31''  
现在每个unit的输入向量是4维，其中2维来自样本特征，2维来自上一次的a。  
对于第一次输入，只有来自样本特征，没有上一次的a，此时要给a一个初值。  

使用RNN对字符串中单词分类的例子 12'21''
图中每个x为1 of N encoding得到的向量。  
对某个单词的理解不止基于这个单词，还基于它前面的内容。  


# RNN的 各种架构

1. deep RNN, 14'17''
2. Elam Network
t-1的h作为t的输入， 14'50''左  
3. Jordan Network  
t-1的y作为t的输入，14'50''右  
Jordan的效果优于Elam Network，因为作为输入的y比h好控制。  
4. 双向RNN  
正向算一次得到h1，反向算一次得到h2  
h1和h2的结果得到y，16'42''  
优点：network在产生output时看过的input更广。  
5. LSTM, long short-term memory，长短期记忆单元  

|Gate Name|作用|1|0|
|---|---|---|---|
|Input Gate|输入是否能写入memory|能写入|不能写入|
|Output Gate|是否将memory cell中的内容输入|能输出|不能输出|
|Forget Gate|是否保留当前memroy cell中的内容|保留|不保留|

Gate的值其实不是非0即1，而是(0,1)的值，表示打开程度。  
所有Gate的值都是network自己学到的，激活函数为sigmoid

因此LSTM有4个输入：
1. 要写入memory的值z
2. Input Gate的输入zi  
3. Output Gate的输入zo  
4. Forget Gate的输入zf  
LSTM有1个输出:memory中的新值。  

4个输入，4个输入都是由同一个input产生，分别使用4组参数而输入4个输入。这意味着它需要的参数是普通unit的4倍。    

问：长短期记忆单元到底是长期还是短期？  
答：本质上的短期记忆。因为相比于普通RNN的memory（没有记忆功能，t时刻的值直接覆盖t-1时刻的memory）来说，它的记忆是长的。  
long是short-term memory的形式词。  

图（24'09''）  
3个Gate的激活函数f通常是sigmoid函数。用于产生(0,1)的值，表示Gate的打开程度。  
c' = g(z)f(zi) + cf(zf)  
a = h(c')f(zo)    27'40''

xt向量的维数 = 下一层的LSTM unit个数。  
4组参数w,b   wi,bi  wo,bo   wf,bf  
分别与xt相乘得到z,zi,zo,zf  
z,zi,zo,zf向量与xt维数相同。  
向量中的每一项分别对应一个LSTM unit.  
实际上是整个向量一起算的，就像前馈网络中一次算一层一样。  
46'41'', 47'10''
42'04'', 44'59'', 45'51''. 46'12'', 46'40

# RNN的cost function  
相当于对每个时刻的输入作分类，而使用分类问题的代价函数，例如cross-entropy  
BPTT,backpropagation through time，算法用于计算RNN中参数的梯度。  
RNN的loss曲线可能是这样的：7'27''  
问：为什么RNN的loss会剧烈地抖动？  
答：RNN的error surface要么很平，要么很陡。平坦和陡峭的交界的地方构成悬崖。   
如果从悬崖下面update到悬崖上面，loss就会陡增。9'57''  
如果点正好落在悬崖上，梯度会突然非常大，然后参数就飞出去了。  
解决方法：截断Clipping，if gradient > threshold: gradient = threshold。  

问：为什么会有平坦和悬崖？  
答1：因为sigmoid unit?老师说不是这个原因。  
在前馈网络中，在hidden layer中使用sigmoid unit会导致这种情况。sigmoid unit->ReLU就能解决这个问题。  
答2：推导BPTT的公式可以分析出来原因。  
答3：直观分析。14'29''  
w的梯度= \delta w 对\delta C的影响。  
构造图中这样的一个简单的RNN，令unit的输入w为1，输出w为1，只在t0时刻有一个输入，值为1。观察unit的transition weight对C的影响。  
令w=1，则y1000=1  
w=1.01, y1000=20000 -- 悬崖
w=0.99, y1000=0 --- 平坦
w一但有影响，影响就是天崩地裂。这是因为同样的w在transition过程中反复使用。放大了它的作用。    
解决方法LSTM。  
LSTM只能解决梯度消失的问题，不能解决梯度爆炸的问题。悬崖仍然存在，通常将lr设置得比较小。  

问：为什么LSTM能解决梯度消失的问题？20'15''  
RNN和LSTM处理memory cell的操作不同。  
在RNN中，t时刻计算出的值直接存入memory cell中，覆盖t-1时刻的值。  
而在LSTM中，新memory = 旧memory * Gate + input。  
可见，只有Forget Gate不关闭，weight对memory的影响将永远存在。  
因此在实际训练过程中，应该将forget Gate设计为，在大多数情况下，forget gate都是开着的。  

# 其它解决梯度消失问题的unit

GRU --- Gate Recurrent Unit  
它是基于LSTM的改进：  
1. 只有2个Gate，将Input Gate与Forget Gate联动起来。当Input Gate打开时，Forget Gate自动关闭。    
2. 参数少了1/4，不容易发生过拟合。  
3. 最终效果差不多。  

Clockwise RNN  
SCRN 

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
