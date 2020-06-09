From 李宏毅

# Attention-based Model基础版（只读模型）

![](/assets/images/Chapter10/88.png)   

## 应用

1. seq2seq  
2. 阅读理解 squad比赛  

## 基于Attention-based技术的Memory Network  

![](/assets/images/Chapter10/89.png)   
1. 把Document中的每个句子转成向量，N个句子得到N个高维向量  
2. 降维，N个高维向量降到N个低维向量x  
3. 将Query转成向量q  
4. 计算q与x的match score，每个x对应一个score，定义为$\alpha$  
5. 根据$\alpha$对x做加权求和$\sum_i\alpha_i x_i$  
6. 5的结果和q一起进入一个DNN  
7. 由DNN产生answer  

# Attention-based Model进阶版

Sentence to vector can be jointly trained.  

## 改进1

![](/assets/images/Chapter10/90.png)   
Document产生2个向量，分别是x和h。  
第4步使用x计算match score，用于决定从哪些向量抽取信息。  
第5步使用h生成最终被抽取的信息。  

## 改进2
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

---------------------------------------------------------

From Ag

翻译模型的特点：  
![](/assets/images/Chapter10/56.png)   
先输入整个原文序列，再输出整个翻译序列。  
事实上，如果原文序列很长，一次性记住全部内容是困难的。  
改进方法：attention based模型，一次翻译一部分。  

# encoder  
使用双向RNN来计算每个单词的特征。双向RNN的unit可以是GRU或LSTM或其它类型的Unit。  
每个单词通过双向RNN会得到2个activition，分别定义为`\overrightarrow{a^t}`和`\overleftarrow{a^t}`。并用一个符号来表现这两个activation。    

```
a^t = (\overrightarrow{a^t}, \overleftarrow{a^t})
```

![](/assets/images/Chapter10/57.png)   

# decoder  
decoder部分的结构和翻译模型是基本上一样的。
![](/assets/images/Chapter10/58.png)   
decoder使用某个context作为输入，context用向量C表示，是一个与encoder中的activation有关的向量。    

# 注意力权重   

关键在于怎样连接encoder与decoder。翻译模型使用顺序连接，先encoder，再decoder。  
注意力模型使用注意力权重将encoder的activation和decoder的context联系到一起。  
![](/assets/images/Chapter10/59.png)   
定义注意力权重$\alpha^{t1,t2}$为：**当生成y的第t1个单词时，对原文第t2个词的注意力应该是多少？**    
$$
\begin{aligned}
\sum_{t2}\alpha^{t1,t2} = 1   \\
C^{t1} = \sum_{t2}\alpha^{t1, t2} a^{t2}
\end{aligned}
$$

**注意$a$和$\alpha$的区别。**  

计算$\alpha^{t1,t2}$的公式为：  
$$
\alpha^{t1,t2} = \frac{\exp(e^{t1,t2})}{\sum_{t2}\exp(e^{t1,t2})}
$$

后面没听懂，帖个图在这里。  
![](/assets/images/Chapter10/60.png)   

# 时间复杂度

注意力模型的时间复杂度为$O(n^3)$，复杂度有点高。  
由于机器翻译的输入、输出通常不是会太，因此$O(n^3)$也是可以接受的。  
