目标：通过word的context来理解word的关系。  
难点：怎样挖掘word的context?  

# Count based算法

原理：Wi和Wj经常co-occor，则V(Wi)和V(Wj)接近。  

## Glove Vector

Ag也有讲这个算法[link](https://windmissing.github.io/Bible-DeepLearning/Chapter10/ReasonableAnalogies.html)  

# Predition Based算法

## 基本算法

1. 训练一个NN，输入$w_{i-1}$，预测$w_i$是每个单词的概率。  
![](/assets/images/Chapter10/64.png)  
2. 把NN中的第1个hidden layer拿出来，即z  
3. 用z代表单词wi  

原文中NN部分只有1个hidden layer。因此1个hidden layer的计算量，因此可以跑大量的Data。  

## 算法拓展

基于$w_{i-1}$  拓展为  基于wi的几N个单词，  
同时结合参数共享的思想。  
![](/assets/images/Chapter10/65.png)  
参数共享的原理：同一个单词放在wi之前的不同位置，效果应该相同。  

## 算法变形

（1）基于wi前后的单词预测wi  
（2）基于wi预测wi前后的单词  
![](/assets/images/Chapter10/66.png)  

# Embedding算法的应用

## Word Embedding

1. 相同关系的两个词的向量的相对位置类似  
![](/assets/images/Chapter10/67.png)  
图中左边为国家与首都的向量关系，右边为动词三次的向量关系。  
2. 相同关系的两个词的向量相减，结果落在同一区域。  
![](/assets/images/Chapter10/68.png)   

## 多语言 Embedding

![](/assets/images/Chapter10/69.png)   

1. 对两种语言分别做embedding，此时两种embedding没有任何关系。  
2. 基于一些确定的中英词对，将两种embedding映射到同一空间。图如图中绿底中文和绿框英文。  
3. 此时出现新的中文embedding与英文embedding，经过相同的映射后，会出现在附近的位置。例如图中绿底中文和黄底英文。  