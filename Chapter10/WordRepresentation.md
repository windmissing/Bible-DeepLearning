# One-Hot表示法

例如单词表为：lexicon = {apple, ball, cat}
那么这三个单词的向量依次为：  
apple = [1 0 0 0]  
ball = [0 1 0 0]  
cat = [0 0 1 0]
其它单词 = [0 0 0 1]  

缺点：  
把每个词孤立起来（任意两个向量的内积为0），这使得模型对词的泛化能力不强。  
例如，从“I want a glass of orange juice”泛化到“I want a glass of apple juice.”  
# word hashing：

![](/assets/images/Chapter10/39.png)

# 特征化表示： word embedding

![](/assets/images/Chapter10/40.png)  
这只是一个人为构造的例子。实际上这个矩阵中每一行的含义以及矩阵中的数值都是DL学习出来的。  

## 可视化embedding

使用t-SNE把n维的embedding转换到二维的空间。  
![](/assets/images/Chapter10/41.png)  

## word embedding的优点  

1. word embedding通过非常大量的语料库（1-100B）生成。  
使用者可以预先训练好或者直接下载开源的word embedding，然后把它迁移到自己的模型中并微调。  
2. 即使自己的模型只有很小的训练集（100K），也能有比较好的效果。  
3. embedding向量比one-hot向量小。  

word embedding广泛应用于NLP问题中，这是一种迁移学习。  

word embedding和[face encoding](https://windmissing.github.io/Bible-DeepLearning/Chapter9/FaceDetection.html)的原理类似，区别是使用时：  
face encoding输入的可能是没有见过的图像。  
word embedding输入的是字典中的单词。  
