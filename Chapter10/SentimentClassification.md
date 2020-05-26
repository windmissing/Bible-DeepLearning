情绪分类的主要挑战： 没有那么多有标记的训练集  
解决方法： Word Embedding  

# 简单的情绪分类  

![](/assets/images/Chapter10/46.png)   
1. 把one shot向量转成word embedding向量  
2. 句子中的所有向量做Average操作。  
3. 基于softmax生成分类结果  
优点：适用于任意长度的文本  
缺点：缺少词序，例如：  
Completely lacking in good taste, good service and good ambience. 
这句很可能会误分类。  

# RNN的Many to One结构

![](/assets/images/Chapter10/47.png)   