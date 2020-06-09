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

## 多领域 Embedding

1. 同一类别的图像出现在Embedding空间的同一位置  
2. 一张新图，通过Embedding向量周边向量的类别来确定这张新图的类别。  
参考上文提到的Zero-Shot问题。  
![](/assets/images/Chapter10/70.png)  

## 文件Embedding

不同长度的文件 --> 相同长度的vector  
静态Embedding法  
![](/assets/images/Chapter10/71.png)  
缺点：Bag of word没有考虑到内容中单词的顺序  
改进：没讲

## 领域 Neighbor Embedding

Manifold Learning，相当于非线性的降维  
经典例子：地球表面  
引入原因：由于低维空间的点在高维空间扭曲，导致原本的“距离”可能没有意义。  
解决方法：把点在低维空间拉平   
![](/assets/images/Chapter10/72.png)  

### 方法一：LLE

1. 任意选择一个点xi   
2. 任意选择点xi的Neighbour，即点xj  
3. 定义xi与xj的关系为$W_{i,j}$  
4. 最小化以下公式：  
$$
\sum_i||x^i - \sum_j W_{i,j}x^j||_2
$$

5. 降维，把xi, xj转成zi,zj，降维后$W_{i,j}$不变。  
$$
\sum_i||z^i - \sum_j W_{i,j}z^j||_2
$$

优点：  
不需要知道原xi, xj，只要知道$W_{i,j}$，就可以求出降维后的zi, zj。  

### 方法二：Laplacian Eigenmaps

两个点的距离不是欧氏距离，而是两个点间的[high density path](https://windmissing.github.io/Bible-DeepLearning/Chapter7/6SemiSupervised.html#%E7%AE%97%E6%B3%95%E4%BA%8C%EF%BC%9Agraph-based-approach)  

![](/assets/images/Chapter10/73.png)   

$$
S = \frac{1}{2}\sum_{i,j}W_{i,j}(z^i - z^j)^2
$$

$\sum_{i,j}$代表遍历所有的数据对，但只有$W_{i,j}$大的情况下，才会考虑对(z^i - z^j)^2影响。  
$W_{i,j}$大的情况下，(x^i - x^j)^2肯定小，按照以上公式，这种情况$(z^i - z^j)^2$也必须小。   
另外，为了防止训练结果为所有的z都为0，还需要再加一个要求：  
$$
Span(z1, z2, \cdots, zm) = R^m
$$

### 方法三：t-SNE

T-distributed Stochastic Neighbour Embedding

方法一、二存在的问题：  
只要求相似的点靠近，没有要求不同的点分开，所以最后所有的点都挤到一起。  
![](/assets/images/Chapter10/74.png)   
T-SNE可以解决这样的问题。  

1. 计算所有点对的相似度： S(xi, xj)  
2. Normalization  
$$
P(x^j|x^i) = \frac{S(x^i, x^j)}{sum_{k\neq i}S(x^i, x^k)}
$$

3. 假设xi, xj对应的转换结果为zi, zj，计算S'(zi, zj)  
4. Normalization  
$$
Q(z^j|z^i) = \frac{S'(z^i, z^j)}{sum_{k\neq i}S(z^i, z^k)}
$$

5. 计算分布P与分布Q的相似度，使用的指示是KL divergence(KL散度)  
$$
\begin{aligned}
L &=& \sum_i KL\left(P(*|x^i)||Q(*|z^i)\right)  \\
&=& \sum_i \sum_j P(x^j|x^i)\log \frac{P(x^j|x^i)}{Q(x^j|x^i)}
\end{aligned}
$$

缺点：
（1）计算量大  
解决方法：先使用PCA降维，例如：50维 --PCA--> 10维 --t-SNE--> 2维   
（2）如果来一个新的x，要求x对应的z，需要把算法重新跑一遍。  
解决方法：该算法不用于训练模型，而是主要用于数据的可视化  

t-SNE的一个关键的创新点：  
x的相似度计算公式S和z的相似度计算公式S'不同。  
$$
\begin{aligned}
S(x^i, x^j) = \exp(-||x^i-x^j||_2)  \\
S'(z^i, z^j) = 1/1 + ||z^i-z^j||_2
\end{aligned}
$$

S与S'的关系如图所示：  
![](/assets/images/Chapter10/75.png)   
当xi与xj接近时，zi与zj也比较接近。  
当xi与xj比较远时，zi与zj的间隔更远（强化gap）。  
效果：  
![](/assets/images/Chapter10/76.png)   