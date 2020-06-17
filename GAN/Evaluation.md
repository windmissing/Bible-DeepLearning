# 方法一：用likelihood来衡量

$$
L = \frac{1}{N}\sum_i\log P_G(x_i)
$$

其中：  
$x_i$为从真实样本中sample出的data  
$P_G(x_i)$为G产生数据xi的概率。实际上这个概率根本没法算    
解决方法：用Kernal Density Estimation来估计$P_G(x_i)$  
1. 用PG产生大量data --- 到底是多少data?  
2. 用高斯混合模型逼近这些data --- 到底是多少个高斯模型？  
3. 用高斯混合模型估测$P_G(x_i)$  
这个方法存在一些问题，除了上面提到的问题以外，还有一个关键问题：likelehood不一定能衡量PG的好坏。  

# 方法二：Objective Evaluation

用一个off-the-shell的classifier来判断oject的好坏。  

## 评分标准  

1. 单张图像够清晰  

用classifier对一张图像做分类，得到属于各个类的概率组成的分布。  
分布越集中，图像越清晰  

2. 整体够diverse

用G生成一组图像，分别对其中的每个图像做分类。  
每个图像得到一个分布，把所有的分布都加起来，得到综合的分布。  
综合分布越平均，整体越diverse。  

## 把评分标准数字化

Inception score =   
$$
\sum_x\sum_y P(y|x)\log P(y|x) - \sum_y P(y) \log P(y)
$$

公式第1项代表1的得分，第2项代表2的得分。  

## 未解决的问题

如果GAN只是记住了database里的某些图像，而不是创造新图像，用这种方式识别不出来。  