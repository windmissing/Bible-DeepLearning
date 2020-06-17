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