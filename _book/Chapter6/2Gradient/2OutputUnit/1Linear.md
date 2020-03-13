[仿射函数](https://windmising.gitbook.io/mathematics-basic-for-ml/gao-deng-shu-xue/function)  

$$
\hat y(h) = W^Th + b
$$

特点：  
1. 假设p(y|x)符合高斯分布，则$$\hat y$$为p(y|x)的均值。  
2. [?]计算p(y|x)的协方差也更容易。  
3. 神经元不会饱和，易于使用基于梯度的优化算法。
[?][link](https://windmising.gitbook.io/nielsen-nndl/introduction-2/crossentropy-dai-jia-han-shu/2)上说使用cross-entropy代价函数就能解决输出神经元饱和的问题的。本文已经默认用cross-entropy了，为什么还要考虑饱和的问题？  