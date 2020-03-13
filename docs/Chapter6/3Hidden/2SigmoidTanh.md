[logistic sigmoid](https://windmising.gitbook.io/mathematics-basic-for-ml/gai-shuai-lun/functions):  
$$
g(z) = \sigma(z)
$$
[双曲正切激活函数](https://windmising.gitbook.io/nielsen-nndl/introduction-2/qi-ta-ji-shu/3)：  
$$
g(z) = tanh(z) = 2\sigma(2z-1)
$$

特点：  
1. 它们在ReLU出现之前应该广泛，现在不鼓励用于前馈网络了。  
2. 仅当z接近0时sigmoid才对输入敏感。  
广饱和性使它基于梯度的学习变得困难。  
需要使用“能抵消sigmoid饱和性”的代价函数。  
3. tanh比sigmoid表现。[?]解释没看懂。  