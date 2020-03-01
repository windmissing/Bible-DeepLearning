$$
\hat y = \sigma(w^Th+b)
$$
其中$$\sigma$$为[logistic sigmoid函数](https://windmising.gitbook.io/mathematics-basic-for-ml/gai-shuai-lun/functions#logistic-sigmoid-han-shu)

[?]分对数这一段没看懂  

特点：  
1. 配合“用于最大似然的代价函数$$-\log P(y|x)$$”有一个很好的性质  
$$
J(\theta) = \xi((1-2y)z)
$$
2. 不能使用其它代价函数，例如MSE，会使神经元在|z|很大时饱和。  
3. sigmoid的output为(0,1)，不是[0,1]  
4. [?]最后一句没看懂