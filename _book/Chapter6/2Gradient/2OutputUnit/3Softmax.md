$$
z = W^Th+b \\
softmax(z)_i = \frac{\exp(z_i)}{sum_j \exp(z_j)}
$$

特点：  
1. 常用作分类器的输出  
2. 使用最大对数似然训练时效果好（类似sigmoid）  
3. 强烈地惩罚最活跃的不正确预测  
4. 其它许多代价函数不适用，因为它们不能抵消softmax中的指数，例如MSE  
5. [?]softmax激活函数会饱和这一段没看懂，2不是已经解决饱和问题了吗？  
6. 一种softmax函数的变化：  
$$
softmax(z) = softmax(z - \max z_i)
$$
7. softmax的限制版本：令z的一个元素为固定值  
8. soft是将函数变得连续可微的意思。softmax其实应该命名为softargmax。  