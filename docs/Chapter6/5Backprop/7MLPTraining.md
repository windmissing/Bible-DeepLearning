这一节将6.5.4的算法进一步细化。  

假设这是一个三层的MLP网络。  
输入层：X
隐藏层：参数为W(1)，输入为X，激活函数为H=max{0, XW(1)}  
输出层：参数为W(2)，输入为H，用于计算类的非归一化对数的概率。  

[?]类的非归一化对数概率？

$$J_{MLE}$$为非归一化的对数概率的交叉熵。  
正则项为W(1)和W(2)的二阶范式。  
最终cost为：  
$$
J = J_{MLE} + \lambda\left(\sum_{i,j}(W^{(1)}_{i,j})^2 + (W^{(2)}_{i,j})^2\right)
$$

图为这个网络的正反传播的计算图：  
![](http://windmissing.github.io/images_for_gitbook/Bible-DeepLearning/6.png)  
[?]反向传播算法可以自动生成梯度？这句话什么意思？