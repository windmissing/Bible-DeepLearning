SELU是一种自带[Normalization](https://windmissing.github.io/Bible-DeepLearning/Chapter8/7Strategies/1BatchNormalization.html)效果的Unit。  

ELU = Exponential Linear Unit  
SELU = Scaled ELU  
![](/assets/images/Chapter6/23.png)  

$$
a = 
\begin{cases}
\lambda z && z\ge0 \\
\lambda\alpha(e^z-1) && z < 0
\end{cases}
$$

$\alpha$和$\lambda$都是某个固定值。  

问：为什么说SELU自带Normalization功能？  
答：假设x的均值为0，方差为1，那么经过z=wx, a=SELU(z)的运算之后，得到的a也是均值为0，方差为1。  

**以上只是SELU的简化版。真正的RELU更复杂，可以让任意输入朝Normalization演化**。    