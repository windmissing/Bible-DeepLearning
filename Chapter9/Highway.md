# GRU VS Highway Netword  

Highway Network基于GRU对Unit做了一些改进：  

- GRU  
![](/assets/images/Chapter9/34.png)   

- Highway Network Unit  
![](/assets/images/Chapter9/35.png)   

主要改进为：  
1. 去掉 Input $x^t$和Output $y^t$，只有第一个Unit有Input，最后一个Unit有Output  
2. 输入$h^t$换成$a^{t-1}$  
3. 去掉reset gate，保证$a^{t-1}$一定能进入下一个step  

Highway Network Unit的计算过程：  
$$
\begin{aligned}
h' = \sigma(W a^{t-1})  \\
z = \sigma(W' a^{t-1})  \\
a^t = z \odot a^{t-1} + (1-z)\odot h'
\end{aligned}
$$

这相当于在layer方向增加gate，以达到使layer更深的目的。  
如果只接将z设置成0.5，就成了[残差网络](https://windmissing.github.io/Bible-DeepLearning/Chapter9/ResNet.html)。  
![](/assets/images/Chapter9/36.png)  

Highway Network可以看作是Network自动学到要有多少hidden layer。  
![](/assets/images/Chapter9/37.png)   
根据data决定实际使用几层layer。  
