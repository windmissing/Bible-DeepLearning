> **[success]**  
这里的"表示"是个名词  

前文所述的权重衰减直接惩罚模型参数。
另一种策略是惩罚神经网络中的激活单元，稀疏化激活单元。
这种策略间接地对模型参数施加了复杂惩罚。

> **[success] L1/L2正则化 VS 稀疏表示**    
正则化：对参数惩罚，诱导参数为0  
稀疏表示：对激活单元正则化，诱导元素为0   
正则化的稀疏表示，表达惩罚的方式相同，只是惩罚项的内容不同。  
**稀疏表示 VS [dropout](https://windmissing.github.io/Bible-DeepLearning/Chapter7/12Dropout.html)**  
目的都是减少激活单元，实现方式不同。  
稀疏表示使用惩罚或约束的方式。  
dropout使用抽样的方式。  

我们已经讨论过（在\secref{sec:l1_regularization}中）$L^1$惩罚如何诱导稀疏的参数，即许多参数为零（或接近于零）。
另一方面，表示的稀疏性描述了一个表示中许多元素是零（或接近零）的情况。
我们可以线性回归的情况下简单说明这种区别：  
![](/assets/images/Chapter7/8.png)   
> **[success]**  
[L1参数正则化](https://windmissing.github.io/Bible-DeepLearning/Chapter7/1ParameterNormPenalties/2L1.html)诱导稀疏参数是指“许多**参数(w)**为0”  

![](/assets/images/Chapter7/9.png)   
> **[success]**  
稀疏表示想要的是“许多**元素(h)**为0”  

第一个表达式是参数稀疏的线性回归模型的例子。  
第二个表达式是数据$x$具有稀疏表示 $h$的线性回归。  


也就是说，$h$是$x$的一个函数，在某种意义上表示存在于$x$中的信息，但只是用一个稀疏向量表示。

表示的正则化可以使用参数正则化中同种类型的机制实现。  
> **[success]**  
"表示的正则化"是“怎样惩罚激活单元”的意思。  
方式一：范数惩罚  
方式二：施加约束

表示的范数惩罚正则化是通过向损失函数 $J$添加对表示的范数惩罚来实现的。
我们将这个惩罚记作$\Omega(h)$。
和以前一样，我们将正则化后的损失函数记作$\tilde J$：  
$$
\begin{aligned}
 \tilde J(\theta; X, y) =  J(\theta; X, y)  + \alpha \Omega(h),
\end{aligned}
$$

其中$\alpha \in [0, \infty]$ 权衡范数惩罚项的相对贡献，越大的$\alpha$对应越多的正则化。

正如对参数的$L^1$惩罚诱导参数稀疏性，对表示元素的$L^1$惩罚诱导稀疏的表示：
$\Omega(h) = ||h||_1 = \sum_i |h_i|$。
当然$L^1$惩罚是使表示稀疏的方法之一。
其他方法还包括从表示上的Student-t先验导出的惩罚\citep{Olshausen+Field-1996,Bergstra-Phd-2011}和KL散度惩罚\citep{Larochelle+Bengio-2008}，这些方法对于将表示中的元素约束于单位区间上特别有用。  
> **[warning]** [?] Student-t先验导出的惩罚?  [?] [KL散度](https://windmissing.github.io/mathematics_basic_for_ML/Information/KL.html)惩罚?  

\cite{HonglakL2008-small}和\cite{Goodfellow2009}都提供了正则化几个样本平均激活的例子，即令$\frac{1}{m}\sum_i h^{(i)}$接近某些目标值（如每项都是$.01$的向量）。

还有一些其他方法通过激活值的硬性约束来获得表示稀疏。
例如，\textbf{正交匹配追踪}(orthogonal matching pursuit)\citep{pati93orthogonal}通过解决以下约束优化问题将输入值$x$编码成表示 $h$  
$$
\begin{aligned}
 {\arg\min}_{h, ||h||_0 < k} ||x - W h||^2,
\end{aligned}
$$

其中$||h||_0 $是$h$中非零项的个数。
当$W$被约束为正交时，我们可以高效地解决这个问题。  
> **[warning]**  怎么将W约束为正交？  

这种方法通常被称为\ENNAME{OMP}-$k$，通过$k$指定允许的非零特征数量。
\cite{Coates2011b}证明\ENNAME{OMP}-$1$可以成为深度架构中非常有效的特征提取器。

含有隐藏单元的模型在本质上都能变得稀疏。
在本书中，我们将看到在各种情况下使用稀疏正则化的例子。  
> **[warning]**  例子在哪？ 







