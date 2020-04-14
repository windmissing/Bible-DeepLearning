[?]这里的表示貌似是个名字，是个什么东西？  

前面介绍的正则化大多是直接用于参数来惩罚复杂参数。  
这一节则通过惩罚激活神经元来惩罚复杂参数。  

[L1参数正则化](https://windmising.gitbook.io/bible-deeplearning/0introduction-1/0introduction/2l1)诱导稀疏参数是指“许多**参数(w)**为0”  
![](http://windmissing.github.io/images_for_gitbook/Bible-DeepLearning/10.png) 
但稀疏表示想要的是“许多**元素(h)**为0”  
![](http://windmissing.github.io/images_for_gitbook/Bible-DeepLearning/11.png) 
参考[L1参数正则化](https://windmising.gitbook.io/bible-deeplearning/0introduction-1/0introduction/2l1)让诱导参数为0的方法为“增加关于w的L1正则项”  
$$
\tilde J(w;X,y) = \alpha ||w||_1 + J(w;X,y)
$$

同理，要诱导元素h为0，就“增加关于h的正则项”。  
$$
\tilde J(w;X,y) = \alpha \Omega(h) + J(w;X,y)
$$
这个正则项也是L1正则项。  
$$
\Omega(h) = ||h||_1 = \sum_i|h_i|
$$
$$\Omega(h)$$也可以是其它正则项，例如[?]KL散度惩罚、[?]正则匹配追踪。  

---------------------------------------

前文所述的权重衰减直接惩罚模型参数。
另一种策略是惩罚神经网络中的激活单元，稀疏化激活单元。
这种策略间接地对模型参数施加了复杂惩罚。

我们已经讨论过（在\secref{sec:l1_regularization}中）$L^1$惩罚如何诱导稀疏的参数，即许多参数为零（或接近于零）。
另一方面，表示的稀疏性描述了一个表示中许多元素是零（或接近零）的情况。
我们可以线性回归的情况下简单说明这种区别：
\begin{aligned}
\underset{y \in \Bbb R^m}{
 \begin{bmatrix}
  18 \\  5 \\ 15 \\ -9 \\ -3
 \end{bmatrix}} = 
 \underset{A \in \Bbb R^{m \times n}}{
 \begin{bmatrix}
  4 & 0 & 0 & -2 & 0 & 0 \\
  0 & 0 & -1 & 0 & 3 & 0 \\
  0 & 5 & 0 & 0 & 0 & 0 \\
  1 & 0 & 0 & -1 & 0 & -4 \\
  1 & 0 & 0 & 0 & -5 & 0
 \end{bmatrix}} 
  \underset{x \in \Bbb R^n}{
  \begin{bmatrix}
 2 \\ 3\\ -2\\ -5 \\ 1 \\ 4
 \end{bmatrix} }\\
 \underset{y \in \Bbb R^m}{
 \begin{bmatrix}
  -14 \\  1 \\ 19 \\  2 \\ 23
 \end{bmatrix}} = 
 \underset{B \in \Bbb R^{m \times n}}{
 \begin{bmatrix}
  3 & -1 & 2 & -5 & 4 & 1 \\
  4 & 2 & -3 & -1 & 1 & 3 \\
  -1 & 5 & 4 & 2 & -3 & -2 \\
  3 & 1 & 2 & -3 & 0 & -3 \\
  -5 & 4 & -2 & 2 & -5 & -1
 \end{bmatrix}} 
  \underset{h \in \Bbb R^n}{
  \begin{bmatrix}
 0 \\ 2 \\ 0 \\ 0 \\ -3 \\ 0
 \end{bmatrix} }
\end{aligned}

% -- 247 --

第一个表达式是参数稀疏的线性回归模型的例子。
第二个表达式是数据$x$具有稀疏表示 $h$的线性回归。
也就是说，$h$是$x$的一个函数，在某种意义上表示存在于$x$中的信息，但只是用一个稀疏向量表示。

表示的正则化可以使用参数正则化中同种类型的机制实现。

表示的范数惩罚正则化是通过向损失函数 $J$添加对表示的范数惩罚来实现的。
我们将这个惩罚记作$\Omega(h)$。
和以前一样，我们将正则化后的损失函数记作$\tilde J$：
\begin{aligned}
 \tilde J(\theta; X, y) =  J(\theta; X, y)  + \alpha \Omega(h),
\end{aligned}
其中$\alpha \in [0, \infty]$ 权衡范数惩罚项的相对贡献，越大的$\alpha$对应越多的正则化。

正如对参数的$L^1$惩罚诱导参数稀疏性，对表示元素的$L^1$惩罚诱导稀疏的表示：
$\Omega(h) = \norm{h}_1 = \sum_i |h_i|$。
当然$L^1$惩罚是使表示稀疏的方法之一。
其他方法还包括从表示上的\ENNAME{Student}-$t$先验导出的惩罚\citep{Olshausen+Field-1996,Bergstra-Phd-2011}和KL散度惩罚\citep{Larochelle+Bengio-2008}，这些方法对于将表示中的元素约束于单位区间上特别有用。
\cite{HonglakL2008-small}和\cite{Goodfellow2009}都提供了正则化几个样本平均激活的例子，即令$\frac{1}{m}\sum_i h^{(i)}$接近某些目标值（如每项都是$.01$的向量）。

还有一些其他方法通过激活值的硬性约束来获得表示稀疏。
例如，\textbf{正交匹配追踪}(orthogonal matching pursuit)\citep{pati93orthogonal}通过解决以下约束优化问题将输入值$x$编码成表示 $h$
\begin{aligned}
 \underset{h, \norm{h}_0 < k}{\arg\min} \norm{x - W h}^2,
\end{aligned}
其中$\norm{h}_0 $是$h$中非零项的个数。
当$W$被约束为正交时，我们可以高效地解决这个问题。
这种方法通常被称为\ENNAME{OMP}-$k$，通过$k$指定允许的非零特征数量。
\cite{Coates2011b}证明\ENNAME{OMP}-$1$可以成为深度架构中非常有效的特征提取器。

% -- 248 --

含有隐藏单元的模型在本质上都能变得稀疏。
在本书中，我们将看到在各种情况下使用稀疏正则化的例子。