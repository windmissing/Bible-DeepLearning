

|欠定问题|问题表现|正则化解决方法|
|---|---|---|
|$$X^TX$$是奇异矩阵|许多算法如线性回归和PCA，都依赖对矩阵$$X^TX$$求逆。如果$$X^TX$$是奇异的，这些方法算法都会失效。|$$X^TX + aI$$|
|[?]问题没有闭式解可能是欠定问题|例如感知机中的w，如果w实现完美可分，那么2w会以更高似然实现完美可分，迭代算法会持续增加w，算法将不会收敛。|例如限制`||w||=1`|
|欠定线性方程组|方程组有无穷多解|[Moore_penrose](https://windmising.gitbook.io/mathematics-basic-for-ml/xian-xing-dai-shu/svd)|


--------------------------------------

> **[success]**  
欠约束问题也称欠定问题，指有无穷多解的问题。  
欠定问题会导致一些常用的方法变得不可用，可以使用正则化来解决欠定问题。  

在某些情况下，为了正确定义机器学习问题，正则化是必要的。  
> **[success]**  
欠定问题1：$X^\top X$是奇异矩阵  
问题表现：许多算法如[线性回归](https://windmising.gitbook.io/liu-yu-bo-play-with-machine-learning/5-1)和[PCA](https://windmising.gitbook.io/liu-yu-bo-play-with-machine-learning/7-1)，都依赖对矩阵$X^\top X$求逆。如果$X^\top X$是奇异的，这些方法算法都会失效。  
正则化的解决方法：$X^\top X + aI$  

机器学习中许多线性模型，包括线性回归和PCA，都依赖于对矩阵$X^\top X$求逆。
只要$X^\top X$是奇异的，这些方法就会失效。
当数据生成分布在一些方向上确实没有差异时，或因为例子较少（即相对输入特征的维数来说）而在一些方向上没有观察到方差时，这个矩阵就是奇异的。
在这种情况下，（人们往往使得）正则化的许多形式对应求逆$X^\top X + \alpha I$。
（因为）这个正则化矩阵可以保证是可逆的。

> **[success]**  
欠定问题2：问题没有闭式解可能是欠定问题   
问题表现：例如[感知机](https://windmissing.github.io/LiHang-TongJiXueXiFangFa/Chapter2/perceptron.html)中的w，如果w实现完美可分，那么2w会以更高似然实现完美可分，迭代算法会持续增加w，算法将不会收敛。    
正则化的解决方法：例如限制`||w||=1`  

相关矩阵可逆时，这些线性问题有闭式解。
没有闭式解的问题也可能是欠定的。
一个例子是应用于线性可分问题的逻辑回归。
如果权重向量$w$能够实现完美分类，那么$2 w$也会以更高似然实现完美分类。
类似随机梯度下降的迭代优化算法将持续增加$w$的大小，理论上永远不会停止。
在实践中，数值实现的梯度下降最终会达到导致数值溢出的超大权重，此时的行为将取决于程序员如何处理这些不是真正数字的值。

大多数形式的正则化能够保证应用于欠定问题的迭代方法收敛。
例如，当似然的斜率等于权重衰减的系数时， 权重衰减将阻止梯度下降继续增加权重的大小。  
> **[warning]** [?]

使用正则化解决欠定问题的想法不局限于机器学习。
同样的想法在几个基本线性代数问题中也非常有用。

> **[success]**  
欠定问题3：欠定线性方程组  
问题表现：方程组有无穷多解  
正则化的解决方法：[Moore_penrose](https://windmising.gitbook.io/mathematics-basic-for-ml/xian-xing-dai-shu/svd)

正如我们在\secref{sec:the_moore_penrose_pseudoinverse}看到的，我们可以使用\ENNAME{Moore-Penrose}求解欠定线性方程。 
回想$X$伪逆$X^+$的一个定义：  
$$
\begin{aligned} 
 X^+ = \lim_{\alpha \searrow 0} (X^\top X + \alpha I)^{-1}X^\top.
\end{aligned}
$$

现在我们可以将\secref{eq:729pseudo}看作进行具有权重衰减的线性回归。
具体来说，当正则化系数趋向0时，公式7.29是公式7.17的极限。  
> **[success]**  
[公式7.17](https://windmissing.github.io/Bible-DeepLearning/Chapter7/1ParameterNormPenalties/1L2.html)  

因此，我们可以将伪逆解释为使用正则化来稳定欠定问题。