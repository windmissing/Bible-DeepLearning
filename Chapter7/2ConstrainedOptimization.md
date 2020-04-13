# 怎样把正则化转换为带约束的最优化问题  

考虑经过参数范数正则化的代价函数：  
$$
\begin{aligned}
 \tilde{J}(\theta;X, y) = J(\theta;X, y) + \alpha \Omega(\theta) .
\end{aligned}
$$

> **[success]**  
正则化的目的是限制参数的大小，手段是把参数加入代价函数中。  
要达到同样的目的，还可以有别的方法 --- 增加参数的约束。  
$$
\begin{aligned}
\min J(\theta;X,y) \\
s.t. \Omega(\theta)<k
\end{aligned}
$$

> 使用[KKT](https://windmising.gitbook.io/mathematics-basic-for-ml/shu-zhi-ji-suan/constrainedoptimization)算法求解带约束的最小化问题  
求解这些带约束的最小化问题不是这一节的重点。  
这一节的重点是比较“显式约束”和“参数范数惩罚”这两个思考角度。  

回顾[第4.4节](https://windmissing.github.io/mathematics_basic_for_ML/NumericalComputation/ConstrainedOptimization.html)我们可以构造一个广义Lagrange函数来最小化带约束的函数，即在原始目标函数上添加一系列惩罚项。
每个惩罚是一个被称为Karush–Kuhn–Tucker乘子的系数以及一个表示约束是否满足的函数之间的乘积。
如果我们想约束$\Omega(\theta)$小于某个常数$k$，我们可以构建广义Lagrange函数  
$$
\begin{aligned}
 \Bbb L(\theta, \alpha; X, y) = J(\theta; X, y) + \alpha (\Omega(\theta) - k).
\end{aligned}
$$

> **[success]**  
KKT算法第1步：写出广义Lagrange函数  

这个约束问题的解由下式给出  
$$
\begin{aligned}
 \theta^* = \arg\min_{\theta} \max_{\alpha, \alpha \geq 0} \Bbb L(\theta, \alpha).
\end{aligned}
$$

> **[success]**  
KKT算法第2步：把最优解写成极小极大问题的形式   
KKT算法第3步：转化为极大极小问题  
$$
\begin{aligned}
 \theta^* = \arg\max_{\alpha, \alpha \geq 0}\min_{\theta}\Bbb L(\theta, \alpha).
\end{aligned}
$$

如第4.4节中描述的，解决这个问题我们需要对$\theta$和$\alpha$都做出调整。
第4.5节给出了一个带$L^2$约束的线性回归实例。
还有许多不同的优化方法，有些可能会使用梯度下降而其他可能会使用梯度为0的解析解，但在所有过程中$\alpha$在$\Omega(\theta) > k$时必须增加，在$\Omega(\theta) < k$时必须减小。
所有正值的$\alpha$都鼓励$\Omega(\theta)$收缩。
最优值$\alpha^*$也将鼓励$\Omega(\theta)$收缩，但不会强到使得$\Omega(\theta)$小于$k$。

为了洞察约束的影响，我们可以固定$\alpha^*$，把这个问题看成只跟$\theta$有关的函数：  
$$
\begin{aligned}
 \theta^* = \arg\min_{\theta} \Bbb L(\theta, \alpha^*) = 
 \arg\min_{\theta}
 J(\theta; X, y) + \alpha^* \Omega(\theta).
\end{aligned}
$$

> **[success]**  
KKT算法第4步：固定$\alpha^*$，求$\theta^*$  

这和最小化$\tilde J$的正则化训练问题是完全一样的。  
> **[warning]** 怎么求$\theta^*$？也有梯度下降法？  

因此，我们可以把参数范数惩罚看作对权重强加的约束。  

# 正则化转 VS 带约束的最优化问题

如果$\Omega$是$L^2$范数，那么权重就是被约束在一个$L^2$球中。
如果$\Omega$是$L^1$范数，那么权重就是被约束在一个$L^1$范数限制的区域中。
通常我们不知道权重衰减系数$\alpha^*$约束的区域大小，因为$\alpha^*$的值不直接告诉我们$k$的值。
原则上我们可以解得$k$，但$k$和$\alpha^*$之间的关系取决于$J$的形式。
虽然我们不知道约束区域的确切大小，但我们可以通过增加或者减小$\alpha$来大致扩大或收缩约束区域。
较大的$\alpha$，将得到一个较小的约束区域。
较小的$\alpha$，将得到一个较大的约束区域。  


有时候，我们希望使用显式的限制，而不是惩罚。
如第4.4节所述，我们可以修改下降算法（如随机梯度下降算法），使其先计算$J(\theta)$的下降步，然后将$\theta$投影到满足$\Omega(\theta) < k$的最近点。  
> **[warning]** "将$\theta$投影到满足$\Omega(\theta) < k$的最近点"是什么意思？  

如果我们知道什么样的$k$是合适的，而不想花时间寻找对应于此$k$处的$\alpha$值，这会非常有用。

> **[success]**  
显式约束的好处1：如果知道什么样的k是合适的，使用“显式约束”可以省去不必要的搜索。    

另一个使用显式约束和重投影而不是使用惩罚强加约束的原因是惩罚可能会导致目标函数非凸而使算法陷入局部极小(对应于小的$\theta$）。  
> **[warning]** [?]重投影是什么意思？  

当训练神经网络时，这通常表现为训练带有几个``死亡单元''的神经网络。
这些单元不会对网络学到的函数有太大影响，因为进入或离开它们的权重都非常小。
当使用权重范数的惩罚训练时，即使可以通过增加权重以显著减少$J$，这些配置也可能是局部最优的。
因为重投影实现的显式约束不鼓励权重接近原点，所以在这些情况下效果更好。
通过重投影实现的显式约束只在权重变大并试图离开限制区域时产生作用。

> **[success]**  
效果略有不同：  
“参数范数惩罚”鼓励参数向原点靠近。  
而“显式约束”只关心参数是不是在约束区域内，只有在参数试图离开约束区域时约束才会起作用。  
显式约束的好处2：“参数范数惩罚”容易使算法陷入局部最小，而“显式约束”不会。  
显式约束的好处3：“显式约束”对优化过程增加了稳定性。

最后，因为重投影的显式约束还对优化过程增加了一定的稳定性，所以这是另一个好处。  
当使用较高的学习率时，很可能进入正反馈，即大的权重诱导大梯度，然后使得权重获得较大更新。
如果这些更新持续增加权重的大小，$\theta$就会迅速增大，直到离原点很远而发生溢出。
重投影的显式约束可以防止这种反馈环引起权重无限制地持续增加。
\cite{Hinton-et-al-arxiv2012}建议结合使用约束和高学习速率，这样能更快地探索参数空间，并保持一定的稳定性。
> **[warning]** 这一段没看懂。  

\cite{Hinton-et-al-arxiv2012}尤其推荐由\cite{Srebro05}引入的策略：约束神经网络层的权重矩阵每列的范数，而不是限制整个权重矩阵的~\ENNAME{Frobenius}~范数。
分别限制每一列的范数可以防止某一隐藏单元有非常大的权重。
如果我们将此约束转换成~\ENNAME{Lagrange}~函数中的一个惩罚，这将与$L^2$ 权重衰减类似但每个隐藏单元的权重都具有单独的~KKT~乘子。
每个~KKT~乘子分别会被动态更新，以使每个隐藏单元服从约束。  
> **[success]**  
显式约束的好处4：可以分别限制每一层的参数。   

在实践中，列范数的限制总是通过重投影的显式约束来实现。