> **[info]** condition在这里应译为“状态”而不是“条件”。因此此处的“ill condition”译为病态。但在本书其它地方将它译为“状态条件”，这是不对的。  

在优化凸函数时，会遇到一些挑战。
这其中最突出的是Hessian矩阵$H$的病态。
这是数值优化、凸优化或其他形式的优化中普遍存在的问题，更多细节请回顾\sec?。

病态问题一般被认为存在于神经网络训练过程中。
**病态体现在随机梯度下降会"卡"在某些地方，此时即使很小的更新步长也会增加代价函数。**  
> **[success]** 
[病态矩阵](https://windmissing.github.io/mathematics_basic_for_ML/LinearAlgebra/IllConditioning.html)  
[Hessian矩阵的病态问题](https://windmissing.github.io/mathematics_basic_for_ML/LinearAlgebra/Hessian.html)  
问：病态的原始定义为“输入的微小差异导致输出的巨大变化。”但病态在神经网络中的表现为“即使很小的更新步长也会增加代价函数。”这两者是怎么联系上的？  
答：病态的H表现为f的等高线是同心椭圆。病态越严重，椭圆越扁，即长轴和短轴差别越大。  
这种情况有点像没有做特征归一化的梯度下降法，在尺度比较大的特征上，学习率太小会难以收敛。在尺度较小的特征上，学习率太大会震荡而无法收敛。  
这里的情况有点类似，在长轴方向上，学习率太小会难以收敛。在短轴方向上，学习率太大会震荡而无法收敛。  
“即使很小的更新步长也会增加代价函数”就是在长轴方向上难以收敛的表现。  
“输入的微小差异导致输出的巨大变化”就是在短轴方向上震荡的表现。  
特征归一化可以解决特征不在同一尺度的问题，H的病态问题可以用类似的方法解决吗？  

回顾公式4.9，代价函数的二阶泰勒级数展开预测梯度下降中的$-\epsilon g$步骤会增加  
$$
\begin{aligned}
    \frac{1}{2} \epsilon^2 g^\top Hg - \epsilon g^\top g && (8.10)
\end{aligned}
$$

到代价中。  
> **[success]**  
$$
\begin{aligned}
f(x^0-\epsilon g) \approx f(x^0) - \epsilon g^\top g + \frac{1}{2} \epsilon^2 g^\top Hg  && (4.9)
\end{aligned}
$$

> 公式4.9是f(x)在$x^0$处泰勒展开的二阶近似。  
$f(x^0-\epsilon g)$为x0在负梯度方向移到一个步长得到的值。  
由于是向负梯度的方向移到，理论上$f(x^0-\epsilon g) < f(x^0)$  
实际上$f(x^0-\epsilon g) - f(x^0) = \frac{1}{2} \epsilon^2 g^\top Hg - \epsilon g^\top g$，取决于公式8.10  

当$\frac{1}{2} \epsilon^2 g^\top Hg$超过$\epsilon g^\top g$时，梯度的病态会成为问题。
我们可以通过监测平方梯度范数$g^\top g$和$g^\top Hg$，来判断病态是否不利于神经网络训练任务。
在很多情况中，梯度范数不会在训练过程中显著缩小，但是$g^\top Hg$的增长会超过一个数量级。
其结果是尽管梯度很强，学习会变得非常缓慢，因为学习率必须收缩以弥补更强的曲率。  
> **[warning]**  梯度范数是$g^\top g$和$g^\top Hg$各自的梯度范数？还是公式8.10的梯度范数？与$g^\top Hg$是什么关系？    

如\fig?所示，成功训练的神经网络中，梯度显著增加。

{% reveal %}
{% raw %}
\begin{figure}[!htb]
\ifOpenSource
\centerline{\includegraphics{figure.pdf}}
\else
\centerline{\includegraphics{Chapter8/figures/grad_norm_increases_color}}
\fi
\caption{梯度下降通常不会到达任何类型的临界点。
此示例中，在用于对象检测的卷积网络的整个训练期间， 梯度范数持续增加。
\emph{(左)}各个梯度计算的范数如何随时间分布的散点图。
为了方便作图，每轮仅绘制一个梯度范数。 
我们将所有梯度范数的移动平均绘制为实曲线。
梯度范数明显随时间增加，而不是如我们所期望的那样随训练过程收敛到临界点而减小。
\emph{(右)}尽管梯度递增，训练过程却相当成功。 
验证集上的分类误差可以降低到较低水平。
}
\end{figure}
{% endraw %}
{% endreveal %}

尽管病态还存在于除了神经网络训练的其他情况中，有些适用于其他情况的解决病态的技术并不适用于神经网络。
例如，牛顿法在解决带有病态条件的\,Hessian\,矩阵的凸优化问题时，是一个非常优秀的工具，  
> **[success]**  问：牛顿法为什么能解决H的病态问题？  
答：[牛顿法 VS 梯度下降法]https://windmissing.github.io/mathematics_basic_for_ML/NumericalComputation/Newton.html#%E7%89%9B%E9%A1%BF%E6%B3%95-vs-%E6%A2%AF%E5%BA%A6%E4%B8%8B%E9%99%8D%E6%B3%95)  
可以看出，牛顿法不需要指定学习率，而是根据$H^{-1}$计算出来一组值来作为学习率。   
牛顿法的学习率是根据H算出来的，是适配于H矩阵的。即使是病态的H，而计算出合适这个H的学习率。  
因此能解决H的病态问题。  
[DL中解决H矩阵病态问题的方法](https://windmissing.github.io/Bible-DeepLearning/Chapter8/3BasicAlgorithms/2Momentum.html)  


但是我们将会在以下小节中说明牛顿法运用到神经网络时需要很大的改动。