
-

在优化凸函数时，会遇到一些挑战。
这其中最突出的是Hessian矩阵$H$的病态。
这是数值优化、凸优化或其他形式的优化中普遍存在的问题，更多细节请回顾\sec?。

病态问题一般被认为存在于神经网络训练过程中。
**病态体现在随机梯度下降会"卡"在某些地方，此时即使很小的更新步长也会增加代价函数。**  
> **[warning]**  到底什么是病态？  
4.2节说：病态是指输入的微小差异导致输出的巨大变化。  
8.2.1节说：病态是指即使很小的更新步长也会增加代价函数。  


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
> **[warning]**  牛顿法为什么能解决病态问题？  

但是我们将会在以下小节中说明牛顿法运用到神经网络时需要很大的改动。