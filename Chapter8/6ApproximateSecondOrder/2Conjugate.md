共轭梯度法是一种通过迭代下降的共轭方向以有效避免\,Hessian\,矩阵求逆计算的方法。  
> **[warning]** 共轭方向？  

这种方法的灵感来自于对最速下降方法弱点的仔细研究（详细信息请查看\sec?），其中线搜索迭代地用于与梯度相关的方向上。  
> **[warning]**  
[?] 最速下降法是什么？4.3节说最速下降法就是梯度下降法？  
弱点见下文。  
[?] 线搜索？

\fig?说明了该方法在二次碗型目标中如何表现的，是一个相当低效的来回往复，锯齿形模式。
这是因为每一个由梯度给定的线搜索方向，都保证正交于上一个线搜索方向。  
> **[warning]** 正交于上一个线搜索方向？  

{% reveal %}
\begin{figure}[!htb]
\ifOpenSource
\centerline{\includegraphics{figure.pdf}}
\else
\centerline{\includegraphics{Chapter8/figures/steepest_descent_quadratic_color}}
\fi
\caption{将最速下降法应用于二次代价表面。
在每个步骤，最速下降法沿着由初始点处的梯度定义的线跳到最低代价的点。
这解决了\fig?中使用固定学习率所遇到的一些问题，但即使使用最佳步长，算法仍然朝最优方向曲折前进。
根据定义，在沿着给定方向的目标最小值处，最终点处的梯度与该方向正交。
}
\end{figure}
{% endreveal %}

假设上一个搜索方向是$d_{t-1}$。   
在极小值处，也就是（上一次）线搜索终止的地方，方向$d_{t-1}$处的方向导数为零：$\nabla_{\theta} J(\theta) \cdot d_{t-1} = 0$。
因为该点的梯度定义了当前的搜索方向，$d_t = \nabla_{\theta} J(\theta)$将不会贡献于方向$d_{t-1}$。
因此方向$d_t$正交于$d_{t-1}$。
最速下降多次迭代中，方向$d_{t-1}$和$d_t$之间的关系如\fig?所示。
如图展示的，下降正交方向的选择不会保持前一搜索方向上的最小值。
这产生了锯齿形的过程。
在当前梯度方向下降到极小值，我们必须重新最小化之前梯度方向上的目标。
因此，**通过遵循每次线搜索结束时的梯度，我们在某种程度上撤销了在之前线搜索的方向上取得的进展。**  
> **[success]** 这就是最速下降法的弱点  

共轭梯度法试图解决这个问题。

在共轭梯度法中，我们寻求一个和先前线搜索方向共轭的搜索方向，即它不会撤销该方向上的进展。
在训练迭代$t$时，下一步的搜索方向$d_t$的形式如下：  
$$
\begin{aligned}
    d_t = \nabla_{\theta} J(\theta) + \beta_t d_{t-1},
\end{aligned}
$$

其中，系数$\beta_t$的大小控制我们应沿方向$d_{t-1}$加回多少到当前搜索方向上。

> **[success]**  
公式中第一项代表当前方向，第二项代表之前方向。通过控制$\beta$实现共轭。  

如果$d_t^\top H d_{t-1} = 0$，其中$H$是\,Hessian\,矩阵，则两个方向$d_t$和$d_{t-1}$被称为共轭的。  
> **[warning]** [?]共轭矩阵？  

适应共轭的直接方法会涉及到$H$特征向量的计算以选择$\beta_t$。  
> **[warning]** $\beta_t$和H矩阵是什么关系？  

这将无法满足我们的开发目标：寻找在大问题比牛顿法计算更加可行的方法。
我们能否不进行这些计算而得到共轭方向？
幸运的是这个问题的答案是肯定的。

两种用于计算$\beta_t$的流行方法是：   
> **[warning]** 这两个公式都看不懂  

1. Fletcher-Reeves:  
$$
\begin{aligned}
    \beta_t = \frac{ \nabla_{\theta} J(\theta_t)^\top \nabla_{\theta} J(\theta_t) }
{ \nabla_{\theta} J(\theta_{t-1})^\top \nabla_{\theta} J(\theta_{t-1}) }
\end{aligned}
$$

2. Polak-Ribi\`{e}re:  
$$
\begin{aligned}
    \beta_t = \frac{ (\nabla_{\theta} J(\theta_t) - \nabla_{\theta} J(\theta_{t-1}))^\top \nabla_{\theta} J(\theta_t) }
{ \nabla_{\theta} J(\theta_{t-1})^\top \nabla_{\theta} J(\theta_{t-1}) }
\end{aligned}
\end{enumerate}
$$

对于二次曲面而言，共轭方向确保梯度沿着前一方向幅度不会变大。
因此，我们在前一方向上仍然处于极小值。  
> **[warning]** 为什么梯度方向不变就仍是极小值？  

其结果是，在$k$-维参数空间中，共轭梯度法只需要至多$k$次线搜索就能达到极小值。
共轭梯度法如\alg?所示。

{% reveal %}
{% raw %}
\begin{algorithm}[ht]
\caption{共轭梯度法}
\begin{algorithmic}
\REQUIRE 初始参数 $\theta_{0}$
\REQUIRE 包含$m$个样本的训练集
\STATE 初始化 $rho_{0} = 0$
\STATE 初始化 $g_0 = 0$
\STATE 初始化 $t = 1$
\WHILE{没有达到停止准则}
    \STATE 初始化梯度 ${g}_{t} = 0$
    \STATE 计算梯度：$g_{t} \leftarrow
         \frac{1}{m}\nabla_{\theta} \sum_i L(f(x^{(i)};\theta),y^{(i)})$ 
    \STATE 计算 $\beta_{t} = \frac{(g_{t}-g_{t-1})^\top g_{t}}{g_{t-1}^\top g_{t-1}}$  (Polak-Ribi\`{e}re)
    \STATE (非线性共轭梯度法：视情况可重置$\beta_{t}$为零，
           例如  $t$是常数$k$的倍数时，如 $k=5$)
    \STATE 计算搜索方向： $rho_{t} = -g_{t} + \beta_{t} rho_{t-1}$ 
    \STATE 执行线搜索寻找：$\epsilon^{*} = \arg\!\min_{\epsilon}
    \frac{1}{m} \sum_{i=1}^{m}L(f(x^{(i)};\theta_t + \epsilon rho_t),y^{(i)})$ 
    \STATE （对于真正二次的代价函数，存在$\epsilon^*$的解析解，而无需显式地搜索）
    \STATE 应用更新：$\theta_{t+1} = \theta_{t}+ \epsilon^{*} rho_{t}$
    \STATE $t \leftarrow t + 1$
\ENDWHILE
\end{algorithmic}
\end{algorithm}
{% endraw %}
{% endreveal %}

\paragraph{非线性共轭梯度法：}
目前，我们已经讨论了用于二次目标函数的共轭梯度法。
当然，本章我们主要关注于探索训练神经网络和其他相关深度学习模型的优化方法，其对应的目标函数比二次函数复杂得多。
> **[warning]** 是否二次对共轭梯度算法有什么影响？  

或许令人惊讶，共轭梯度法在这种情况下仍然是适用的，尽管需要作一些修改。
没有目标是二次的保证，共轭方向也不再保证在以前方向上的目标仍是极小值。
其结果是，\textbf{非线性共轭梯度法}\,算法会包括一些偶尔的重设，共轭梯度法沿未修改的梯度重启线搜索。  
> **[warning]** 非线性共轭梯度法？

实践者报告在实践中使用非线性共轭梯度法训练神经网络是合理的，尽管在开始非线性共轭梯度法前使用随机梯度下降迭代若干步来初始化效果更好。
另外，尽管（非线性）共轭梯度法传统上作为批方法，小批量版本已经成功用于训练神经网络~{cite?}。
针对神经网路的共轭梯度法应用早已被提出，例如缩放的共轭梯度法{cite?}。  
> **[warning]** 缩放的共轭梯度法？  














