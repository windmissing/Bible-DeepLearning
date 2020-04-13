Adam是另一种学习率自适应的优化算法，如算法8.7所示。
“Adam”这个名字派生自短语“**adaptive moments**”。
在前述算法背景下，它也许最好被看作**结合了RMSProp和动量的具有一些重要区别的变种**。   
> **[success]**  
（1）将动量应用于缩放后的梯度。  
Adam = RMSProp + Momentum。  
（2）偏置修正，修正从原点初始化的一阶矩（动量项）和（非中心的）二阶矩的估计。  
$$
\begin{aligned}
\hat{s} \leftarrow \frac{s}{1-\rho_1^t} \\  
\hat{r} \leftarrow \frac{r}{1-\rho_2^t}
\end{aligned}
$$

　　  
> **[warning]** 没有看出怎么结合动量变种？  

# Adam算法原理  

> **[warning]** 原理完全没看懂  

首先，在Adam中，动量直接并入了梯度一阶矩（指数加权）的估计。  
> **[warning]** 什么是“梯度一阶矩（指数加权）”？从公式上看没有看到跟指数有什么关系。  

将动量加入RMSProp最直观的方法是将动量应用于缩放后的梯度。  
> **[warning]** 算法过程中没有看到计算v这一步？  

结合缩放的动量使用没有明确的理论动机。
其次，Adam包括偏置修正，修正从原点初始化的一阶矩（动量项）和（非中心的）二阶矩的估计（\algref{alg:adam}）。  
> **[warning]** 一阶矩?二阶矩?  

RMSProp也采用了（非中心的）二阶矩估计，然而缺失了修正因子。  
> **[warning]** 修改因子？  

因此，不像Adam，RMSProp二阶矩估计可能在训练初期有很高的偏置。  

# 效果

Adam通常被认为**对超参数的选择相当鲁棒**，尽管学习率有时需要改为与建议的默认值不同的值。

> **[success]**  
计算梯度：$g \leftarrow \frac{1}{m} \nabla_{\theta} \sum_i L(f(x^{(i)};\theta),y^{(i)})$   
$t \leftarrow t + 1$  
更新有偏一阶矩估计： $s \leftarrow \rho_1 s + (1-\rho_1) g$  
更新有偏二阶矩估计：$r \leftarrow \rho_2 r + (1-\rho_2) g \odot g$  
修正一阶矩的偏差：$\hat{s} \leftarrow \frac{s}{1-\rho_1^t}$  
修正二阶矩的偏差：$\hat{r} \leftarrow \frac{r}{1-\rho_2^t}$  
计算更新：$\Delta \theta = - \epsilon \frac{\hat{s}}{\sqrt{\hat{r}} + \delta}$  
应用更新：$\theta \leftarrow \theta + \Delta \theta$  

{% reveal %}
```
{% raw %}
\begin{algorithm}[ht]
\caption{Adam算法}
\label{alg:adam}
\begin{algorithmic}
\REQUIRE 步长 $\epsilon$ （建议默认为： $0.001$）
\REQUIRE 矩估计的指数衰减速率， $\rho_1$ 和 $\rho_2$ 在区间 $[0, 1)$内。
（建议默认为：分别为$0.9$ 和 $0.999$）
\REQUIRE 用于数值稳定的小常数 $\delta$  （建议默认为： $10^{-8}$）
\REQUIRE 初始参数 $\theta$
\STATE 初始化一阶和二阶矩变量 $s = 0 $, $r = 0$
\STATE 初始化时间步 $t=0$ 
\WHILE{没有达到停止准则}
    \STATE 从训练集中采包含$m$个样本$\{ x^{(1)},\cdots, x^{(m)}\}$ 的小批量，对应目标为$y^{(i)}$。
    \STATE 计算梯度：$g \leftarrow \frac{1}{m} \nabla_{\theta} \sum_i L(f(x^{(i)};\theta),y^{(i)})$ 
    \STATE $t \leftarrow t + 1$
    \STATE 更新有偏一阶矩估计： $s \leftarrow \rho_1 s + (1-\rho_1) g$
    \STATE 更新有偏二阶矩估计：$r \leftarrow \rho_2 r + (1-\rho_2) g \odot g$
    \STATE 修正一阶矩的偏差：$\hat{s} \leftarrow \frac{s}{1-\rho_1^t}$
    \STATE 修正二阶矩的偏差：$\hat{r} \leftarrow \frac{r}{1-\rho_2^t}$
    \STATE 计算更新：$\Delta \theta = - \epsilon \frac{\hat{s}}{\sqrt{\hat{r}} + \delta}$ \ \  （逐元素应用操作）
    \STATE 应用更新：$\theta \leftarrow \theta + \Delta \theta$
\ENDWHILE
\end{algorithmic}
\end{algorithm}
{% endraw %}
```
{% endreveal %}