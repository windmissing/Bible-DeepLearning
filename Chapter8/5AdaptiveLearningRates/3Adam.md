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

# Adam算法原理  

首先，在Adam中，动量直接并入了梯度一阶矩（指数加权）的估计。  
将动量加入RMSProp最直观的方法是将动量应用于缩放后的梯度。  
> **[success]**  
动量算法对导数（一阶）做了指数衰减平均。  
RMSProp对导数的平方（二阶）做了指数衰减平均。  
将这两种方法的结合即同时计算导数和导数平方的指数衰减平均。   

结合缩放的动量使用没有明确的理论动机。
其次，Adam包括偏置修正，修正从原点初始化的一阶矩（动量项）和（非中心的）二阶矩的估计（\algref{alg:adam}）。  
RMSProp也采用了（非中心的）二阶矩估计，然而缺失了修正因子。  
> **[success]** 使用指数衰减平均需要做[偏差修正](https://windmissing.github.io/mathematics_basic_for_ML/Mathematics/ExponentialDecay.html)  
RMSProp算法计算了梯度平方（二阶）的指数衰减平均，但没有对这个平均做修正。  
Adam算法计算了梯度（一阶）的指数衰减平均和梯度平方（二阶）的指数衰减平均，并对这两个平均都做了修正。  

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
**Ag建议参数：**：  
$\rho_1$ = 0.9  
$\rho_2$ = 0.999  
$\delta$不重要。  

```
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
```


