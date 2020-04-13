# 算法

受Nesterov加速梯度算法启发，sutskever提出了动量算法的一个变种。  
> **[warning]** Nesterov加速梯度算法?  

这种情况的更新规则如下：  

$$
\begin{aligned}
    v &\leftarrow \alpha v - \epsilon \nabla_{\theta} \left[
    \frac{1}{m} \sum_{i=1}^m L\big( f(x^{(i)}; \theta + \alpha v), y^{(i)} \big)
 \right], \\
    \theta &\leftarrow \theta + v
\end{aligned}
$$

其中参数$\alpha$和$\epsilon$发挥了和标准动量方法中类似的作用。 
Nesterov 动量和标准动量之间的区别体现在梯度计算上。
Nesterov 动量中，梯度计算在施加当前速度之后。
因此，Nesterov 动量可以解释为往标准动量方法中添加了一个**校正因子**。  
> **[warning]** 怎样理解把“这一步临时更新”看作是添加一个校正因子？  

完整的\,Nesterov 动量算法如算法8.3所示。

> **[success]**  
**临时更新：$\tilde \theta \leftarrow \theta + \alpha v$**
计算梯度：$g \leftarrow \frac{1}{m} \nabla_{\tilde \theta} \sum_i L(f(x^{(i)};\tilde \theta),y^{(i)})$  
更新速度：$v \leftarrow \alpha v - \epsilon g$    
更新参数：$\theta \leftarrow \theta + v$


{% reveal %}
```
{% raw %}
\begin{algorithm}[ht]
\caption{使用\,Nesterov 动量的随机梯度下降（SGD）}
\label{alg:nesterov}
\begin{algorithmic}
\REQUIRE  学习率 $\epsilon$， 动量参数 $\alpha$
\REQUIRE 初始参数 $\theta$，初始速度 $v$
\WHILE{没有达到停止准则}
    \STATE 从训练集中采包含$m$个样本$\{ x^{(1)},\cdots, x^{(m)}\}$ 的小批量，对应目标为$y^{(i)}$。
    \STATE 应用临时更新： $\tilde{\theta} \leftarrow \theta  + \alpha v$
         \STATE 计算梯度（在临时点）：$g \leftarrow 
         \frac{1}{m} \nabla_{\tilde{\theta}} \sum_i L(f(x^{(i)};\tilde{\theta}),y^{(i)})$
    \STATE 计算速度更新：$v \leftarrow \alpha v - 
    \epsilon g$
    \STATE 应用更新：$\theta \leftarrow \theta + v$ 
\ENDWHILE
\end{algorithmic}
\end{algorithm}
{% endraw %}
```
{% endreveal %}

# 效果

在凸批量梯度的情况下，Nesterov 动量将额外误差收敛率从$O(1/k)$（$k$步后）改进到$O(1/k^2)$，如Nesterov83b所示。
可惜，在随机梯度的情况下，Nesterov 动量没有改进收敛率。