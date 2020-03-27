# RMSProp 算法

RMSProp算法修改AdaGrad以在非凸设定下效果更好，**改变梯度积累为指数加权的移动平均**。
AdaGrad旨在应用于凸问题时快速收敛。
当应用于非凸函数训练神经网络时，学习轨迹可能穿过了很多不同的结构，最终到达一个局部是凸的碗状的区域。  
> **[warning]** 想像不出来？

AdaGrad根据平方梯度的整个历史收缩学习率，可能使得学习率在达到这样的凸结构前就变得太小了。  
> **[warning]** 整个历史的收缩学习率为什么会造成这样的效果。  

RMSProp使用**指数衰减平均以丢弃遥远过去的历史，使其能够在找到碗状凸结构后快速收敛**，  
> **[warning]** 指数衰减平均是什么意思？算法中没有体现出指数。  

它就像一个初始化于该碗状结构的AdaGrad算法实例。  
> **[warning]** [?]初始化于该碗状结构的AdaGrad算法实例?

RMSProp的标准形式如算法8.5所示，结合Nesterov动量的形式如算法8.6所示。
相比于AdaGrad，使用移动平均引入了一个新的超参数$\rho$，用来控制移动平均的长度范围。

> **[success]** 这一节对比了三种算法  
$$
\begin{aligned}
&& Adagrad && RMSProp && Nesterov + RMSProp \\
\text{临时更新} && && && \tilde \theta \leftarrow \theta + \alpha v\\
\text{计算梯度} && g \leftarrow \frac{1}{m} \nabla_{\theta} \sum_i L(f(x^{(i)};\theta),y^{(i)}) && g \leftarrow \frac{1}{m} \nabla_{\theta} \sum_i L(f(x^{(i)};\theta),y^{(i)}) && g \leftarrow \frac{1}{m} \nabla_{\tilde \theta} \sum_i L(f(x^{(i)};\tilde \theta),y^{(i)})  \\
\text{累积梯度} && r \leftarrow r + g \odot g && r \leftarrow \rho r + (1-\rho) g \odot g && r \leftarrow \rho r + (1-\rho) g \odot g \\
\text{参数更新} && \Delta \theta \leftarrow - \frac{\epsilon}{\delta+ \sqrt{r}} \odot g && \Delta \theta \leftarrow - \frac{\epsilon}{\sqrt {\delta+ r}} \odot g && v \leftarrow \alpha v -\frac{\epsilon}{\sqrt{r}} \odot g \\
\text{应用更新} && \theta \leftarrow \theta + \Delta \theta && \theta \leftarrow \theta + \Delta \theta && \theta \leftarrow \theta + v
\end{aligned}
$$

# RMSProp 效果

经验上，RMSProp已被证明是一种**有效且实用的深度神经网络优化算法**。
目前它是深度学习从业者经常采用的优化方法之一。

> **[success]** 
Adagrad：在某些深度学习模型上效果不错  
RMSProp：有效且实用的深度神经网络优化算法  
Nesterov + RMSProp：RMSProp用于深度神经网络。而Nesterov对SGD没有效果，这两个合在一起想干吗？

{% reveal %}
```
{% raw %}
\begin{algorithm}[ht]
\caption{RMSProp算法}
\label{alg:rms_prop}
\begin{algorithmic}
\REQUIRE 全局学习率 $\epsilon$，衰减速率$\rho$
\REQUIRE  初始参数$\Vtheta$
\REQUIRE 小常数$\delta$，通常设为$10^{-6}$（用于被小数除时的数值稳定）
\STATE 初始化累积变量 $\Vr = 0$
\WHILE{没有达到停止准则}
    \STATE 从训练集中采包含$m$个样本$\{ \Vx^{(1)},\dots, \Vx^{(m)}\}$ 的小批量，对应目标为$\Vy^{(i)}$。
    \STATE 计算梯度：$\Vg \leftarrow  
         \frac{1}{m} \nabla_{\Vtheta} \sum_i L(f(\Vx^{(i)};\Vtheta),\Vy^{(i)})$ 
    \STATE 累积平方梯度：$\Vr \leftarrow \rho
    \Vr + (1-\rho) \Vg \odot \Vg$
    \STATE 计算参数更新：$\Delta \Vtheta =
    -\frac{\epsilon}{\sqrt{\delta + \Vr}} \odot \Vg$  \ \  ($\frac{1}{\sqrt{\delta + \Vr}}$ 逐元素应用)
    \STATE 应用更新：$\Vtheta \leftarrow \Vtheta + \Delta \Vtheta$
\ENDWHILE
\end{algorithmic}
\end{algorithm}

\begin{algorithm}[ht]
\caption{使用Nesterov\,动量的RMSProp算法}
\label{alg:rms_nesterov}
\begin{algorithmic}
\REQUIRE 全局学习率 $\epsilon$，衰减速率$\rho$， 动量系数$\alpha$
\REQUIRE 初始参数$\Vtheta$，初始参数$\Vv$
\STATE 初始化累积变量 $\Vr = 0$
\WHILE{没有达到停止准则} % NOTE: do not capitalize the condition
    \STATE 从训练集中采包含$m$个样本$\{ \Vx^{(1)},\dots, \Vx^{(m)}\}$ 的小批量，对应目标为$\Vy^{(i)}$。
    \STATE 计算临时更新：$\tilde{\Vtheta} \leftarrow \Vtheta + \alpha \Vv$
    \STATE 计算梯度：$\Vg \leftarrow  
         \frac{1}{m} \nabla_{\tilde{\Vtheta}} \sum_i L(f(\Vx^{(i)};\tilde{\Vtheta}),\Vy^{(i)})$ 
    \STATE  累积梯度：$\Vr \leftarrow \rho
    \Vr + (1-\rho) \Vg \odot \Vg$
    \STATE  计算速度更新：$\Vv \leftarrow \alpha \Vv
    -\frac{\epsilon}{\sqrt{\Vr}} \odot \Vg$ \ \  ($\frac{1}{\sqrt{\Vr}}$ 逐元素应用)
    \STATE 应用更新：$\Vtheta \leftarrow \Vtheta + \Vv$
\ENDWHILE
\end{algorithmic}
\end{algorithm}
{% endraw %}
```
{% endreveal %}