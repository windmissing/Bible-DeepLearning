{AdaGrad}算法，如算法8.4所示，独立地适应所有模型参数的学习率，按照每个参数的梯度历史值的平方和的平方根成反比缩放每个参数。  
> **[success]**  
r是向量、$\Delta \theta$也是向量，可见每一个参数都有一个单独的调整。  

具有损失最大偏导的参数相应地有一个快速下降的学习率，而具有小偏导的参数在学习率上有相对较小的下降。  
> **[warning]** 根据算法中的公式，偏导大->g大->r大->$\frac{\epsilon}{\delta+ \sqrt{r}}$小->$\Delta \theta$->下降慢。与上面的结论相反。  

净效果是在参数空间中更为平缓的倾斜方向会取得更大的进步。  
> **[warning]** 平缓的方向应该对应偏导较小，但是进步却更大，这个结论与上一句相反。  

在凸优化背景中，AdaGrad 算法具有一些令人满意的理论性质。
然而，经验上已经发现，对于训练深度神经网络模型而言，**从训练开始时**积累梯度平方会导致有效学习率过早和过量的减小。
AdaGrad在某些深度学习模型上效果不错，但不是全部。  
> **[warning]** 适用用于怎样的模型？  



> **[success]**  
计算梯度： $g \leftarrow \frac{1}{m} \nabla_{\theta} \sum_i L(f(x^{(i)};\theta),y^{(i)})$   
累积平方梯度：$r \leftarrow r + g \odot g$    
计算更新：$\Delta \theta \leftarrow - \frac{\epsilon}{\delta+ \sqrt{r}} \odot g$  
应用更新：$\theta \leftarrow \theta + \Delta \theta$  


{% reveal %}
```
{% raw %}
\begin{algorithm}[ht]
\caption{AdaGrad算法}
\label{alg:ada_grad}
\begin{algorithmic}
\REQUIRE 全局学习率 $\epsilon$
\REQUIRE 初始参数$\Vtheta$
\REQUIRE 小常数$\delta$，为了数值稳定大约设为$10^{-7}$
\STATE 初始化梯度累积变量$\Vr = 0$
\WHILE{没有达到停止准则}
    \STATE 从训练集中采包含$m$个样本$\{ \Vx^{(1)},\dots, \Vx^{(m)}\}$ 的小批量，对应目标为$\Vy^{(i)}$。
    \STATE 计算梯度： $\Vg \leftarrow  
         \frac{1}{m} \nabla_{\Vtheta} \sum_i L(f(\Vx^{(i)};\Vtheta),\Vy^{(i)})$ 
    \STATE 累积平方梯度：$\Vr \leftarrow \Vr + \Vg \odot \Vg$
    \STATE 计算更新：$\Delta \Vtheta \leftarrow -
    \frac{\epsilon}{\delta+ \sqrt{\Vr}} \odot\Vg$  \ \  （逐元素地应用除和求平方根）
    \STATE 应用更新：$\Vtheta \leftarrow \Vtheta + \Delta \Vtheta$
\ENDWHILE
\end{algorithmic}
\end{algorithm}
{% endraw %}
```
{% endreveal %}