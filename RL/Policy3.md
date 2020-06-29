# On policy VS Off policy

On policy:  agent在与环境互动的过程中学习    
Off policy： agent看别人与环境互动的过程来学习。  

On polocy的缺点：  
每一次迭代都要重新sample data。要花很多时间在sample data上。  

改进方法：  
使用$\pi_{\theta'}$的sample来训练$\theta$。  
$\theta'$是固定的，因此同一组sample data可以用多次。  

# Importance Sampling

已知：  
$$
E_{x\sim p}[f(x)] \approx \frac{1}{N}\sum f(x)
$$

公式要求的x是从P中sample出的data，但我们只有从Q sample出的data。如何基于Q中的Data计算以上公式？经过公式推导得：  
$$
E_{x\sim p}[f(x)] = E_{x\sim q}[f(x)\frac{p(x)}{q(x)}]
$$

这样就转成了基于Q的data的计算公式。这个技巧叫做Importance Sampling，不只上用于这里，在很多地方都有应用。  
公式中的$\frac{p(x)}{q(x)}$称为Important Weight。  

以上等式是用积分的形式推导出来的，实际计算的时候是使用sample data来计算的，所以计算结果只是近似。如果sample的点数太少，可能结果差别会比较大。只有sample的数据足够大，才能得到近似的结果。  
![](/assets/images/Chapter7/64.png)    

# 回到Reinforce Learning

offline policy是指用$\theta'$与环境做互动来更新$\theta$。  
$x\sim p$相当于$\tau \sim p_\theta(\tau)$，$x \sim q$相当于$\tau\sim p_{\theta'}(\tau)$，得：  
$$
\begin{aligned}
\nabla \bar R_\theta &=& E_{\tau\sim p_{\theta}(\tau)}\left[R(\tau)\nabla \log p_{\theta(\tau)}\right]   \\
&=& E_{\tau\sim p_{\theta'}(\tau)}\left[\frac{p_{\theta}(\tau)}{p_{\theta'}(\tau)}R(\tau)\nabla \log p_{\theta(\tau)}\right]
\end{aligned}
$$

再结合上面提到的“公式改进”，得：  
$$
\begin{aligned}
\nabla \bar R_\theta &=& E_{(s_t,a_t)\sim \pi_{\theta'}}\left[\frac{p_{\theta}(s_t,a_t)}{p_{\theta'}(s_t,a_t)}A^{\theta'}(s_t,a_t)\nabla \log p_{\theta(a_t^n|s_t^n)}\right]
\end{aligned}
$$

反推出object function：  
$$
J^{\theta'}(\theta) = E_{(s_t,a_t)\sim \pi_{\theta'}}\left[\frac{p_{\theta}(s_t,a_t)}{p_{\theta'}(s_t,a_t)}A^{\theta'}(s_t,a_t)\right]
$$

