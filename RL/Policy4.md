使用off policy代替on policy的前提是，$p_\theta$和$p_{\theta'}$不能差太多，否则效果不好。  
问：如何保证“$p_\theta$和$p_{\theta'}$不能差太多”？  
答：近端优化策略 PPO/TRPO  

# PPO 正则化

对$J^{\theta'}(\theta)$增加一个正则化项：  
$$
J^{\theta'}_{PPO}(\theta) = J^{\theta'}(\theta) - \beta KL(\theta, \theta')
$$

# TRPO 带限制最优化

$$
\begin{aligned}
J^{\theta'}_{TRPO}(\theta) = J^{\theta'}(\theta)   \\
s.t. && KL(\theta, \theta') < S
\end{aligned}
$$

# $KL(\theta, \theta')$

$KL(\theta, \theta')$代表参数$\theta$和$\theta'$的距离。但不是它们的欧式距离或者KL散度。  
而是这两个参数对某个state产生的action的分布的KL散度。  

# PPO算法过程  

1. 初始化$\theta^0$  
2. 使用$\theta^k$与环境互动，收集$\{s_t, a_t\}$，计算$A^\theta{s_t, a_t}$  
3. 更新$\theta$，优化以下目标，可进行多次迭代：  
$$
\begin{aligned}
J^{\theta^k}_{PPO}(\theta) = J^{\theta^k}(\theta) - \beta KL(\theta, \theta^k)  \\
J^{\theta^k}(\theta) \approx \sum_{s_t,a_t} \frac{p_\theta(a_t|s_t)}{p^k_\theta(a_t|s_t)}A^{\theta^k}(s_t, a_t)
\end{aligned}
$$
4. 多次迭代后调整$\beta$：  
if $KL(\theta, \theta^k)$ > threshold_max, then $\beta \uparrow$  
if $KL(\theta, \theta^k)$ < threshold_min, then $\beta \downarrow$  

# PPO2算法过程

1. 同上  
2. 同上  
3. 目标函数替换为：  
$$
\begin{aligned}
J^{\theta^k}_{PPO2}(\theta) &=& \sum_{s_t,a_t} \min 
\left(PA, \text{clip} \left(P, 1-\epsilon, 1+\epsilon\right)A \right)   \\
P &=& \frac{p_\theta(a_t|s_t)}{p^k_\theta(a_t|s_t)}  \\
A &=& A^{\theta^k}(s_t, a_t)
\end{aligned}
$$

P、A是我为了简化表达自己加的变量。这个公式的直观解释如下：  
![](/assets/images/Chapter7/65.png)    
图中绿色虚线代表P，蓝色虚线代表$\text{clip} (P, 1-\epsilon, 1+\epsilon)$  
结合上后面的A：  
当A>0时，实际生效的是P是左图红色部分。当A<0时，实际生效的是右图红色部分。这是min的效果。  
假设A>0，  
希望$p_\theta(a_t|s_t)$越大越好 ---> 沿红线上移   
不希望P太大 ---> 移到$1+\epsilon$就没有benefit了，就不会动了。  
A<0情况同理。  