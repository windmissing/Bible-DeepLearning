# Policy Based算法 --- learning an Actor

## 定义

定义actor为function $\pi$，则：  
$$
\text{Action} = \pi(\text{State})
$$

其中：  
$\pi$：用NN训练出来的一个模型  
State：模型的输入，为向量或矩阵。例如游戏界面的像素值。  
Action：模型的输出，由softmax生成的一个分布。  

定义：  
s：state向量  
a：action向量  
r：reward向量  
T：向量长度，1 episode = T轮迭代  
$R_\theta$：总reward，为episode中所有迭代的reward之和。  

对于同一个$\theta$，一个episode得到的$R_\theta$也不同，因此定义期望值：  
$$
\bar R_\theta = E[R_\theta]
$$

目标是最大化$\bar R_\theta$，不是$R_\theta$，也不是$r_t$。  

## 推导期望的公式

定义：  
$$
\begin{aligned}
\tau = {s_1, a_1, r_1, \cdots, s_T, a_T, r_T}   && \text{一局游戏的全过程}\\
R(\tau) = \sum_{t=1}^T r_n
\end{aligned}
$$

则：  
$$
\bar R_{\theta} \sum_\tau R(\tau) P(\tau|\theta)
$$

$\sum_\tau$代表穷举所有的游戏经过。事实上这是不可能的。  
解决方法：用$\pi_\theta$玩N局，采样出N个$\tau$，用这N个$\tau$代替所有的$\tau$。  
因此有：  
$$
\sum_\tau P(\tau|\theta) = \frac{1}{N}
$$

## 梯度上升法最大化期望

根据梯度上升法：  
$$
\theta^t = \theta^{t-1} + \eta\nabla\bar R_{\theta^{t-1}}
$$

这里关键是求出$\bar R_{\theta}$的偏导：  
![](/assets/images/Chapter7/23.png)    
![](/assets/images/Chapter7/24.png)    
![](/assets/images/Chapter7/25.png)    
![](/assets/images/Chapter7/26.png)    

## 偏导公式的直观解释

当$R(\tau^n)>0$时，就是增加这一轮中的Action的概率，即$P(a_t^n|s_t^n,\theta)$的概率。  
当$R(\tau^n)<0$时，就是减少这一轮中的Action的概率。  

1. 公式中基于$R(\tau^n)$调整这个episode中所有action的概率，而不是$r_t$调整$a_t$的概率，是为了解决难点2。  
2. 为什么是$\nabla\log p(a_t^n|s_t^n, \theta)$，而不是$\nabla p(a_t^n|s_t^n, \theta)$？  
答：  
$$
\nabla\log p(a_t^n | s_t^n, \theta) = \frac{\nabla p(a_t^n|s_t^n, \theta)}{p(a_t^n|s_t^n, \theta)}
$$

$\nabla\log p(a_t^n|s_t^n, \theta)$和$\nabla p(a_t^n|s_t^n, \theta)$差别在于分母$p(a_t^n|s_t^n, \theta)$，这个分母相当于归一化的过程。  

