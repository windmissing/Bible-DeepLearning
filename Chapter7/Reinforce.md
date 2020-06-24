# background

## 深度学习一次迭代的三个步骤  

1. 环境 -> 机器：state  
2. 机器 -> 环境：action  
3. 环境 -> 机器：reward  

定义：
一轮迭代 = state -> action -> reward。  如果没有反馈，reward = 0。  
一episode = 一局游戏，有赢/输结果的。  
目标：maximize the expected cumulative reward per spisode。  

## 监督学习 VS 强化学习

监督学习：从数据(State, Action)学习，学习的好坏取决于数据(State, Action)的好坏，因此需要大量数据。  
强化学习：根据自己的(State, Action)经验学习，因此需要大量的经验。  

## 强化学习的难点  

1. reward delay  
2. 有些Action没有reward，甚至可能有牺牲。但它对帮助得到reward有重要贡献。  
3. 需要Machine探索未尝试过的行为。  

## 算法分类

1. policy based算法 --- 学actor  
2. value based算法 --- 学critic  
3. policy + value 算法 --- A3C算法  

Alpha GO = polocy based + value based + model based  
model based算法主要用于棋类游戏  

# 应用

## 应用于下棋
生成两个agent，互相对弈，以胜负作为reward。  

## 应用于Chat-Bot
生成两个agent，互相对话。  
另外训练一个NN用于判断talk的好坏，并给予reward。  

## 应用于电子游戏

Gym：https://gym.openai.com/  
Universe：https://openai.com/blog/universe/

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

## 公式改进

### 改进一：Add a baseline

考虑这样的场景：  
假设某个state s在四个$\tau$中有出现。在这四次中，有三次做了Action a1，每次得到的reward为r=1。有一次做了Action a2，得到了reward r = 2。  
在这种情况下，应该认为在a2比a1好，但由于a2出现的次数太少，它的贡献得不到重视。  
因此增加这样一个Normalization，防止偏好出现几率高的action。  
3. 如果$R(\tau^n)$永远为正，那么根据softmax的计算公式，$R(\tau^n)$大的上升，$R(\tau^n)$小的下降。  
但如果某个action一直没有尝试过，那么它的分数肯定最低，会永远下降。  
![](/assets/images/Chapter7/27.png)    
解决方法：  
![](/assets/images/Chapter7/28.png)    

### 改进二：Assign Suitable Credit

一系列action相应的会有一系列的score。  
每个action的好坏都由总的score决定。  
但实际上很合理的情况是，  
1. 一个action只会影响它之后的得分，不会影响它之前的得分。所有由这个action之后的score之和来判断这个Action的好坏。   
![](/assets/images/Chapter7/62.png)    
2. 一个action对后面一系列的score的影响中，离Action的动作越远，action的影响越小。因此给score增加一个discount factor。  
![](/assets/images/Chapter7/63.png)    
3. b由NN训练得到。可以是state dependent的。  
advantage function $A^\theta(s_t, a_t)$

## On policy VS Off policy

On policy:  agent在与环境互动的过程中学习    
Off policy： agent看别人与环境互动的过程来学习。  

On polocy的缺点：  
每一次迭代都要重新sample data。要花很多时间在sample data上。  

改进方法：  
使用$\pi_{\theta'}$的sample来训练$\theta$。  
$\theta'$是固定的，因此同一组sample data可以用多次。  

### Importance Sampling

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

### 回到Reinforce Learning

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

### 近端优化策略

使用off policy代替on policy的前提是，$p_\theta$和$p_{\theta'}$不能差太多，否则效果不好。  
问：如何保证“$p_\theta$和$p_{\theta'}$不能差太多”？  
答：近端优化策略 PPO/TRPO  

#### PPO 正则化

对$J^{\theta'}(\theta)$增加一个正则化项：  
$$
J^{\theta'}_{PPO}(\theta) = J^{\theta'}(\theta) - \beta KL(\theta, \theta')
$$

#### TRPO 带限制最优化

$$
\begin{aligned}
J^{\theta'}_{TRPO}(\theta) = J^{\theta'}(\theta)   \\
s.t. && KL(\theta, \theta') < S
\end{aligned}
$$

#### $KL(\theta, \theta')$

$KL(\theta, \theta')$代表参数$\theta$和$\theta'$的距离。但不是它们的欧式距离或者KL散度。  
而是这两个参数对某个state产生的action的分布的KL散度。  

#### PPO算法过程  

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

#### PPO2算法过程

1. 同上  
2. 同上  
3. 目标函数替换为：  
$
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


# value based算法 --- learning a critic

critic不是决定action，而是评价action的好坏。  
Q-learning

