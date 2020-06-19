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




# value based算法 --- learning a critic

critic不是决定action，而是评价action的好坏。  
Q-learning

