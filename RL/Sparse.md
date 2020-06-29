稀疏奖励 Sparse Reward  
指大多数情况action没有reward的游戏。  

# Reward Shaping

developer自己设计一些reward来引导agnet  
这些reward不是来自环境真正的reward。  

## 根据游戏设计

缺点：需要domain knowledge  

## Curiosity Reward  

![](/assets/images/RL/5.png)    

ICM = intrinsic curiosity module

ICM根据(s1, a1, s2)计算出$r1^i$  
$$
R(\tau) = \sum(r_t + r_t^i)
$$

### ICM的设计  

![](/assets/images/RL/6.png)    
1. Network1根据at和st预测$\hat s_{t+1}$  
2. 比较$\hat s_{t+1}$和$s_{t+1}$  
3. diff越大，$r_t^i$越高  
即：  
action越无法预测，action的reward越大。（鼓励冒险）  
存在的问题：  
some states is hard to predict, but not important。  
例如：风吹草动。  
机器不能什么都不做只是站着看风吹草动。  

### ICM改进版  

增加feature extraction  

![](/assets/images/RL/7.png)    
feature extraction通过Network2实现，用于把state中没有意义的东西去掉。  
N2的训练方法为：根据$\hat s_{t+1}$和$s_{t+1}$预测$\hat a_t$

# Curriculum Learning

给机器的学习做规划，labelled data由简单到难  

# Hierarchial Reinforcement Learning