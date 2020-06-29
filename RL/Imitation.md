模仿学习 Imitation Learning  

使用场景：  
机器无法从环境中得到reward，只能看expert的demonstration  
例如chatbot，难以决定聊得好不好，但可以收集很多人的真实对话。  

# Behavior Cloning  

基于expert收集labelled data，把它当作监督学习来做.  
存在的问题：  
1. Experts only samples limited observation  
![](/assets/images/RL/8.png)    
例如开车，expert不会把车开始左上角，所以永远sample不到处于左上角的data。  
解决方法：data aggregation  
2. agent会学习expert的一些与action不相关的个人习惯  
3. Training data和Testing data不match  


# Innverse Reinforcement Learning --- IRL

RL:  
![](/assets/images/RL/9.png)    
根据Env和Reward选择Optimal Action  
IRL:  
![](/assets/images/RL/10.png)    
先根据expert和Env反推reward，再根据reward和Env选择Optimal Action。  
为什么要反推reward function?  
答：Modeling reward可能很简单。简单的reward function可以导出复杂的policy。  

## framework of IRL:  
1. expert $\hat \pi$与游戏互动得到的N个sample $\hat \tau$  
2. Actor $\pi$与游戏互动得到的N个sample $\tau$   
3. 先验假设：$\hat \tau$是最棒的，$\hat \tau$的分数一定高于$\tau$  
4. 学习一个reward function使得：  
$$
\sum R(\hat \tau_n) > \sum R(\tau_n)
$$

5. 用reward function + RL找actor $\pi'$  
![](/assets/images/RL/11.png)    
6. 用$\pi'$代替$\pi$，进入下一个迭代  

## RAN Vs. IRL

![](/assets/images/RL/12.png)    