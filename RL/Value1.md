# value based算法

这种方法不直接学习policy，而是学习critic。  
critic不直接采取行为，而是评价现在的行为。  

## 定义 state value function $V^\pi(s)$  

对于一个特定的$\pi$，输入当前state $s$，输出期望的cumulated reward。    
有两种方法来衡量$V^\pi(s)$，分别是MC法和TD法。  

### Monte-Carlo (MC) based 蒙特卡罗法  

使用$\pi$与$s_a$真实互动并统计reward $G_a$。  
收集labelled data (s_a, G_a)。  
使用labelled data来训练NN，这是一个回归问题。    
缺点：必须玩到游戏结束才能收到到reward。有些游戏要玩很久，太耗时。  

### Temporal-difference (TD) based

$$
V^\pi(s_t) = V^\pi(s_{t+1}) + r_t
$$

![](/assets/images/Chapter7/66.png)    

### MC Vs. TD

![](/assets/images/Chapter7/67.png)    
TD比较稳，MC比较精确  
TD更常用

## 定义 state-action value function $Q^\pi(s, a)$   

对于一个特定的$\pi$，输入当前state $s$，**并强制采取动作a**，输出期望的cumulated reward。    
$Q^\pi(s, a)$是一个NN，可以有以下两种结构：  
![](/assets/images/Chapter7/68.png)    

