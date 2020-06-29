# Double DQN

定义两个function，分别是Q和Q'，  
$$
Q(s_t, a_t) = r_t + \max Q'(s_{t+1}, \arg\max_a Q(s_{t+1}, a))
$$

Q用于选择action，Q'用于评价action。  
只有Q和Q'同时对某个action高估，才会使总体结果高估。  

实际上，本来就存在两个Q，分别是Q和$\hat Q$。  
可以直接把$\hat Q$当Q'用。  

# Deuling DQN

与Double DQN的区别是network structure不同。  
![](/assets/images/Chapter7/70.png)    
图中上面为普通Q-Learning，正面是Deuling DQN。  
Deuling DQN的NN不直接输出Q value，而是输出V(s)和A(s, a)。这两者这和为Q value。  

这个结果是怎么起作用的？  
![](/assets/images/Chapter7/71.png)    
假如 - 某个state的大部分action都做同样的更新，  
那么 - 不直接更新Q(s, a)，而是更新V(s)  
这样 - 这个state的所有action都得到了更新，即使是没有sample到的Action。   
为了避免 - V(s) = 0，没有Dueling DQN的效果，  
增加限制 - 令更新A的要求比较高，使NN更倾向于更新V(s)  
例如 - 要求A(s,a)之和为0  

# Prioritized Reply

从buffer中sample data时，不是uniform地sample，而是重点选择TD error比较大的sample。  

# Multi-Step

MC和TD的balance  
buffer中的data不是$(s_t, a_t, r_t, s_{t+1})$，而是  
$$
(s_t, a_t, r_t, \cdots, s_{t+N}, a_{t+N}, r_{t+N}, s_{t+N+1})
$$

![](/assets/images/Chapter7/72.png)    

# Noise Net

Noise On Action -> Noise on Q parameters  
具体做法是在每一局开始时对Q的参数加noise。  

Noise On Action和Noise on parameters的目的都是Exploration，区别是：  
Noise On Action在每一局中同样的State得到action都是随机的，即“随机乱试”。  
Noise on parameters在同一局中同的State得到的action是相同的。不同局中同样的state得到的Action可能不同。即“有系统地试”、“explore in a constant way”。  

# Distributional Q-Function

不是输出$Q^\pi(s,a)$的值，或出$Q^\pi(s)$的期望，而是输出$Q^\pi(s)$关于a的分布。   
![](/assets/images/Chapter7/73.png)    

实验发现，Distributional Q-Function可以解决Q value高估的问题，此时Double DQN就不需要了。  

