# Q-Learning

1. 使用一个$\pi$与环境互动，生成labelled data  
2. 根据labelled data学到$Q^\pi(s, a)$  
3. 根据$Q^\pi(s, a)$找到更好的$\pi'$  
4. 用$\pi'$代替$\pi$，goto step 1  

## 怎么评价哪个$\pi$更好？  

答： $V^{\pi'}(s) \ge V^\pi(s)$ for all state s，则认为$\pi'$比$\pi$好。  

## 怎么根据$Q^\pi(s, a)$找到$\pi'$？  

答：  
$$
\pi'(s) = \arg\max_a Q^\pi(s, a)
$$

continuous a不适用。  

## 怎么根据labelled data学到Q？  

答： Target Network  
![](/assets/images/Chapter7/69.png)    
注意：  
因为左边的$Q^\pi$和右边的$Q^\pi$是同一个模型，当左边的$Q^\pi$因为迭代而更新时，右边的$Q^\pi$也会改变。  
左边$Q^\pi$更新 --> 右边$Q^\pi$改变 --> 右边的输出改变 --> 左边的目标改变。  
如果一个NN的目标一直在改变，这个NN就会很难train。  
解决方法：右边的$Q^\pi$先fix住，训练左边的$Q^\pi$。左边的$Q^\pi$更新一定次数后把它参数同步到右边的$Q^\pi$。  

## Exploration

$$
\pi'(s) = \arg\max_a Q^\pi(s, a)
$$

这个公式存在的问题是：  
如果$Q^\pi(s, a_1)=1$，而$Q^\pi(s, a_2)$、$Q^\pi(s, a_3)$没有被sample过，其值为默认值0。  
那么以后每次遇到s永远会采取$a_1$。  
解决方案：探索机制  

### Epsilon Greedy  

$$
a = 
\begin{cases}
\arg\max_a Q(s_a) && , 1-\epsilon \text{的机率}\\
\text{random} && , \text{otherwise}
\end{cases}
$$

### Boltzmann探索

$$
\begin{aligned}
P(a|s) = \frac{\exp(Q(s, a))}{\sum_a \exp(Q(s, a))}
\end{aligned}
$$

## Replay Buffer

这是一个用于存互动经验的buffer，每一项的格式为$s_t, a_t, r_t, s_{t+1}$  
每一轮迭代都可以产生N个data，但buffer可以开很大，里面可以存很多轮data。  
因此step 2用于生成Q的labelled data不止是都个$\pi$的互动data。  

优点：  
1. 充分利用data，减少与环境互动的时间  
2. batch里面的data越diverse越好。这些方法便data更diverse。  

## Q-Learning的过程

1. 初始化两个NN，分别是$Q$和$\hat Q$，且$\hat Q = Q$  
2. given $s_t$，take $a_t$ based on Q(epsilon greedy)  
3. 得到$r_t$和$s_{t+1}$，把$(s_t, a_t, r_t, s_{t+1})$存入buffer  
4. 从buffer sample一个batch的data  
5. 计算target：  
$$
y = r_i + \max_a \hat Q(s_{i+1}, a)
$$

6. 更新Q的参数，使$Q(s_i, a_i)$接近y（回归问题），goto step 4
7. 令$\hat Q = Q$

## Q-Learning存在的问题：Q value的估测值高于实际值。  

为什么Q value的估测值会特别高？  

$$
Q(s_t, a_t) = r_t + \max Q(s_{t+1}, a_t)
$$

Q是一个NN，可能会对某个action高估，某些action低估。  
而max倾向于选择被高估的action。  

解决方法：Double DQN

