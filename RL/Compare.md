# Q-Learning Vs Policy Based

Q-Learning比policy gradient好train，因为：  
只要学到Q function，就能得到一个比较好的policy。  

Q-Learning的缺点是：难以处理action是continuous的情况。  

## 什么时候action是连续的？  

答：例如开车，action可以是方向盘转多少度。  

## continuous action对Q-Learning有什么问题?  

答：Q-Learning的一个步骤是求解：  
$$
a = \arg\max_a Q(s, a)
$$

但a无法穷举

## 解决方法  

1. sample一组action  
缺点：这样得到的Action不会太精确  
2. 用gradient ascend来解a的最优化问题  
缺点：这样运算量大，且会遇到local minima的问题。  
3. 对$Q^\pi$这个NN做特别的设计，使得容易计算最优化问题   
![](/assets/images/Chapter7/74.png)    
$$
\begin{aligned}
Q(s, a) = -(a-\mu(s))^\top\sum(s)(a-\mu(s)) + V(s)  \\
\mu(s) = \arg\max_a Q(s,a)
\end{aligned}
$$

