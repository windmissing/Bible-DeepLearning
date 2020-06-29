# 改进一：Add a baseline

考虑这样的场景：  
假设某个state s在四个$\tau$中有出现。在这四次中，有三次做了Action a1，每次得到的reward为r=1。有一次做了Action a2，得到了reward r = 2。  
在这种情况下，应该认为在a2比a1好，但由于a2出现的次数太少，它的贡献得不到重视。  
因此增加这样一个Normalization，防止偏好出现几率高的action。  
3. 如果$R(\tau^n)$永远为正，那么根据softmax的计算公式，$R(\tau^n)$大的上升，$R(\tau^n)$小的下降。  
但如果某个action一直没有尝试过，那么它的分数肯定最低，会永远下降。  
![](/assets/images/Chapter7/27.png)    
解决方法：  
![](/assets/images/Chapter7/28.png)    

# 改进二：Assign Suitable Credit

一系列action相应的会有一系列的score。  
每个action的好坏都由总的score决定。  
但实际上很合理的情况是，  
1. 一个action只会影响它之后的得分，不会影响它之前的得分。所有由这个action之后的score之和来判断这个Action的好坏。   
![](/assets/images/Chapter7/62.png)    
2. 一个action对后面一系列的score的影响中，离Action的动作越远，action的影响越小。因此给score增加一个discount factor。  
![](/assets/images/Chapter7/63.png)    
3. b由NN训练得到。可以是state dependent的。  
advantage function $A^\theta(s_t, a_t)$

