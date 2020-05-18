# 什么是残差网络

正常的网络层是这样的：  
![](/assets/images/Chapter9/12.png)    
残差块是这样的：  
![](/assets/images/Chapter9/13.png)    
残差块使网络可以更深。网络深度与训练集性能的关系：  
![](/assets/images/Chapter9/14.png)    

# 为什么残差网络有用？

$$
\begin{aligned}
a^{l+2} &=& g(z^{l+2} + a^l) \\
&=&g(W^{l+2}a^{l+1} + b^{l+2} + a^l)  && \text{由于L2 正则化的影响,W会shrink,假设此时W=0} \\
&=& g(a^l) && \text{假设使用ReLU}  \\
a^{l}
\end{aligned}
$$

根据以上公式$a^{L+2}$可以很容易地得到与a^l相同的结果。（性能不变）  
运气好的话，a^{l+2}能得到比$a^l$好的结果。（性能提升）  
为什么使用残差块用的跳跃连接，要求使用same卷积，或者将$a^l$映射到$z^{l+2}$的大小。  