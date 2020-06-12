假设要生成的x是高维空间中的一个点  
在高维的Image Space中，只有一小部分点example出来是合理的。  
即：要产生的x符合一个固定的distribution。目标是找出这个distribution。  

定义目标distribution为$P_{\text{data}}(x)$   

# 在GAN之前，使用最大似然估计来找$P_{\text{data}}(x)$  

![](/assets/images/GAN/7.png)
1. 定义一个distribution $P_G(X;\theta)$  
2. 调整$\theta$，使$P_G$与$P_{\text{data}}(x)$越接近越好。  
- 2.1 从$P_{\text{data}}(x)$中sample出m个点$x_i$  
- 2.2 计算$P_G(x_i; \theta)$  
- 2.3 定义最大似然估计公式$L = \prod_{i=1}^mP_G(x_i;\theta)$  
- 2.4 找出最大化L的参数$\theta^*$

# 最大似然估计 = 最小KL散度  

$$
\begin{aligned}
\theta^* &=& \arg\min_{\theta} \prod_{i=1}^mP_G(x_i;\theta)   \\
&=& \arg\min_{\theta} KL(P_{\text{data}}||P_G)
\end{aligned}
$$

# 后面的我也不知道在讲什么，只是把讲的内容记下来

$$
\begin{aligned}
G* = \arg\min_G \text{Div}(P_G, P{\text{data}})
\end{aligned}
$$

公式中的$P_G$未知，所以无法直接比较这两个分布的KL散度  

从$P_G$和$P{\text{data}}$各sample出一些data  
$$
V(G, D) = E_{X\sim P{\text{data}}}[\log D(x)] + E_{X\sim P{\text{G}}}[\log (1-D(x))]  
$$

在计算公式过程中：  
D(x)可以是任意function，通常是由NN训练得到。  
分布$P_G$是固定不变的。  
解以上公式得：  
$$
\begin{aligned}
D^* &=& \arg\max_D V(D, G)  \\
&=& \arg\max_D P{\text{data}}(x) \log D(x) + P_G(x) \log(1-D(x))  \\
&=& \arg\max_D a \log D + b \log (1-D)
\end{aligned}
$$

说明：上面公式中，为了简化计算，人为定义出：  
$$
\begin{aligned}
a &=& P{\text{data}}(x) \\
b &=& P_G(x) \\
D &=& D(x)   \\
f(D) &=& a \log D + b \log (1-D)
\end{aligned}
$$

直接寻找f(D)偏导为0的点：  
令$\frac{df(D)}{dD} = 0$，得：$D^* = \frac{a}{a+b}$  
把$D^*$代入V(G,D)得：  
$$
\begin{aligned}
V(G, D^*) &=& -2\log 2 &+& KL(P{\text{data}}||\frac{P{\text{data}}+P_G}{2}) + KL(P{\text{G}}||\frac{P{\text{data}}+P_G}{2})  \\
&=& -2\log 2 &+& 2JSD(P{\text{data}}||P_G)
\end{aligned}
$$

公式中，JSD代表[Jesen-Shannon Divergence](https://windmissing.github.io/mathematics_basic_for_ML/Information/KL.html)

上面提到公式G*的计算：
$$
\begin{aligned}
G^* = \arg\min_G \text{Div}(P_G, P{\text{data}})
\end{aligned}
$$

G*无法直接计算，根据上面的推导得到：  
$$
\begin{aligned}
G^* = \arg\min_G\max_D V(G, D)   \\
D^* = \arg\max_D V(G, D)
\end{aligned}
$$

**怎么理解上面这两个公式：**  
![](/assets/images/GAN/8.png)  
图中红点代表能使V(D, G)最大的$D^*$  
G3是这三个中最优的G*。  

**训练步骤：**  
1. 初始化G和D  
2. 迭代  
- 2.1 固定住G，更新D  
- 2.2 固定住D，更新G  

由于计算$G^* = \arg\min_G\max_D V(G, D)$这一步要求固定住D，因此定义loss function为：  
$$
L(G) = \max_D(V, G)
$$

问：带max的分段函数怎么求导？  
答：以下图为例：  
![](/assets/images/GAN/9.png)  
$$
\frac{df(x)}{dx} = \frac{df_i(x)}{dx}
$$

if $f_i(x)$ is the max one。  

求G*的迭代过程没看懂，直接上图：  
![](/assets/images/GAN/10.png)   
后面就放弃了

