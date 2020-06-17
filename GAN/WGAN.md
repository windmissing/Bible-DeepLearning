WGAN = Wasserstein GAN

# WGAN的主要改进

1. 更新G的参数过程中，用Earth Mover's Divergence代表JS Divergence
WGAN可以解决不overlap情况下Divergence不变的问题。  

2. 更新D的参数过程中，增加1-Lipschitz限制，得到PG到Pdata可以平滑过渡

## Earth Mover's Divergence

### case 1

![](/assets/images/GAN/19.png)   

W(P, Q) = d

### case 2

![](/assets/images/GAN/20.png)   

分布如图P、Q所有，可以构造不同的move plan把P变成Q，不同的move plan会得到不同的距离，例如：  
![](/assets/images/GAN/21.png)   
因此需要穷举所有的move plan找到最小的move plan。  
定义某个move plan为$\gamma$，$B(\gamma)$为对应的距离：  
$$
\begin{aligned}
B(\gamma) = \sum_{x_p, x_q}\gamma(x_p, x_q)||x_p-x_q||   \\
W(P, Q) = \min B(\gamma)
\end{aligned}
$$

例如这里个case中最好的move plan是这样的：  
![](/assets/images/GAN/22.png)   

W可以解决JS在不overlap情况下距离不变的问题。  

## 1-Lipschitz限制

### 为什么限制

D的目标增加Pdata的D(x)，减小PG的D(x)。  
![](/assets/images/GAN/23.png)   
但只是这样的目标，可能永远无法收敛，因为左边可以无限下降而右边可以无限上升。  
到最后左边跟右边之间差距非常大而形式悬崖，悬崖导致难以从左边优化到右边。  
因此增加D的形状限制，要求D是平滑的。1-Lipschitz就是一种平滑限制。  
Lipschitz平滑定义为：  
$$
||f(x_1)-f(x_2)|| \le K||x_1-x_2||
$$

当K为1时，称为1-Lipschitz平滑。  

### 怎么限制

论文作者没有很的方案，只是建议weight clipping。  
这种方法不一定能真的实现1-Lipschitz限制。  

# WGAN-GP的主要改进

## 改进一

在WGAN的基础上，提供了一种实现1-Lipschitz限制的方法。  

原理：  
$D \in$1-Lipschitz等价于$||\nabla_x D(x)||\le 1$for all x  
for all x意味着要遍历所有的可能，即：  
$$
V(G, D) = \max\left(V(G, D), \lambda\int_x \max(0, ||\nabla_xD(x)||-1)dx\right)
$$

公式中\int_x即代表遍历所有的可能，但这是不可行的，因此公式改为：  
$$
V(G, D) = \max\left(V(G, D), \lambda E_{x\in P_{\text{penalty}}}\left[\max(0, ||\nabla_xD(x)||-1)\right]\right)
$$

什么是$x\in P_{\text{penalty}}$?  
![](/assets/images/GAN/24.png)   
1. 在PG和Pdata中各任意取2个点。  
2. 两个点分别连线  
3. 连线上随机sample的点就是penalty  

## 改进二

WGAN中为了让D平滑，只惩罚了梯度>1的情况，而不care梯度<的情况。  
实验结果表明，梯度越接近1效果越好。因为V(G,D)公式改为：  
$$
V(G, D) = \max\left(V(G, D), \lambda E_{x\in P_{\text{penalty}}}\left[(||\nabla_xD(x)||-1)^2\right]\right)
$$

# Spectrum Norm

Keep gradient norm smaller than 1 everywhere

# 怎么把GAN改成WGAN

![](/assets/images/GAN/25.png)   
