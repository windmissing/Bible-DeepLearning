Meta Learning = Learn to Learn  

# 什么是Meta Learning?  

LLL与Meta Learning的关系？  

同：学过很多task之后，期待在新的task上会学得更好  
异：LLL是让同一个model学会所有的task。Meta Learning是让Model学会学习的能力。  

什么是Mechine Learning？  
1. 定义一个F（mechine learning algorithm），输入大量训练数据，输出一组参数f*  
2. 基于参数f*，输入testing data，输出预测结果  

Mechine Learning是一种找f的能力。  
Meta Learning是一种找F的能力。  

Meta Learning的步骤：  
![](/assets/images/1209400866/27.png)  
1. 定义一组learning algorithm F
图是的红框原本都是人工设计出来的。选择不同的设计就是不同的F。  
2. 评价一个F的好坏  
![](/assets/images/1209400866/28.png)  
$$
L(F) = \sum_{n=1}^N l^n
$$

3. 找到最好的F  

# MAML

Model Agnostic Meta-Learning  
只关注“初始化参数” 
![](/assets/images/1209400866/29.png)   
$$
\begin{aligned}
L(\phi) = \sum_{n=1}^N l^n(\hat \theta^n)  \\
\phi = \phi - \eta \nabla_{\phi} L(\phi)
\end{aligned}
$$

![](/assets/images/1209400866/30.png)   
横轴：model参数  
纵轴：loss  

$\phi$参数可能在task 1和task 2上效果一般，  
但是从$\phi$出发，  
在task 1上能容易地找到最优参数$\hat \theta^1$，  
在task 2上能容易地找到最优参数$\hat \theta^2$，  
因此$\phi$是一个好的初始化参数。  

![](/assets/images/1209400866/31.png)   
这个$\phi$不是一个好的初始化参数，因为它会导致task 2收敛不到最小。  

MAML假设：  
最后得到一个比较好的初始化参数$\phi$以后，只需经过一次gradient descent，就能找到最好的$\hat \theta$。  
以下是推导公式：  
$$
\begin{aligned}
\hat \theta = \phi - \epsilon \nabla_{\phi} l(\phi)  \\
L(\phi) = \sum_{n=1}^N l^n(\hat \theta^n) \\
\phi = \phi - \eta \nabla_{\phi} L(\phi)   \\
\nabla_{\phi}L(\theta) = \sum_{n=1}^N \nabla_\phi l^n(\hat \theta ^n)  \\
\nabla_{\phi} l(\hat \theta) = \frac{\partial l(\hat \theta)}{\partial \theta_i} = \sum_j \frac{\partial l(\hat \theta)}{\partial hat \theta_j} \frac{\partial hat \theta_j}{\partial \theta_i} \approx \frac{\partial hat \theta_j}{\partial hat \theta_i} = \nabla_{\hat\theta}l(\hat \theta)
\end{aligned}
$$

![](/assets/images/1209400866/32.png)   

# Reptile

![](/assets/images/1209400866/33.png)     
![](/assets/images/1209400866/34.png)     