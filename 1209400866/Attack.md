对抗模型，要对抗的是：  
- 噪音  
- human制造出来的有恶意的攻击  

# 什么是攻击  

输入正常的x0，NN会得到正确的结果  
但通过经心设计$\Delta x$，使得NN会在$x+\Delta x$上犯错，而人类无法分辨$x+\Delta x$和x。  

# 怎样攻击

- 正常训练  

![](/assets/images/1209400866/2.png)  

$$
L(\theta) = C(y^0, y^{\text{true}})
$$

x是固定的，调整$\theta$

- 无目标攻击

![](/assets/images/1209400866/3.png)  

$$
L(x') = - C(y', y^{\text{true}})
$$

$\theta$是固定的，调整x  

- 有目标攻击

![](/assets/images/1209400866/4.png)  

$$
L(x') = - C(y', y^{\text{true}}) + C(y', y^{\text{false}})
$$

- 额外限制

$$
d(x^0, x') \le \epsilon
$$

不要被发现

## d是怎么定义的？

[L2-norm、L-infinity](https://windmissing.github.io/mathematics_basic_for_ML/LinearAlgebra/norm.html)  

图像应用上，L-infinity比L2-norm更合适

## 怎样求解x  

### 方法一：梯度下降法
$$
\begin{aligned}
x^* = \arg\min_{d(x^0, x') \le \epsilon} L(x')
\end{aligned}
$$

这是一个带限制的最优化问题  

1. 将x'初始化为x0  
2. 用梯度下降法更新xt  
$$
x^t = x^{t-1} - \eta \nabla L(x^{t-1})
$$

3. 增加限制  
如果{d(x^0, x') > \epsilon}，则xt = fix(xt)
![](/assets/images/1209400866/5.png)  

### 方法二：FGSM  

Fast Gradient Sign Method

$$
\begin{aligned}
x^* = x^0 - \epsilon \Delta x  \\
\Delta x = \text{sign} (\frac{\partial L}{\partial x'})
\end{aligned}
$$

![](/assets/images/1209400866/6.png)  

直观解释：  
FGSM不在意$\Delta x$的值，只在意它的方向  
例如$\Delta x$指向左右任意一个方向，x*都会走到右上角  
可以看任是一个LR巨大的一次GD  
![](/assets/images/1209400866/7.png)  

# White Box VS Black Box

## 白盒攻击 - 知道NN的参数

以上方法都是白盒攻击  

## 黑盒攻击 - 不知道NN的参数，有NN的训练集  

1. 用training data训练一个NN proxy  
2. 用白盒攻击方法找到能攻击NN proxy的图像  
3. 用这张图像去攻击NN

## 黑盒攻击 - 不知道NN的参数，没有NN的训练集

1. 用自己的training data喂NN，得到output  
2. 用自己的training data + NN的output得到labelled data  
3. 用labelled data训练NN proxy  
4. 用白盒攻击方法找到能攻击NN proxy的图像  
5. 用这张图像去攻击NN

# 其它攻击方法

## Universal Adversal Attack

不是针对某个样本制作攻击  
制作一张通过的$\Delta x$图像  
这张图像叠加到任意图像上都有攻击作用  
可以用于white或者black  

## Adversarial Reprogramming  

通过攻击改变一个NN的功能

