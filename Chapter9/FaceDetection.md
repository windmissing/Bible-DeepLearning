人脸验证问题：  
输入：图像，name/ID，输出：图像和name/ID是否对应  
人脸识别问题：  
输入：datasets， image  
输出：image对应的name/ID  

# 人脸验证的挑战 

One-Shot Challenge：需要通过单一一张图片，就识别到这个人  
当DL只有一个样本时，表现非常不好。  

解决方法：  
不直接训练f(图像)=id。  
而是训练f(图像1、图像2) = 相似度。  
if f(img1, img2) <= t ==> 同一个人
if f(img1, img2) > t  ==> 不是同一个人

# Siamese Network

图像1 --NN--> 向量x1  
图像2 --NN--> 向量x2  
d(图像1，图像2) = $||x_1 - x_2||^2_2$  
用向量x1、x2代替原始的样本图像1、图像2。  

## Triplet Loss 三元组损失函数  

定义三种图像分别为Anchor(A)、Positive(P)、Negative(N)  
定义一个三元组的损失函数为：  
$$
L(A, P, N) = max(||f(A) - f(P)||^2 - ||f(A) - f(N)||^2 + \alpha)
$$

其中：  
f代表将图像通过神经网络转成向量。  
公式原理为：“同一个人的两张图像的距离” 应比 “不同人的两张图像的距离” 小$\alpha$。  
$\alpha$为间隔。  

整个训练集的损失函数为每个三元的损失之和。  

Note：  
训练时不是one-shot的，训练集中每个人至少要有2张照片（分别用做A和P），才能完成训练。  
在预测时可以是one-shot的。  

## 怎样根据训练集生成三元组？  

1. 随机生成  
缺点：不等式d(A, P) + a <= d(A, N)太容易满足了，NN从中学不到东西。  
2. 选择“hard to train”的三元组，即d(A, P)和d(A, N)接近的三元组。  
优点：加速算法学习

# 把人脸识别转成二分类问题

![](/assets/images/Chapter9/26.png)   
每张图像生成一个向量，例如xi --> f(xi)，xj --> f(xj)  
$$
\hat y = \sigma\left(\sum_k w_i d(f(x_i), f(x_j)) + b \right)
$$

其中$d(f(x_i), f(x_j))$代表d(f(x_i), f(x_j))代表f(x_j)和f(x_j)的相似度。  
例如X方相关度：  
$$
d(f(x_i), f(x_j)) = \frac{(f(x_i)-f(x_j))^2}{f(x_i)+f(x_j)}
$$

在预测时，database中图像的f(x)可以提前准备好（预训练）。  
每次只需要重新计算要预测的图像的f(x)即可。  