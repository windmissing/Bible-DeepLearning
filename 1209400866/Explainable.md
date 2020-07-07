例如ML将一张图分类为猫，同时要回答：  
 - 为什么你认为这是一只猫？ --- Local Explanation  
 - 你认为一只猫是怎样的？ --- Global Expanation  

 # Local Explanation  

 ## 方法一：  

 将Object x分解成Component (x1, x2, ..., xN)  
 移除一个compenent之后，对decision的影响越大，说明这个component越重要。  
 一个component可以是一个pixel、segment、方块、单词，或是自己定义的一个单位。  

component的大小必须很小心地设计，太大或太小都得不到好的结果

 ## 方法二：Saliency Map 

对输入向量x的一个分量增加一个小小的扰动$\Delta x$，得到：  
$$
(x_1, x_2, \cdots, x_n, \cdots, x_N) \rightarrow (x_1, x_2, \cdots, x_n+\Delta x, \cdots, x_N)
$$

x的一个分量通常是一个像素。  
由于x的改变，y也会相应地改变：  
$$
y \rightarrow y + \Delta y
$$

计算$|\frac{\Delta y}{\Delta x}|$，值越大，说明x的这个分量越重要。  
缺点：Gradient Sturation  
$$
|\frac{\Delta y}{\Delta x}| = |\frac{\partial y}{\partial x}|
$$

但在某个位置梯度小不代表它不重要。  
![](/assets/images/1209400866/11.png)  

## 方法三：用一个model解释另一个model  LIME

LIME = Local Interpretable Agnostic Explanations  

basic Idea：  
用一个有解释能力的model（例如Linear）去模仿另一个没有解释能力的model（例如NN）  
![](/assets/images/1209400866/13.png)  
但是，Linear Model不可能真的能模拟出NN，只能模拟出NN的一个local region  
![](/assets/images/1209400866/14.png)   
问：怎样fit with linear model?  
答：提取图像的m个特征，定义linear model为：  
$$
y = w_1x_1 + w_2x_2 + \cdots + w_mx_m + b
$$

问：怎么基于linear model解释？  
$w_m \approx 0 \Rightarrow x_m$与结果无关  
$w_m > 0 \Rightarrow x_m$对结果起积极作用
$w_m < 0 \Rightarrow x_m$对结果起消极作用  

# Global Explanation

## Activation Maximization

$$
x^* = \arg\max_x y_i
$$

其中yi为期望的结果的confidence  

这种方法还需要增加一些额外的限制，否则结果不好  

## Image Geneator

例如 GAN、VAE  
![](/assets/images/1209400866/12.png)  
$$
x^* = \arg\max_x y_i \Rightarrow z = \arg\max_z yi, x = G(z)
$$