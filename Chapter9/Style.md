![](/assets/images/Chapter9/27.png)  

# DL Network是怎么学习的

1. 选择layer 1的Unit，观察怎样的输入让这些unit最“兴奋”，得到这样9张图。    

![](/assets/images/Chapter9/28.png)  

2. 用同样的方法观察其它layer，每个layer得到这样一些图：  

![](/assets/images/Chapter9/29.png)  
![](/assets/images/Chapter9/30.png)  
![](/assets/images/Chapter9/31.png)  
![](/assets/images/Chapter9/32.png)  

# 网络迁移的算法

定义：  
原图像为Content，简称C  
风格图像为Style，简称S  
生成图像为Generated，简称G  

1. 随机初始化G  
2. 定义代价函数J（G）  
3. 使用梯度下降法最小化J（G）  
$$
G = G - \frac{\partial J(G)}{\partial G}
$$

直接更新图像G的像素值。  

# 定义代价函数J（G）

代价函数J（G）由两部分组成：G与C的相似度、G与S的相似度  

$$
J(G) = J_C(C, G) + J_S(S, G)
$$

## $J_C$代价函数

假设使用第l层hidden layer来计算content cost  
- l太小，则生成图像太接近原图像  
- l太大，则生成图像与原图像差太多  
因此要合理地选择l，通常选择网络的中间层。  

定义：$a^{[l](C)}$和$a^{[l](G)}$分别为C和G在第l层的激活值(a，也可写作h)  
如果：$a^{[l](C)}$和$a^{[l](G)}$接近  
则：图像C和G接近   
因此  
$$
J_C(C, G) = \frac{1}{2}||a^{[l](C)} - a^{[l](G)}||^2
$$

## $J_S$代价函数

定义一个图像的style为:**correlation between activationas accross channels**  

怎样评价两个通道的correlation？  

