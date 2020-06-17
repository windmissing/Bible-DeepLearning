![](/assets/images/GAN/29.png)   

从VAE的角度看，加上D使用decoder的图像质量更高。  
从GAN的角度看，加上encoder，可以使学习有个目标，学习效果更稳定。  

Encoder的目标：  
1. 最小化reconstructure error  
2. z to normal  
Decoder的目标:  
1. 最小化reconstructure error  
2. 难过D  
D的目标：  
区分是生成的Img还是reconstructed Img。  

# 训练步骤

1. 初始化En， De, Dis  
2. 从database sample出M张图像，计做x  
3. $\tilde z = En(x)$  
4. $\tilde x = De(\tilde z)$  
5. 从某个分布（例如正态分布）中sample出M个code向量，记做z  
6. $\hat x = De(z)$，此时有三种Img，分别是（2）database sample出来的（4）x经过auto-encoder重新生成的（6）decoder随机生成的  
7. 更新En的参数，目标：$||\tilde x -x ||\downarrow$，KL$(P(z|x)||P(z))\downarrow$  
8. 更新De的参数，目标：$||\tilde x -x ||\downarrow$，Dis$(\tilde x)\uparrow$，Dis$(\hat x)\uparrow$  
9. 更新Dis的参数，目标：：$||\tilde x -x ||\uparrow$，Dis$(\tilde x)\uparrow$，Dis$(\hat x)\downarrow$  