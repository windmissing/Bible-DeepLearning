![](/assets/images/GAN/30.png)   

把Encoder和Decoder分开，不是串联，而是并联。  

Encoder：输入sample的真实图像x，输出生成的code z  
Decoder：输入某分布sample出的code z，输出生成的图像x  
Discriminator：输入一对(x, z)，D辨别这一结数据来自Encoder还是Decoder  

# 训练过程

1. 从database sample出M张图像，记为x  
2. $\tilde z = En(x)$  
3. 从某分布sample出M个code，记为$z$  
4. $\tilde x = De(z)$  
5. 更新Dis的参数，目标：Dis$(x, \tilde z)\uparrow$，Dis$(\tilde x, z)\downarrow$  
6. 更新En和De的参数，目标：Dis$(x, \tilde z)\downarrow$，Dis$(\tilde x, z)\uparrow$  
En和De联手骗过Dis  

# BiGAN的原理

设联合分布$(x, \tilde z)$为P，$(\tilde x, z)$为Q  
希望通过Dis的引导，让P和Q越接近越好。  

普通auto-encoder得到的图像：不清晰，且跟原图很像  
BiGAN得到的图像：清晰，且跟原图不像  

# Triple GAN

![](/assets/images/GAN/31.png)   
用于半监督学习