已知有一个G，可以根据G的向量z生成图像x。  
那么怎么找到z?  
解决方法：  
![](/assets/images/GAN/37.png)   
利用G训练一个encoder，利用D初始化encoder

---------------------------------------------

对于一张输入图像Image:  
1. 用以上方法反推出它的vector  
2. 以调头发长短为例，把所有短发Imge的vector取平均，得到v1，把所有长发Imge的vector取平均，得到v2   
3. v2 - v1 = 修改头发长短的向量，$z_{long} = \frac{1}{N_1}\sum_{long}En(x) -  \frac{1}{N_2}\sum_{short}En(x)$  
4. 短发 x -> En(x) + z_long -> z' -> G(z')长发  
