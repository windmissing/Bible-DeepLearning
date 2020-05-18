单通道场景中，1 * 1卷积看上去没什么用。但它在多通道场景中是有用的。  
![](/assets/images/Chapter9/15.png)  
相当于对一个像素的32个通道上所有的点做一次全连接的计算。  

1 * 1卷积又叫network in network

# 应用：改变通道数

例如：

$28\times 28\times 192 * 32\text{个} 1\times 1\times 192 \rightarrow 28\times 28\times 32$  

可以用这种方式增加通道数