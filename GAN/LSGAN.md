LSGAN = Least Square GAN

# PG和Pdata这两个分布没有重叠

解释一：  
因为Img是高维在低维上的manifold，因此即使两个Img在低维上重叠，它们在高维上也是可以分开的。  

解释二：
PG和Pdata的分布本身可能是重叠的，它们实际在公式中使用的是PG和Pdata的sample，它们是不重叠的。  
![](/assets/images/GAN/16.png)  

# PG和Pdata不重叠会有什么问题

对于两个不重叠的分布，应该使用分布之间的距离来衡量它们之间的相似度。  
![](/assets/images/GAN/17.png)   

当前GAN公式中使用的是JS Divergence[link](https://windmissing.github.io/Bible-DeepLearning/GAN/Thoery.html)  
对于JS Divergence，两个不重叠的分布的JSD永远是log2。如上图三种情况，图1和图2没有重叠，JSD是log2，图三完全重叠，JSD是0。  
这带来的问题时，当PG和Pdata为图1的关系时，它不知道要如果改进。也不知道图2是比图1更好的结果。  

# 解决方法

linear代替sigmoid，将分类问题变成了回归问题。  
![](/assets/images/GAN/18.png)   