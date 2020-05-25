本章主要内容：  
- 什么是卷积运算  
- 使用卷积的动机  
- 池化  
- 神经网络中卷积的变种  
- 不同难度数据的卷积  
- 使卷积更高效的方法  
- 神经科学原理

> **[success]**  
> CNN基于这样一些先验知识对前馈网络优化：  
> (1) some pattens are much smaller than the whold image  --- filter  
> (2) the same pattens appear in different regions  --- shared parameters  
> (3) subsampling the pixels will not change the object --- pooling  
> 完整的CNN网络结构：  
> ![](/assets/images/Chapter9/2.png)  

