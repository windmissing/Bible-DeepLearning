如何定义CNN的超参数
答：尽量不要自己设置超参数，而是查看文献中别人采用了哪些超参数。以前是一些经典的模型。  

# Ag例子

![](/assets/images/Chapter9/9.png) 

Input层和Pool层没有参数，大部分参数在FC层。  
Activation Size越来越小，如果下降太快可能会影响网络性能。    
 
# LeNet-5 
![](/assets/images/Chapter9/10.png)    
1. W减小，H减小，C增大。  
2. CONV -> POOL -> CONV -> POOL -> FC -> FC -> output，经典结构  
3. 使用sigmoid/tanh --- 废弃  
4. 使用复杂的计算来处理POOL中的通道 --- 当时性能限制，废弃  
5. 池化使用了非线性函数 --- 废弃

# AlexNet
![](/assets/images/Chapter9/11.png)    
1. 特征数量大。    
2. 能处理非常相似的基本模型。  
3. 使用ReLU  
4. 多GPU  
5. 局部响应归一化 --- 不常用

# VCG- 16

1. 相对一致的结构  
2. 特征数量非常大