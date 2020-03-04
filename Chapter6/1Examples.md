# 任务  

使用一个功能完整的前馈网络学习XOR函数。  

# 已知：  
XOR函数有两个非0即1的输入。只有当前x1、x2中有一个是1时输出为1，其它情况输出为0。  
目标：y = f*(x) = XOR(x1,x2)  
输出：$$y = f(x; \theta)$$  
训练样本集：$$X = \{[0,0]^T, [0,1]^T, [1,0]^T, [1,1]^T\}$$  

# 选择

选择损失函数为MSE，即：  
$$
J(\theta) = \frac{1}{4}\sum_{x \in X}(f^*(x)-f(x;\theta))^2
$$
选择模型为线性模型，即：  
$$
f(x;\theta) = f(x;w, b) = x^Tw+b
$$
[?]通过闭解形式解得J(\theta)的最小化为w = 0, b = 1/2
显示这个结果无法表达XOR函数  

# 改进

使用不同特征空间，即输入和输出之间增加一个中间层。  
![](http://windmissing.github.io/images_for_gitbook/Bible-DeepLearning/1.png)  
其中：  
中间层：  
$$
h = f^{(1)}(x; W, c)  \\
\require{AMScd}
\begin{CD}
    向量 @>矩阵W，向量c>> 向量
\end{CD}
$$
输出层：  
$$
y = f^{(2)}(h; w, b) \\
\require{AMScd}
\begin{CD}
    向量 @>向量w，标量b>> 标量
\end{CD}
$$
完整输出为：  
$$
f(x; W, c, w, b) = f^{(2)}(f^{(1)}(x)) \tag{1}
$$

# f(1)

显然f(1)不是一种线性变换，否则还是相当于y是x的线性组合。  
f(1)通常为这样的形式：  
$$
h = g(W^Tx+c)   \tag{2}，g是激活函数
$$
g一个作用于整个向量的函数，通常默认为ReLU函数。
**ReLU = rectified linear unit = 整流线性单元**
$$
g(z) = \max(0, z)   \tag{3}
$$
![](http://windmissing.github.io/images_for_gitbook/Nielsen-NNDL/6.png)  

将公式（2）（3）代入公式（1）得：  
$$
f(x; W, c, w, b) = w^T\max(0, W^Tx+c)+b \tag{4}
$$
通过公式巧妙设置公式（4）中的参数，可以使得f(x)的结果与XOR完全相同。  

**问题是如何设置合理的参数？**
