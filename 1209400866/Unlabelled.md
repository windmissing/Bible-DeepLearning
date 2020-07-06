data中大部分是normal，有少量的anomaly。但都是unlabelled，不哪个是anomaly。  
为什么跳过Case 2？因为实际场景都case 3，即使你认为是data都是clear的，但也有可能其实是polluted，只是你不知道而已。  

# 方法一：概率统计

1. 定义[概率密度函数](https://windmissing.github.io/mathematics_basic_for_ML/Probability/probability_distribution.html)为$f_\theta(x)$  
这个函数由参数$\theta$决定f(x)的形状。  
$\theta$可以是一个值或一个向量，是向量代表有多个参数。  
$\theta$未知，要根据x求出$\theta$，从而决定了f(x)的形状。  
$f_\theta(x)$通常使用[多维高斯分布](https://windmissing.github.io/mathematics_basic_for_ML/Probability/distribution.html)。  
2. 定义对数似然函数  
$$
L(\theta) = \log \left[ f_\theta(x^1)f_\theta(x^2)\cdots f_\theta(x^N)   \right]
$$

3. 求得到最大对数似然的$\theta$  
$$
\theta^* = \arg\max_\theta L(\theta)
$$

# 方法二：auto-encoder

[auto-encoder](https://windmissing.github.io/Bible-DeepLearning/Chapter7/AutoEncoder.html)  

把x转成code再还原成$\hat x$  
x与$\hat x$越接近，说明x越正常  

# 其它方法

one-class SVM, isolated forest