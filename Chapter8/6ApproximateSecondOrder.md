$$
J(\theta) = \frac{1}{m}\sum_{i=1}^mL(f(x^{(i)};\theta), y^{(i)}) + 正则项
$$

# 8.6.1 牛顿法
$$
J(\theta) \approx = J(\theta_0) + (\theta - \theta_0)^T\nabla_\theta J(\theta_0) + \frac{1}{2}(\theta - \theta_0)^TH(\theta-\theta_0) \\
\theta * = \theta_0 - H^{-1}\nabla_\theta J(\theta_0)
$$
直接跳到极小值  
只要H是正定的，就能正常迭代。  
H非正定时牛顿法会出错，解决方法：正则化，即H+aI  
$$H^-1$$的计算量非常大。  

