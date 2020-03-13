动量的主要目的是解决两个问题：  
1. Hessian 矩阵的病态条件  
2. 随机梯度的方差。

更新规则：  
$$
v \leftarrow av-\epsilon\nabla_\theta\left(\frac{1}{m}\sum_{i=1}^mL\left(f(x^{(i)},\theta), y^{(i)}\right)\right)  \\
\theta \leftarrow \theta + v
$$
v代表速度，也相当于动量。  
$$a \in [0,1)$$决定了之前梯度的贡献的衰减有多快。  

