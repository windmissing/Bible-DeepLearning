已知z=f(y),f=g(x)

# 标量

x,y,z都是实数，f,g都是实数到实数的映射。  

$$
\frac{dz}{dx} = \frac{dz}{dy}\frac{dy}{dx} = f'(y)g'(x)
$$

# 向量

$$x \in R^m, y\in R^n, z\in R$$  
$$g: R^m \rightarrow R^n, f: R^n \rightarrow R$$

$$
\frac{\partial z}{\partial x_i} = \sum_j \frac{\partial z}{\partial y_j}\frac{\partial y_j}{\partial x_i}
$$
也可以写成：  
$$
\nabla_xz = (\frac{\partial y}{\partial x})^T\nabla_yz
$$
其中$$\frac{\partial y}{\partial x}$$为g的n*m的[Jacobian矩阵](https://windmising.gitbook.io/mathematics-basic-for-ml/xian-xing-dai-shu/special_matrix)。  

# 张量

[?]没看懂