# 定义  

||||
|---|---|---|
|损失函数|$$L(\hat y, y)$$||
|总代价|J|$$J= L(\hat y, y) + \lambda \Omega(\theta)$$|
|正则项|$$\Omega(\theta)$$|$$\theta$$包含所有参数（权重和偏置）|
|模型的权重矩阵|$$W^l$$||
|模型的偏置参数|$$b^l$$||
|程序的输出|x||
|目标输出|y||
|实际输出|$$\hat y$$||
|网络的深度|L|
|某一层的输出|$$h^l$$|同时也是下一层的输入。<br>书上用的是h，我有时候会写成a|
|某一层的加权输入|$$z^l$$|书上用的是a，我更喜欢用z|

# 算法6.3



算法6.3通过正向传播计算每一层的unit的加权输入z和激活输出h。  

```python
h_0 = x
for l = 1,...,L do:
    z_k = b_k + W_k.dot(h_(k-1))
    h_k = f(z_k)
end for
y_hat = h_L
J = L(y_hat, y) + lambda * omega(theta)
```

# 算法6.4

算法6.4通过反向传播计算每一层的unit的h的偏导、z的偏导、w的偏导、b的偏导。  

## 定义
*书上的g有两个用处，为了区分，我把它的两个用处分别用gh和gz*
gh为损失函数$$L(\hat y, y)$$对输出$$h^l$$的偏导。  
gz为损失函数$$L(\hat y, y)$$对加权输入$$z^l$$的偏导。  
w的偏导为总代价J对权重矩阵$$W^l$$的偏导。  
b的偏导为总代价J对偏置参数|$$b^l$$的偏导。  

## 根据定义计算第L层的情况

第L层的特殊在于$$h^L = \hat y$$  

$$
\begin{eqnarray}
gh^L & = & \frac{\partial L(\hat y, y)}{\partial h^L}
 = \nabla_{\hat y}L(\hat y, y)\\
gz^L & = & \frac{\partial L(\hat y, y)}{\partial z^L}
 = \frac{\partial L(\hat y, y)}{\partial h^L}\frac{\partial h^L}{\partial z^L}
 = gh^L \bigodot f'(z^L) \\
\nabla_{W^L}J & = & \frac{\partial L(\hat y, y)}{\partial W^L} +  \frac{\partial \lambda \Omega(\theta)}{\partial W^L}
 = \frac{\partial L(\hat y, y)}{\partial z^L} \frac{\partial z^L}{\partial W^L}+ \frac{\partial \lambda \Omega(\theta)}{\partial W^L} \\
 & = & gz^L (x^L)^T + \lambda \nabla_{W^L}\Omega(\theta)
 = gz^L(h^{L-1})^T + \lambda \nabla_{W^L}\Omega(\theta) \\
\nabla_{b^L}J & = & \frac{\partial L(\hat y, y)}{\partial b^L} +  \frac{\partial \lambda \Omega(\theta)}{\partial b^L} = \frac{\partial L(\hat y, y)}{\partial z^L} \frac{\partial z^L}{\partial b^L}+ \frac{\partial \lambda \Omega(\theta)}{\partial b^L} \\
 & = & gz^L + \lambda \nabla_{b^L}\Omega(\theta)
\end{eqnarray}
$$

## 根据定义计算第l层的情况

```for l = L-1, ..., 1 do:```  
$$
\begin{eqnarray}
gh^l & = & \frac{\partial L(\hat y, y)}{\partial h^l}
= \sum_i \frac{\partial L(\hat y, y)}{\partial z^{l+1}_i}\frac{\partial z^{l+1}_i}{\partial h^l}
= \sum_i \frac{\partial L(\hat y, y)}{\partial z^{l+1}_i}\frac{\partial z^{l+1}_i}{\partial x^{l+1}} \\
& = &(W^{l+1})^Tg \\
gz^l & = & 
\frac{\partial L(\hat y, y)}{\partial z^l}
= \frac{\partial L(\hat y, y)}{\partial h^{l}_i}\frac{\partial h^{l}}{\partial z^l} = gh^l \bigodot f'(z^l) \\
\nabla_{W^l}J & = & gz^l(h^{l-1})^T + \lambda \nabla_{W^l}\Omega(\theta) \\
\nabla_{b^l}J & = & gz^l + \lambda \nabla_{b^l}\Omega(\theta)
\end{eqnarray}
$$

公式中的$$f'(z^l)$$是主要的计算量。  
计算出来的是一个矩阵，称为[Jacobian矩阵](https://windmising.gitbook.io/mathematics-basic-for-ml/xian-xing-dai-shu/special_matrix)。  