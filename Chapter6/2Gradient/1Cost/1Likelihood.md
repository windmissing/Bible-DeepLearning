<!--sec data-title="something" data-id="section0" data-show=false ces-->

大多数神经网络使用[最大似然估计](https://windmising.gitbook.io/mathematics-basic-for-ml/gai-shuai-lun/likelihood)来训练，由此推导出来的代价函数为：  
$$
J(\theta) = -E_{X,y \sim \hat P_{data}}\log p_{model}(y|x)
$$

使用这种代价函数的优点是：  
1. 不用为每个模型设计代价函数，p(y|x)定了代价函数就这了。  
2. log的形式可以消除某些单元的指数效果，避免神经元饱和。  
3. [?]应用于实践中时通常没有最小值

<!--endsec-->
-------------------------------------------------------------------------------

大多数现代的神经网络使用最大似然来训练。
这意味着代价函数就是负的对数似然，它与训练数据和模型分布间的交叉熵等价。
这个代价函数表示为（6.12）  
$$
J(\theta) = -{\Bbb E}_{X,y \sim \hat{p}_\text{data}} \log p_\text{model} (y \mid x)
$$
{% reveal %}
```
\begin{equation}
J(\theta) = -{\Bbb E}_{X,y \sim \hat{p}_\text{data}} \log p_\text{model} (y \mid x) \tag {6.12}
\end{equation}
```
{% endreveal %}

> **[success]**
> [最大似然估计](https://windmising.gitbook.io/mathematics-basic-for-ml/gai-shuai-lun/likelihood)  

代价函数的具体形式随着模型而改变，取决于$$\log p_\text{model}$$的具体形式。
上述方程的展开形式通常会有一些项不依赖于模型的参数，我们可以舍去。
例如，正如我们在第5.1.1节中看到的，如果$$p_\text{model}(y\mid x) = N(y;f(x;\theta), I)$$，那么我们就重新得到了均方误差代价，  

> **[info]** 5.1.1 -> 5.5.1

{% reveal %}
```
\begin{equation}
J(\theta) = \frac{1}{2} {\Bbb E}_{X, y \sim  \hat{p}_\text{data}} || y - f(x; \theta) ||^2 + \text{const} \tag {6.13}
\end{equation}
```
{% endreveal %}

$$
J(\theta) = \frac{1}{2} {\Bbb E}_{X, y \sim  \hat{p}_\text{data}} || y - f(x; \theta) ||^2 + \text{const}
$$
> **[warning]**  
> [?]把$$p_\text{model}(y\mid x)$$代入公式（6.12），怎么会得到（6.13）？  
> [?]这一节本来在讲交叉熵代价函数，怎么到这里变成均方误差代价了？
