计算循环神经网络的梯度是容易的。
我们可以简单地将\sec?中的推广反向传播算法应用于展开的计算图，而不需要特殊化的算法。
由反向传播计算得到的梯度，并结合任何通用的基于梯度的技术就可以训练RNN。

为了获得BPTT算法行为的一些直观理解，我们举例说明如何通过BPTT计算上述RNN公式（\eqn?和\eqn?）的梯度。
计算图的节点包括参数$U,V,W, b$和$c$，以及以$t$为索引的节点序列$x^{(t)}, h^{(t)},o^{(t)}$和$L^{(t)}$。
对于每一个节点$N$，我们需要基于$N$后面的节点的梯度，递归地计算梯度$\nabla_{N} L$。
我们从紧接着最终损失的节点开始递归：  
$$
\begin{aligned}
 \frac{\partial L}{\partial L^{(t)}} = 1.
\end{aligned}
$$

> **[warning]** 这个公式不知道怎么推出来的？  

在这个导数中，我们假设输出$o^{(t)}$作为softmax函数的参数，我们可以从softmax函数可以获得关于输出概率的向量$\hat{y}$。
我们也假设损失是迄今为止给定了输入后的真实目标$y^{(t)}$的负对数似然。  
> **[success]**  
以下公式推导中：  
t代表任意一个时刻，不是某个具体的时刻。  
$\tau$代表当前时刻。  

对于所有$i,t$，关于时间步$t$输出的梯度$\nabla_{o^{(t)}} L$如下：
$$
\begin{aligned}
 (\nabla_{o^{(t)}} L)_i =  \frac{\partial L}{\partial o_i^{(t)}} 
 =  \frac{\partial L}{\partial L^{(t)}}  \frac{\partial L^{(t)}}{\partial o_i^{(t)}}  
 = \hat y_i^{(t)} - \mathbf{1}_{i,y^{(t)}}.
\end{aligned}
$$

我们从序列的末尾开始，反向进行计算。
在最后的时间步$\tau$, $h^{(\tau)}$只有$o^{(\tau)}$作为后续节点，因此这个梯度很简单：
$$
\begin{aligned}
 \nabla_{h^{(\tau)}} L = V^\top \nabla_{o^{(\tau)}} L.
\end{aligned}
$$

然后，我们可以从时刻$t=\tau-1$到$t=1$反向迭代， 通过时间反向传播梯度，注意$h^{(t)}(t < \tau)$同时具有$o^{(t)}$和$h^{(t+1)}$两个后续节点。
因此，它的梯度由下式计算
$$
\begin{aligned}
\nabla_{h^{(t)}} L = \Big( \frac{\partial h^{(t+1)}}{ \partial h^{(t)}}  \Big)^\top(\nabla_{h^{(t+1)}} L) + \Big( \frac{\partial o^{(t)}}{ \partial h^{(t)}}  \Big)^\top (\nabla_{o^{(t)}} L) \\
= W^\top (\nabla_{h^{(t+1)}} L) \text{diag} \Big( 1 - (h^{(t+1)})^2 \Big) + V^\top ( \nabla_{o^{(t)}} L ),
\end{aligned}
$$

> **[warning]**  
$$
\begin{aligned}
\Big( \frac{\partial h^{(t+1)}}{ \partial h^{(t)}}  \Big)^\top = W^\top  \\
(\nabla_{h^{(t+1)}} L) \text{还是} (\nabla_{h^{(t+1)}} L) \\
\text{diag} \Big( 1 - (h^{(t+1)})^2 \Big) \text{这是哪来的}?
\end{aligned}
$$

其中$\text{diag} \Big( 1 - (h^{(t+1)})^2 \Big) $ 表示包含元素$1 - (h_i^{(t+1)})^2$的对角矩阵。
这是关于时刻$t+1$与隐藏单元$i$关联的双曲正切的Jacobian。  
> **[warning]** 双曲正切的Jacobian?  

一旦获得了计算图内部节点的梯度，我们就可以得到关于参数节点的梯度。
因为参数在许多时间步共享，我们必须在表示这些变量的微积分操作时谨慎对待。
我们希望实现的等式使用\sec?中的{\tt bprop}方法计算计算图中单一边对梯度的贡献。
然而微积分中的$\nabla_{W} f$算子，计算$W$对于$f$的贡献时将计算图中的\emph{所有}边都考虑进去了。
为了消除这种歧义，我们定义只在$t$时刻使用的虚拟变量$W^{(t)}$作为$W$的副本。
然后，我们可以使用$\nabla_{W^{(t)}}$表示权重在时间步$t$对梯度的贡献。

使用这个表示，关于剩下参数的梯度可以由下式给出：
$$
\begin{aligned}
 \nabla_{c} L &=  \sum_t \Big( \frac{\partial o^{(t)}}{\partial c} \Big)^\top \nabla_{o^{(t)}} L 
 = \sum_t \nabla_{o^{(t)}} L ,\\
 \nabla_{b} L &= \sum_t \Big( \frac{\partial h^{(t)}}{\partial b^{(t)}} \Big)^\top \nabla_{h^{(t)}} L 
 = \sum_t \text{diag} \Big( 1 - \big( h^{(t)} \big)^2 \Big)  \nabla_{h^{(t)}} L  ,\\
 \nabla_{V} L &= \sum_t \sum_i \Big( \frac{\partial L} {\partial o_i^{(t)}}\Big) \nabla_{V} o_i^{(t)} 
 = \sum_t (\nabla_{o^{(t)}} L) h^{(t)^\top},\\
 \nabla_{W} L &= \sum_t \sum_i \Big( \frac{\partial L} {\partial h_i^{(t)}}\Big) 
 \nabla_{W^{(t)}} h_i^{(t)} \\
&= \sum_t \text{diag} \Big( 1 - \big( h^{(t)} \big)^2 \Big) ( \nabla_{h^{(t)}} L) h^{(t-1)^\top} ,\\
 \nabla_{U} L &= \sum_t \sum_i \Big( \frac{\partial L} {\partial h_i^{(t)}}\Big) 
 \nabla_{U^{(t)}} h_i^{(t)} \\
&= \sum_t \text{diag} \Big( 1 - \big( h^{(t)} \big)^2 \Big) ( \nabla_{h^{(t)}} L) x^{(t)^\top} ,
\end{aligned}
$$

因为计算图中定义的损失的任何参数都不是训练数据$x^{(t)}$的父节点，所以我们不需要计算关于它的梯度。  
> **[success]**  
![](/assets/images/Chapter10/5.jpg)  
参数包含U，W，V，b，c.  
（1）计算每条边上的梯度  
（2）令{L}=1  
（3）根据串行相乘并行相加的方法计算{U}, {W}, {V}, {b}, {c}  
最后得到与书上相同的结果。  
