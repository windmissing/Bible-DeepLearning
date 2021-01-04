LSTM架构中哪些部分是真正必须的？
还可以设计哪些其他成功架构允许网络动态地控制不同单元的时间尺度和遗忘行为？

最近关于门控RNN的工作给出了这些问题的某些答案，其单元也被称为门控循环单元或GRU{cite?}。
与LSTM的主要区别是，单个门控单元同时控制遗忘因子和更新状态单元的决定。  

> **[success]**  
> ![](/assets/images/Chapter10/37.png)  

更新公式如下：  
$$
\begin{aligned}
 h_i^{(t)} = u_i^{(t-1)} h_i^{(t-1)} + (1 - u_i^{(t-1)}) \sigma 
 \Big( b_i + \sum_j U_{i,j} x_j^{(t)} + \sum_j W_{i,j} r_j^{(t-1)} h_j^{(t-1)} \Big),
\end{aligned}
$$

> **[success]**  
> h = 更新门 * h + (1 - 更新门) * 输入  
> 更新门相当于遗忘门，（1-更新门）相当于输入门。  
> 更新门将遗忘门与输入门联动，相当于历史信息与当前信息的权衡。两者只有其一能较大的影响新的h。  
> 问：输出门去哪了？  
答：GRU中没有输出门。GRU和LSTM的一个区别就是，在GRU中，$a^t=C^t$。在LSTM中，$a^t$ = 输出门 * $C^t$  
> “输入”中包含了复位门。  
> 在其它操作中，通常把x与t-1的h合成一个大的向量。Gate对这个大的向量整体起作用。  
> 而复位门只对输入起作用。  
> 问：为什么要有复位门？  
答：复位门代表了$C^t$与$C^{t-1}$的相关度。  

其中$u$代表"更新"门，$r$表示"复位"门。
它们的值就如通常所定义的：  
$$
\begin{aligned}
 u_i^{(t)} = \sigma \Big( b_i^u + \sum_j U_{i,j}^u x_j^{(t)} + \sum_j W_{i,j}^u h_j^{(t)} \Big),
\end{aligned}
$$

和
$$
\begin{aligned}
 r_i^{(t)} = \sigma \Big( b_i^r + \sum_j U_{i,j}^r x_j^{(t)} + \sum_j W_{i,j}^r h_j^{(t)} \Big).
\end{aligned}
$$

复位和更新门能独立地"忽略"状态向量的一部分。  
> **[warning]** [?] 这一段看不懂   

更新门像条件渗漏累积器一样可以线性门控任意维度，从而选择将它复制（在sigmoid的一个极端）或完全由新的"目标状态"值（朝向渗漏累积器的收敛方向）替换并完全忽略它（在另一个极端）。
复位门控制当前状态中哪些部分用于计算下一个目标状态，在过去状态和未来状态之间引入了附加的非线性效应。
> **[success]**  
> 更新门将遗忘门与输入门联动，相当于历史信息与当前信息的权衡。两者只有其一能较大的影响新的h。  
当u非常接近0时，$C^t$非常接近$C^{(t-1)}$，因此 C能保留很久以前的信息。  
> **GRU的效果**  
> 参数少了1/4，不容易发生过拟合。最终效果差不多。  
**Ng补充：GRU与LSTM的对比**：  
$$
\begin{aligned}
\text{GRU} && \text{LSTM} \\
\hat C^t = \text{tanh}(W_c[r*C^{t-1}, x^t] + b_c) && \hat C^t = \text{tanh}(W_c[a^{t-1}, x^t] + b_c)  \\
u = \sigma(W_u[C^{t-1}, x^t] + b_u) && u = \sigma(W_u[a^{t-1}, x^t] + b_u)  \\
r = \cdots && f = \sigma(W_f[a^{t-1}, x^t] + b_f) \\
&& o = \sigma(W_o[a^{t-1}, x^t] + b_o) \\
C^t = u * \hat C^t + (1-u)* C^{t-1} && C^t = u * \hat C^t + f * C^{t-1}  \\
a^t = C^t && a^t = o * C^t
\end{aligned}
$$

围绕这一主题可以设计更多的变种。
例如复位门（或遗忘门）的输出可以在多个隐藏单元间共享。
或者，全局门的乘积（覆盖一整组的单元，例如整一层）和一个局部门（每单元）可用于结合全局控制和局部控制。
然而，一些调查发现这些LSTM和GRU架构的变种，在广泛的任务中难以明显地同时击败这两个原始架构{cite?}。
{Greff-et-al-arxiv2015}发现其中的**关键因素是遗忘门**，而{Jozefowicz-et-al-ICML2015}发现向LSTM遗忘门加入1的偏置(由{Gers-et-al-2000}提倡)能让LSTM变得与已探索的最佳变种一样健壮。  
> **[success] 各种LSTM变种性能的比较**  
> ![](/assets/images/Chapter10/17.png)  
(1) std LSTM works well  
(2) 将forget gate和input gate联运，参数变少，平均性能没有下降  
(3) 去掉peephold，参数量增加，性能没有明显下降  
(4) forget gate和output gate对性能很重要。  