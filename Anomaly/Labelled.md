# 训练

![](/assets/images/Anomaly/1.png)  

- 训练集：全部都是normal data，$\hat y$是normal类别的分类   
把normal data当作多分类问题来训练，同时输出类别和对分类的置信度。  
- 验证集：样本有normal和anomaly的data，并有label来区别是normal还是anomaly。  
根据performance来决定$\lambda$以及其它超参数。  
$$
x = 
\begin{cases}
\text{normal} && , c(x) > \lambda   \\
\text{anomaly} && , c(x) \le \lambda
\end{cases}
$$

- 测试集：输入样本，根据公式判断normal/anomaly

# Evaluation

如何根据performance判断模型的好坏？是否能用准确率？  
答：不能。  
异常侦测问题中，异常数据可能非常少，属于极度偏斜的数据。这种情况下不适合使用正确率。  
