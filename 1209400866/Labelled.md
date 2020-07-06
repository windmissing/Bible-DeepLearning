假设数据都是labelled data。  
label包含data是normal还是anomaly。  
如果是normal，具体又是normal中的哪一类。  


# 训练

![](/assets/images/1209400866/1.png)  

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

将数据制作成[confusion matrix](https://windmising.gitbook.io/liu-yu-bo-play-with-machine-learning/10-1/10-1)，例如：  
||Anomaly|Normal|
|---|---|---|
|Detected| 1 | 1 |
|Not Detected| 4 | 99 |

左下角的数据为“异常data被判断为正常”的错误，又叫“missing”  
右上角的数据为“正常data被判断为异常”的错误 ，又叫“false alarm”  
$\lambda$取不同的值会得到不同的confusion matrix，哪种模型更好取决于对不同错误的cost的定义。  