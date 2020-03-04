[?]这里的表示貌似是个名字，是个什么东西？  

前面介绍的正则化大多是直接用于参数来惩罚复杂参数。  
这一节则通过惩罚激活神经元来惩罚复杂参数。  

[L1参数正则化](https://windmising.gitbook.io/bible-deeplearning/0introduction-1/0introduction/2l1)诱导稀疏参数是指“许多**参数(w)**为0”  
![](https://github.com/windmissing/Bible-DeepLearning/raw/master/Chapter7/images/3.png)  
但稀疏表示想要的是“许多**元素(h)**为0”  
![](https://github.com/windmissing/Bible-DeepLearning/raw/master/Chapter7/images/4.png)  

参考[L1参数正则化](https://windmising.gitbook.io/bible-deeplearning/0introduction-1/0introduction/2l1)让诱导参数为0的方法为“增加关于w的L1正则项”  
$$
\tilde J(w;X,y) = \alpha ||w||_1 + J(w;X,y)
$$

同理，要诱导元素h为0，就“增加关于h的正则项”。  
$$
\tilde J(w;X,y) = \alpha \Omega(h) + J(w;X,y)
$$
这个正则项也是L1正则项。  
$$
\Omega(h) = ||h||_1 = \sum_i|h_i|
$$
$$\Omega(h)$$也可以是其它正则项，例如[?]KL散度惩罚、[?]正则匹配追踪。  