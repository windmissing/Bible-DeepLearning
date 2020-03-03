# Table of contents

* [Introduction](README.md)
* [第6章 深度前馈网络](Chapter6/0Introduction.md)
    * [6.1 例子：学习XOR](Chapter6/1Examples.md)
    * [6.2 基于梯度的学习](Chapter6/2Gradient/0Introduction.md)
        * [6.2.1 代价函数](Chapter6/2Gradient/1Cost/0Introduction.md)
            * [6.2.1.1] [使用最大似然学习条件分布](Chapter6/2Gradient/1Cost/1Likelihood.md)
            * 6.2.1.2 学习条件统计量 看不懂
        * [6.2.2 输出单元](Chapter6/2Gradient/2OutputUnit/0Introduction.md)
            * [6.2.2.1 用于高斯输出分布的线性神单元](Chapter6/2Gradient/2OutputUnit/1Linear.md)
            * [6.2.2.2 用于Bernoulli输出分布的sigmoid单元](Chapter6/2Gradient/2OutputUnit/2Sigmoid.md)
            * [6.2.2.3 用于Multinoulli输出分布的softmax单元](Chapter6/2Gradient/2OutputUnit/3Softmax.md)
            * [6.2.2.4 其他输出类型](Chapter6/2Gradient/2OutputUnit/4Other.md)
    * [6.3 隐藏单元](Chapter6/3Hidden/0Introduction.md)
        * [6.3.1 ReLU及其扩展](Chapter6/3Hidden/1ReLU.md)
        * [6.3.2 logistic sigmoid与双曲正切函数](Chapter6/3Hidden/2SigmoidTanh.md)
        * [6.3.3 其他隐藏单元](Chapter6/3Hidden/3Other.md)
    * [6.4 架构设计](Chapter6/4Architecture.md)
    * [6.5 反向传播和其他的微分算法](Chapter6/5Backprop/0Introduction.md)
        * [6.5.1 计算图](Chapter6/5Backprop/1ComputationalGraphs.md)
        * [6.5.2 微积分中的链式法则](Chapter6/5Backprop/2ChainRule.md)
        * [6.5.3 递归地使用链式法则来实现反向传播](Chapter6/5Backprop/3Recursively.md)
        * [6.5.4 全连接MLP中的反向传播计算](Chapter6/5Backprop/4FullyConnectedMLP.md)
        * 6.5.5 符号到符号的导数 没看懂
        * 6.5.6 一般化的反向传播 没看懂
        * [6.5.7 实例：用于MLP 训练的反向传播](Chapter6/5Backprop/7MLPTraining.md)
        * 6.5.8 - 6.5.10 全部看不懂
* [第七章深度学习中的正则化](Chapter7/0Introduction.md)
    * [7.1 参数范数惩罚](Chapter7/1ParameterNormPenalties/0Introduction.md)
        * [7.1.1 L2 参数正则化](Chapter7/1ParameterNormPenalties/1L2.md)
        * [7.1.2 L1 参数正则化](Chapter7/1ParameterNormPenalties/2L1.md)
    * [7.2 作为约束的范数惩罚](Chapter7/2ConstrainedOptimization.md)

