条件序列生成，例如：文字转语音、翻译、Chatbot
可以用传统的seq2seq解决，也可以用GAN解决。  
以Chatbot为例，seq2seq的做法是：  
![](/assets/images/GAN/38.png)   
这种方法存在的问题：  
![](/assets/images/GAN/39.png)   
对于How are you这样的提问，通常的回归是I'm fine。如果回答Not bad也是可以的。但使用最大似然方法，机器会认为I'm John比Not bad更合适。  

# 条件序列生成的技术

## 技术一：Reinforcement Learning

### 复习一下[policy Gradient](https://windmissing.github.io/Bible-DeepLearning/Chapter7/Reinforce.html)

![](/assets/images/GAN/40.png)   
$$
\bar R_\theta = \sum_c P(c)\sum_x R(c, x)P_\theta(x|c)
$$

公式中的符号定义如下：  
$\theta$：参数，在这些为固定值  
$\bar R_\theta$：reward的期望  
$\sum_c$： 遍历所有可能的condition  
P(c)：某个condition出现的几率  
$\sum_x$：遍历所有可能的action  
R(c, x)：condition条件下某个action的reward  
$P_\theta(x|c)$：given $\theta$和c时回应为x的几率  
目标：调$\theta^*$，让$\bar R_\theta$最大:  
$$
\begin{aligned}
\bar R_\theta = E_{c\in P(c), x \in P(\theta|c)}[R(c,x)]   && (1)
\end{aligned}
$$

公式（1）中的$c\in P(c), x \in P(\theta|c)$无法穷举所有的以(c, x)，因此用sample的(c1, x1), ..., (cm, xm)代替：  
$$
\begin{aligned}
\bar R_\theta \approx \frac{1}{N}\sum_i R(c^i,x^i)   && (2)
\end{aligned}
$$

公式（2）中已经没有了参数$\theta$，$\theta$完全体现在了sample的数据中了。  
下一步工作是计算$\bar R_\theta$对$\theta$偏导，这种情况就没法做偏导了。  
解决方法：  
先对公式（1）求偏导，然后再代入sample：  
$$
\begin{aligned}
\bar R_\theta = E_{c\in P(c), x \in P(\theta|c)}[R(c,x)\nabla\log P_\theta(x|c)]   && (3)
\end{aligned}
$$

sample代入公式（3）得：  
$$
\begin{aligned}
\bar R_\theta \approx \frac{1}{N}\sum_iR(c^i,x^i)\nabla\log P_\theta(x^i|c^i)   && (4)
\end{aligned}
$$

更新效果：  
$$
\theta \leftarrow \theta  + \eta\nabla\bar R_\theta
$$

如果$R(c^i,x^i) > 0$，则$P_\theta(x^i|c^i) \uparrow$  
如果$R(c^i,x^i) < 0$，则$P_\theta(x^i|c^i) \downarrow$  

**每次更新过$\theta$后要重新sample data**  

### 最大似然法 VS 增强学习法  

||Maximum Likelihood|Reinforement Learning|
|---|---|---|
|目标函数<br>Objective Function|$\frac{1}{N}\sum_i\log P_\theta(x^i\mid c^i)$|$\frac{1}{N}\sum_i R(c^i, x^i)\log P_\theta(x^i\mid c^i)$|
|Gradient|$\frac{1}{N}\sum_i\nabla\log P_\theta(x^i\mid c^i)$|$\frac{1}{N}\sum_i R(c^i, x^i)\nabla\log P_\theta(x^i\mid c^i)$|
|Training Data|已有的标记数据|每次迭代之后sample的数据|

## 技术二：GAN

RL是由人给feedback，GAN则是由D给feedback   
![](/assets/images/GAN/41.png)   
D的output就是reward。  

训练数据集：正确的(c, x)  

训练步骤：  
1. 初始化G（chatbot）和D  
2. 从database sample出正确的(c, x)  
3. 从database sample出c'，计算$\tilde x = G(c')$  
4. 更新D，使得$(c, x)\uparrow$，$(c', \tilde x)\downarrow$  
5. 更新G（chatbot），使得$(c', \tilde x)\uparrow$   
其中G是一个seq2seq：  
![](/assets/images/GAN/42.png)   
这里会有一个问题：此过程包含sampling，无法微分  
解决方法：  
1. Gunbel-softmax，一个数学trick  
2. Continuous Input for Doscriminator，避开sampling，把distribution交给D。此方法需要结合WGAN。    
![](/assets/images/GAN/43.png)   
3. Reinforcement Learning，把R换成D  
![](/assets/images/GAN/44.png)   
$$
\begin{aligned}
nabla \bar R_\theta \approx \frac{1}{N}D(c^i, x^i)\nabla \log P_\theta(x^i|c^i)  &&  (5)\\
\log P_{\theta}(c^i, x^i) = \log P(x_1^i|c^i) + \log P(x_2^i|c^i, x_1^i) + \log P(x_3^i|c^i, x_1^i, x_2^i)   && (6)
\end{aligned}
$$

假如提问：ci = What's your name?  
回答：xi = I don't known.  
这个回答不合适，给了negative的reward，因此公式（6）中的每一项中的概率都要下降。  
但其实第一项$P(x_1^i=I|c^i)$不应该下降，这个句子回答可以以I开关。  
理论上，这不是一个问题，因为只要sample的次数足够多，会出现“I am John”这样的case把$P(x_1^i=I|c^i)$拉起来。  
但实际上，这是一个问题，因为sample的次数永远都不会足够多。  
解决方法：  
$$
\nabla \bar R_\theta \approx \frac{1}{N}\sum_i\sum_t(Q(c^i, x_{1:t}^i)-b)\nabla \log P_\theta(x_t^i|c^i, x_{1:t}^i)
$$

其中$Q(c^i, x_{1:t}^i)-b$为对每个timestep做evaluation。  

### MLE VS GAN

MLE：喜欢回答I'm sorry或I' don't known这样通用的句子。这种回答对应于MLE图像生成算法中的模糊影像。  
GAN：会生成更长更复杂的句子。但不一定更好。  

# 条件序列生成的应用

## 应用一：Text Style Transfer

例如：  
音频：男声转女声  
文本：positive转negative  

### 方法一：直接transfer
![](/assets/images/GAN/45.png)   

### 方法二： projection to common space
![](/assets/images/GAN/46.png)   

## 应用二：摘要提取

### 方法一

输入一篇文章，判断每个句子的重要性。  
把重要的句子连起来就是这篇文章的摘要。  
缺点：句子拼凑起来的摘要不好，要自己产生句子。  

### 方法二：seq2seq

输入大量的（文章，摘要）对，训练一个seq2seq  
缺点：需要大量的labelled data，百万级别  

### 方法三：风格迁移

把文章和摘要看成是两种风格(domain)的文本。  
只需要准备一堆文章和一堆摘要，而不需要文章和摘要有什么关系。  
![](/assets/images/GAN/47.png)   
D用于保证生成出来的东西是摘要。  
R用于保证生成出来的东西与输入有关。  

Note：  
先用非监督学习训练，再用labelled data做fine tune。  
只需要少量的labelled data就可以得到与监督学习一样的效果。  

## 应用三：机器翻译

把不同的语音视为不同的domain  
也可以用于语音识别，把语音和文字视为不同的domain