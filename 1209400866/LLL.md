用同一个模型，持续地学习

# Knowledge Retention, but not intransigence

## Catastropic Forgetting

例如有task 1和task 2:  
![](/assets/images/1209400866/15.png)   
如果同时学task 1和task 2，两个都能学好：  
![](/assets/images/1209400866/17.png)   
如果先学task 1，再学task 2，task 1会忘掉，这种遗忘称为：  
![](/assets/images/1209400866/16.png)   

## Elastic Weight Consolidation (EWC)

Basic Idea：  
model中某些参数对prev task特别重要，训练新task时只改变不重要的参数。  
$$
L'(\theta) = L(\theta) + \lambda \sum_i b_i(\theta_i - \theta^b_i)^2
$$

其中：  
$\theta^b_i$：上一个Task的参数，在这个task中当作常数看待。  
$\theta_i$：这个Task要调的参数  
$b_i$：$\theta^b_i$的gurad，代表了$\theta^b_i$的重要性。  
第一项：学好新task  
第二项：与旧Task的参数差别不要太大。  
$b_i=0$：$\theta^b_i$无保护，$\theta_i$随意发挥  
$b_i=\inf$：$\theta^b_i$强保护，$\theta_i$必须等于$\theta^b_i$  

问：为什么会发生Catastropic Forgetting？  
答：  
![](/assets/images/1209400866/18.png)   
问：怎么定义bi?  
答：$\theta^b$肯定位于谷底。$\theta^b$的二次微分代表$\theta^b$在这个方向的谷底是宽是窄。  
![](/assets/images/1209400866/19.png)   
越平坦说明$\theta^b_i$的改动影响越小，越不需要保护。$b_i$就越小。  
## 为什么EWC有用？  
答：  
![](/assets/images/1209400866/20.png)   

## Generating Data

Multi-Task虽然可以解决LLL的问题，但缺点是需要存储大量的training data。  
可以训练一个能生成training data的NN，这样只存一个NN就可以了。  

# knowledge Transfer

问：为什么不“每个task各训练一个model”？  
答：1.不同task之间学到的东西无法迁移。2，存不下那么多模型。  

问：LLL和transfer learning有什么区别？  
答：TL只要求task 1的学习对Task 2有帮助。LLL进一步要求task 2的学习不会导致遗忘task 1，甚至希望能有助于task 1。  

## 评价LLL的好坏（Evaluation）

### 填表

![](/assets/images/1209400866/21.png)   

$R_{i,j}$：训练过task [1 .. i]之后，task j的测试效果。  
- i < j，$R_{i,j}$代表task i transfer到task j的效果。  
- i > j，$R_{i,j}$代表task j对task i的影响  

### 定义衡量指标

$$
\begin{aligned}
\text{Accuracy} = \frac{1}{T} \sum_{i=1}^T R_{T,i}   \\
\text{Backward Transfer} = \frac{1}{T-1}\sum_{i=1}^{T-1} (R_{T,i} - R_{i,i})   \\
\text{Forward Transfer} = \frac{1}{T-1}\sum_{i=2}^{T} (R_{i-1,i} - R_{0,i})
\end{aligned}
$$

## GEM --- Gradient Episodic Memory  

一种让backward为正的方法。  
正常情况下，Backward Transfer < 0 --- 遗忘  
GEN算法效果，Backward Transfer > 0 --- 触类旁通、知识迁移  
![](/assets/images/1209400866/22.png)   

1. 计算：  
g: current task的梯度方向  
gi：之前的task i的梯度方向  
2. 判断  
g * gi > 0 ==> g' = g  
g * gi < 0 ==> 调整g'，使g' * gi > 0 且 g和g'尽量接近 
3. 更新  
基于g'更新

# Model Expansion, but Parameter Efficiency

## Progressive Neural Networks

![](/assets/images/1209400866/23.png)   

## Expert Gate

![](/assets/images/1209400866/24.png)   

## Net2Net

![](/assets/images/1209400866/25.png)   

前两种方法每次有新的Task就要扩展模型，参数的增加速度与task成正比  
第三种只在当前模型难以提升时才增加结点  

# Curriculum Learning

还是上面例子中的task1和task 2  
![](/assets/images/1209400866/15.png)   
如果先学task 1，再学task 2，task 1会忘掉  
如果先学task 2，再学task 1，task 2不会忘掉  
![](/assets/images/1209400866/26.png)   
说明task的顺序很重要。怎样排序最合适？  

