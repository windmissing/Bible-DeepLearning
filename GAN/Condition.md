# 根据Text生成Image

## 传统的监督学习方法

![](/assets/images/GAN/1.png)  

存在的问题：  
同一个输入在database中可能同时对应多种输出。  
训练的结果会输入各种可能的平均。所致结果很糊而不伦不类。  

## Conditional GAN的学习方法

生成网络G和鉴别网络D：  
![](/assets/images/GAN/2.png)  

存在的问题：  
G只要产生出足够高质量的图像，就可以骗过D。但这个高质量的图像很可能入输入的条件无关。  
高质量但与输入无关的图像也应该是false的图像。现在D无法鉴别这种情况。  
最后G会生成高质量但无视输入的图像。  

## 改进的Conditional GAN

改进方法：  
网络D的输入同时考虑G的输入c和G的输出x。  
D的判断G输出是否合理的标准有：  
（1）x是否高质量  
（2）c和x是否match

定义一个样本为{c, x}对，其中c是输入的条件，这里就是文本，x是生成的目标，这里就是指图像。  
- part 1 训练D
1. 从database中sample出m个positive样本。  
2. 随机生成m个noise图像，配上step1中positive样本的文字，得到的是m个负样本。  
3. 在从database中再sample出m个真实图像，配上step1中positive样本的文字，又得到的是m个负样本。  
4. 更新D的参数以最大化：  
V = 第1组样本的平均分 + （1-第2组样本的平均分） + （1-第3组样本的平均分）   
- part 2 训练G  
5. 生成m个随机图像z  
6. 从database中随机sample出m个文本c  
7. 更新G的参数，只(c, z)在D中的分数越高越好

网络D的两种结构：  
![](/assets/images/GAN/3.png)  

## Stack GAN

先生成小张图，再根据小张图生成大张图  

# 根据Image生成Image

例子：  
![](/assets/images/GAN/4.png)  

## 传统的监督学习方法

同text to Image  

## 条件GAN  

同text to Image  

## patch GAN

当要生成的image很大时，让D每次只检查image的一小块。  

# 其它应用

用于语音增强  
用于vedio生成  
