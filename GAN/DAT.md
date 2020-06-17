# 用于图像

训练集与测试集不match的场景：  
![](/assets/images/GAN/32.png)   

解决方法：  
用Generator从Image中抽feature  
feature满足同样的分布，它们是match的。  
网络结构：  
![](/assets/images/GAN/33.png)   
其中Domain Classifier相当于一个Discriminator  

# 用于语音

以语音seq2seq为例：  
![](/assets/images/GAN/34.png)   
一段音频包含了许多信息，例如发音信息，环境信息，语者信息。  
把不同的信息提取出来可以有不同的应用：  
发音信息 --- 语音识别，语者信息 --- 声纹比对  
提取方法：  
![](/assets/images/GAN/35.png)   
![](/assets/images/GAN/36.png)   
原理类似于DAT