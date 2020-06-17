1. 把input vector分成两个向量：c和z'  
2. 增加一个组件classifier  
classifier根据G生成的x来预测输入的c  
Generator + Classifier类似一个相反的auto-encoder  
3. Discriminator用于检测G的结果好不好  
![](/assets/images/GAN/28.png)   
C和D只有最后一部分不同，前面部分共享参数。  

加入Classifier的好处：  
C必须对x有clear的influence。  
z'代表纯随机的无法解释的部分。  