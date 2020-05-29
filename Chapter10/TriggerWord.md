触发字 = trigger word = 唤醒词  

![](/assets/images/Chapter10/62.png)   
当检测到唤醒词时输出1，其它时间输出0  

缺点：构建了一个很不平衡的训练集  
解决方法：  
![](/assets/images/Chapter10/63.png)   
当检测到唤醒词时多输出几个1.