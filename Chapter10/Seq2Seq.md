翻译模型：Encoder + Decoder  
![](/assets/images/Chapter10/51.png)   
根据图像生成描述：AlexNet + Decoder  
![](/assets/images/Chapter10/52.png)   

# 翻译模型

## 翻译模型与语言模型的区别  

![](/assets/images/Chapter10/53.png)   
因此翻译模型又称条件语言模型。  


## 同一句输入可能得到不同的输出，怎么找到最好的输出？  

已知$p(y) = P(y^t|x, y^1, y^2, \cdots, y^{t-1})$
即怎么找到$y = (y^1, y^2, \cdots, y^{T_y})$，使得P(y|x)最大？  

### 贪心搜索 greedy search

每个时刻t都选择t时刻p(y)最大的y，直至输出<EOS>  
缺点：每一步都选择概率最高的y，但最后句子的整体概率不一定是最高的。  
例如：  
Jane is visiting Africa in September.   
Jane is going to visit Africa in September.  
英语中going比visiting常见，因此P(Jane is visiting|x) < P(Jane is going)，最后会得到第二句。  
实际上整个句子来说，第一句比第二句好。  

### 束搜索/定义搜索 beam search  

1. 根据$P(y^1|x)$选择最好的B个y1  
![](/assets/images/Chapter10/54.png)   
2. 对每个y1分别计算它与各种y2组合的概率。并找出其中概率最高的B组y1, y2  
![](/assets/images/Chapter10/55.png)   
3. 对B个y1y2串，用同样的方法找出B个概率最高的y1y2y3。  
4. 直至最后得到了<EOS>。  

当B=1时，beam search = greedy search