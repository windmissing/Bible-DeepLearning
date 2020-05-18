# Attention based mode

只读模型，57'36''  
可读可写模型，58'21''，又叫Neural Turing Machine  

## 阅读理解

### 文本QA
1. 通过Semantic分析，将每个句子转成一个向量  
2. 输入一个问题，由reading head controller决定，哪个句子与问题相关  
3. 把相关的句子读进模型，产生output
59'59''

### 视觉QA

1. 模型读入一张图  
2. 通过CNN，将每个小区域转成一个向量  
3. 输入一个问题，由reading head controller决定，哪个区域与问题相关  
4. 读入相关的区域，产生output
1:02'36''

### 语音QA

例如托福听力测试  
1:04'17''  
将语音识别、句义分析、attention的综合运用