# 预处理  

音频数据的常见预处理步骤：  
1. 生成声谱图  
2. 伪空白输出  

# 音位 phonemes

音位是语言学家定义的发音的基本单位。  
在以前语音识别模型中：  
音频片段 --> 音位 --> transcript  
在现在的语音识别模型（end-to-end）中：  
音频片段 --> transcript

# 注意力模型做语音识别

![](/assets/images/Chapter10/61.png)   

# CTC cost做语音识别

CTC = Connectionist Temporal Classification  
basic rule: 将空白符之间的重复字符折叠起来。  