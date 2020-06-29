# 什么是异常侦测

训练数据：  
- normal data： ${x^1, x^2, ..., x^N}$  
- anomaly data：${\tilde x^1, \tilde x^2, ..., \tilde x^N}$  

输入:x  
输出：x与训练集是否相似 normal/anomaly  

# 应用

欺诈侦测（盗刷卡）、网络入侵侦测、癌细胞侦测

# 训练方法

是否能当成是二分类问题来训练？  
答：不能，因为  
1. 除了normal都是anomaly，不能把各种anomaly归成一类，也很难穷举其类别  
2. 很多情况下，不容易收集anomaly的资料  

应该怎样学习？根据data分为三种情况：  
1. labelled data   --- open-set recognition  
2. unlabelled data && all data normal  
3. unlabelled data && 有少量anomaly data，但不知道是哪些  