还有很多其它激活函数用作中间层神经元，效果也很好。  
只有效果显著改进的激活函数才会被关注。  
这些介绍一些独特的：  
1. 没有激活函数。g(z) = z  
提供了一种减少网络中参数数量的有效方法。  
例如size=[in, out]，需要in*out个参数  
增加一层后为size=[in, mid, out]，需要(in+out)*mid个参数，如果mid非常小，需要的参数个数就会变少。  
2. [softmax](https://windmising.gitbook.io/bible-deeplearning/0introduction/0introduction/0introduction-1/3softmax)  
softmax在中间层使用时，可作为一种开关。  
[?]仅用于明确地学习操作内存的高级结构中，见10.12  
3. 书上还列了一些其他的，效果不好，不记了。  