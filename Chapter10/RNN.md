# 什么是RNN

CNN的局限性：
某个输入对应的输出，不止取决于输入本身，还取决于之前的输入。  
单纯针对当前输入做训练难以得到好的效果。  5'39''。  
解决方法：把上一次的信息也作为这一次的输入。  

补充：1 of N encoding，把单词转成向量。  
（1）图（2'12''左），再加一个order  
(2)word washing，图（3'34''右）

RNN训练方法：  
上一次输入得到的a(有的地方记作h)存在memory中，供下一次输入的计算过程使用。6'31''  
现在每个unit的输入向量是4维，其中2维来自样本特征，2维来自上一次的a。  
对于第一次输入，只有来自样本特征，没有上一次的a，此时要给a一个初值。  

使用RNN对字符串中单词分类的例子 12'21''
图中每个x为1 of N encoding得到的向量。  
对某个单词的理解不止基于这个单词，还基于它前面的内容。  


# RNN的 各种架构

1. deep RNN, 14'17''
2. Elam Network
t-1的h作为t的输入， 14'50''左  
3. Jordan Network  
t-1的y作为t的输入，14'50''右  
Jordan的效果优于Elam Network，因为作为输入的y比h好控制。  
4. 双向RNN  
正向算一次得到h1，反向算一次得到h2  
h1和h2的结果得到y，16'42''  
优点：network在产生output时看过的input更广。  
5. LSTM, long short-term memory，长短期记忆单元  

|Gate Name|作用|1|0|
|---|---|---|---|
|Input Gate|输入是否能写入memory|能写入|不能写入|
|Output Gate|是否将memory cell中的内容输入|能输出|不能输出|
|Forget Gate|是否保留当前memroy cell中的内容|保留|不保留|

Gate的值其实不是非0即1，而是(0,1)的值，表示打开程度。  
所有Gate的值都是network自己学到的，激活函数为sigmoid

因此LSTM有4个输入：
1. 要写入memory的值z
2. Input Gate的输入zi  
3. Output Gate的输入zo  
4. Forget Gate的输入zf  
LSTM有1个输出:memory中的新值。  

4个输入，4个输入都是由同一个input产生，分别使用4组参数而输入4个输入。这意味着它需要的参数是普通unit的4倍。    

问：长短期记忆单元到底是长期还是短期？  
答：本质上的短期记忆。因为相比于普通RNN的memory（没有记忆功能，t时刻的值直接覆盖t-1时刻的memory）来说，它的记忆是长的。  
long是short-term memory的形式词。  

图（24'09''）  
3个Gate的激活函数f通常是sigmoid函数。用于产生(0,1)的值，表示Gate的打开程度。  
c' = g(z)f(zi) + cf(zf)  
a = h(c')f(zo)    27'40''

xt向量的维数 = 下一层的LSTM unit个数。  
4组参数w,b   wi,bi  wo,bo   wf,bf  
分别与xt相乘得到z,zi,zo,zf  
z,zi,zo,zf向量与xt维数相同。  
向量中的每一项分别对应一个LSTM unit.  
实际上是整个向量一起算的，就像前馈网络中一次算一层一样。  
46'41'', 47'10''
42'04'', 44'59'', 45'51''. 46'12'', 46'40

# RNN的cost function  
相当于对每个时刻的输入作分类，而使用分类问题的代价函数，例如cross-entropy  
BPTT,backpropagation through time，算法用于计算RNN中参数的梯度。  
RNN的loss曲线可能是这样的：7'27''  
问：为什么RNN的loss会剧烈地抖动？  
答：RNN的error surface要么很平，要么很陡。平坦和陡峭的交界的地方构成悬崖。   
如果从悬崖下面update到悬崖上面，loss就会陡增。9'57''  
如果点正好落在悬崖上，梯度会突然非常大，然后参数就飞出去了。  
解决方法：截断Clipping，if gradient > threshold: gradient = threshold。  

问：为什么会有平坦和悬崖？  
答1：因为sigmoid unit?老师说不是这个原因。  
在前馈网络中，在hidden layer中使用sigmoid unit会导致这种情况。sigmoid unit->ReLU就能解决这个问题。  
答2：推导BPTT的公式可以分析出来原因。  
答3：直观分析。14'29''  
w的梯度= \delta w 对\delta C的影响。  
构造图中这样的一个简单的RNN，令unit的输入w为1，输出w为1，只在t0时刻有一个输入，值为1。观察unit的transition weight对C的影响。  
令w=1，则y1000=1  
w=1.01, y1000=20000 -- 悬崖
w=0.99, y1000=0 --- 平坦
w一但有影响，影响就是天崩地裂。这是因为同样的w在transition过程中反复使用。放大了它的作用。    
解决方法LSTM。  
LSTM只能解决梯度消失的问题，不能解决梯度爆炸的问题。悬崖仍然存在，通常将lr设置得比较小。  

问：为什么LSTM能解决梯度消失的问题？20'15''  
RNN和LSTM处理memory cell的操作不同。  
在RNN中，t时刻计算出的值直接存入memory cell中，覆盖t-1时刻的值。  
而在LSTM中，新memory = 旧memory * Gate + input。  
可见，只有Forget Gate不关闭，weight对memory的影响将永远存在。  
因此在实际训练过程中，应该将forget Gate设计为，在大多数情况下，forget gate都是开着的。  

# 其它解决梯度消失问题的unit

GRU --- Gate Recurrent Unit  
它是基于LSTM的改进：  
1. 只有2个Gate，将Input Gate与Forget Gate联动起来。当Input Gate打开时，Forget Gate自动关闭。    
2. 参数少了1/4，不容易发生过拟合。  
3. 最终效果差不多。  

Clockwise RNN  
SCRN 

# RNN的其它应用

## many to one  
输入 vector sequence，输出：one vector  
例如：  
文件的情绪分析，28'32''  
文本的关键词，30'00''  

## many to many，output is shorter

输入输出都是vector sequence，输出的sequence更短。  
例如：语音辨识，输入声音信号，输出字符序列    


