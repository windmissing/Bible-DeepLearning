# passive defense 被动防御

不改变model，增加一个Anomaly Detection

## 增加一个filter，例如smoothing

![](/assets/images/1209400866/8.png)  

## Feature Squeeze

![](/assets/images/1209400866/9.png)  

## Randomization at Inference Phase

![](/assets/images/1209400866/10.png)  

## 缺点

如果防御机制泄漏，攻击仍会生效

# Proactive Defence 主动防御

Training a model that is robust to adversarial attack

## 找出漏洞，补起来

根据图像攻击算法，找到攻击图像，把攻击图像当作训练样本来训练  

## 缺点

训练时只能列举有限的攻击算法，换用算法攻击，仍然能攻破