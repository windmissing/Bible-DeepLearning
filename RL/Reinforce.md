# background

## 深度学习一次迭代的三个步骤  

1. 环境 -> 机器：state  
2. 机器 -> 环境：action  
3. 环境 -> 机器：reward  

定义：
一轮迭代 = state -> action -> reward。  如果没有反馈，reward = 0。  
一episode = 一局游戏，有赢/输结果的。  
目标：maximize the expected cumulative reward per spisode。  

## 监督学习 VS 强化学习

监督学习：从数据(State, Action)学习，学习的好坏取决于数据(State, Action)的好坏，因此需要大量数据。  
强化学习：根据自己的(State, Action)经验学习，因此需要大量的经验。  

## 强化学习的难点  

1. reward delay  
2. 有些Action没有reward，甚至可能有牺牲。但它对帮助得到reward有重要贡献。  
3. 需要Machine探索未尝试过的行为。  

## 算法分类

1. policy based算法 --- 学actor  
2. value based算法 --- 学critic  
3. policy + value 算法 --- A3C算法  

Alpha GO = polocy based + value based + model based  
model based算法主要用于棋类游戏  

# 应用

## 应用于下棋
生成两个agent，互相对弈，以胜负作为reward。  

## 应用于Chat-Bot
生成两个agent，互相对话。  
另外训练一个NN用于判断talk的好坏，并给予reward。  

## 应用于电子游戏

Gym：https://gym.openai.com/  
Universe：https://openai.com/blog/universe/