![](../images/图3-1GPT系列大模型的发展历史.png)

# GPT 实现文本分类和生成
## GPT1 实现文本分类
代码文件 [gpt1.py](gpt1.py) 实现了 GPT1，[train_gpt1.py](train_gpt1.py) 实现了 GPT1 模型训练。运行 [train_gpt1.py](train_gpt1.py) 可以实现模型训练，并生成训练后的效果。
### 数据准备
我们从 [GitHub - BenDerPan/toutiao-text-classfication-dataset: 今日头条中文新闻（文本）分类数据集](https://link.zhihu.com/?target=https%3A//github.com/BenDerPan/toutiao-text-classfication-dataset) 网址里面下载数据，数据来自今日头条客户端。

数据格式如下：
```text
6552431613437805063_!_102_!_news_entertainment_!_谢娜为李浩菲澄清网络谣言，之后她的两个行为给自己加分_!_佟丽娅,网络谣言,快乐大本营,李浩菲,谢娜,观众们
```
每行为一条数据，以_!_分割的个字段，从前往后分别是：新闻ID、分类code（见下文）、分类名称（见下文）、新闻字符串（仅含标题）、新闻关键词。

分类code与名称：
```text
100 民生 故事 news_story
101 文化 文化 news_culture
102 娱乐 娱乐 news_entertainment
103 体育 体育 news_sports
104 财经 财经 news_finance
106 房产 房产 news_house
107 汽车 汽车 news_car
108 教育 教育 news_edu
109 科技 科技 news_tech
110 军事 军事 news_military
112 旅游 旅游 news_travel
113 国际 国际 news_world
114 证券 股票 stock
115 农业 三农 news_agriculture
116 电竞 游戏 news_game
```
整个数据集共有382688条，分布于15个分类中。
### 模型训练与测试
运行代码：
```python
python train_gpt1.py
```
输出训练过程中的日志：
```text
epoch 20, batch 0, loss:3.8556, loss_fine:0.5126, acc:0.8906
epoch 20, batch 1000, loss:3.6283, loss_fine:0.2713, acc:0.9259
epoch 20, batch 2000, loss:3.6260, loss_fine:0.2715, acc:0.9256
epoch 20, batch 3000, loss:3.6289, loss_fine:0.2736, acc:0.9248
epoch 20, batch 4000, loss:3.6265, loss_fine:0.2719, acc:0.9251
epoch 20, save model at ./checkpoint/train_cat/ckpt-10
```
同时，输出给定文本的分类结果：
```text
输入:《狂飙》结局后，张译终于发声了，剧中演员回应一辈子不想见张译
预测输出: 娱乐
==============================================================
输入:教育部下发新通知，将调整今年的高考方向，家长看完心态“崩”了
预测输出: 教育
==============================================================
输入:俄罗斯学会了，发射大批气球飞向乌克兰，乌军导弹快不够用了
预测输出: 军事
```






