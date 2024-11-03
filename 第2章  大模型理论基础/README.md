# 使用Transformer实现机器翻译
## 环境安装
```python
pip install tensorflow
```
## Transformer实现
代码文件 [transformer.py](transformer.py) 实现了 Transformer。
## 使用 Transformer 实现葡萄牙语翻译为英文
代码文件 [trans_port_to_eng.py](trans_port_to_eng.py) 实现了葡萄牙语翻译为英文。\
运行命令：
```python
python trans_port_to_eng.py
```
可以看到以下输出：
```python
输入: este é um problema que temos que resolver.
预测输出: this is a problem that we have to deal with .
真实输出: this is a problem we have to solve .
```
模型训练过程中的loss曲线：
![](../images/图2-1训练曲线.png)

## 使用 Transformer 实现英译汉


![](../images/transformer.png)

