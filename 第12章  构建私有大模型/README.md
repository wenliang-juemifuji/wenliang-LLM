# 基于 ChatGLM3-6B 构建私有大模型
## 环境搭建
将 ChatGLM2-6B 微调代码复制到 ChatGLM3-6B 目录
```text
cp -r ChatGLM2-6B/ptuning ChatGLM3-6B/
```
安装相关依赖：
```text
pip install -r requirements.txt
```
其中，transformers库版本推荐为4.30.2，torch推荐使用2.0及以上的版本，以获得最佳的推理性能。
## 代码调用
```python
>>> from transformers import AutoTokenizer, AutoModel
>>> tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm3-6b", trust_remote_code=True)
>>> model = AutoModel.from_pretrained("THUDM/chatglm3-6b", trust_remote_code=True, device='cuda')
>>> model = model.eval()
>>> response, history = model.chat(tokenizer, "你好", history=[])
>>> print(response)
你好👋!我是人工智能助手ChatGLM3-6B,很高兴见到你,欢迎问我任何问题。
>>> response, history = model.chat(tokenizer, "晚上睡不着应该怎么办", history=history)
>>> print(response)

晚上睡不着可能会让你感到焦虑或不舒服,但以下是一些可以帮助你入睡的方法:
	
制定规律的睡眠时间表:保持规律的睡眠时间表可以帮助你建立健康的睡眠习惯,使你更容易入睡。尽量在每天的相同时间上床,并在同一时间起床。
创造一个舒适的睡眠环境:确保睡眠环境舒适,安静,黑暗且温度适宜。可以使用舒适的床上用品,并保持房间通风。
放松身心:在睡前做些放松的活动,例如泡个热水澡,听些轻柔的音乐,阅读一些有趣的书籍等,有助于缓解紧张和焦虑,使你更容易入睡。
避免饮用含有咖啡因的饮料:咖啡因是一种刺激性物质,会影响你的睡眠质量。尽量避免在睡前饮用含有咖啡因的饮料,例如咖啡,茶和可乐。
避免在床上做与睡眠无关的事情:在床上做些与睡眠无关的事情,例如看电影,玩游戏或工作等,可能会干扰你的睡眠。
尝试呼吸技巧:深呼吸是一种放松技巧,可以帮助你缓解紧张和焦虑,使你更容易入睡。试着慢慢吸气,保持几秒钟,然后缓慢呼气。
如果这些方法无法帮助你入睡,你可以考虑咨询医生或睡眠专家,寻求进一步的建议。
```
## 网页版示例
启动一个基于Gradio的网页版示例：
```text
python web_demo.py
```

![](../images/基于Gradio启动ChatGLM3-6B.png)

除了上面的方式，还可以通过如下命令，启动一个基于Streamlit的网页版示例。
```text
streamlit run web_demo2.py
```

![](../images/基于Streamlit启动ChatGLM3-6B.png)

## 命令行示例
还可以通过命令行启动：
```text
python cli_demo.py
```

![](../images/基于命令行启动ChatGLM3-6B.png)

## 低成本部署
模型默认以FP16精度加载，需要约13 GB的显存。如果您的GPU显存不足，可以选择以量化方式加载模型，具体方法如下：
```python
model = AutoModel.from_pretrained("THUDM/chatglm3-6b",trust_remote_code=True).quantize(4)
```
也可以在CPU上运行模型，但是推理速度会慢很多。具体方法如下（需要至少32 GB的内存）：
```python
model = AutoModel.from_pretrained("THUDM/chatglm3-6b", trust_remote_code=True).float()
```
如果您的Mac使用了Apple Silicon或AMD GPU，可以使用MPS后端在GPU上运行ChatGLM3-6B。
```python
model = AutoModel.from_pretrained("your local path", trust_remote_code=True).to('mps')
```
如果您有多张GPU，但是每张GPU的显存都不够加载完整的模型，您可以使用模型并行的方式，将模型分配到多张GPU上。
```python
from utils import load_model_on_gpus
model = load_model_on_gpus("THUDM/chatglm3-6b", num_gpus=2)
```

## 微调 ChatGLM3-6B
### 数据准备
下载ADGEN数据集，这是一个用于生成广告文案的数据集。ADGEN数据集的任务是根据输入的商品信息生成一段吸引人的广告词。把AdvertiseGen文件夹里的数据分成训练集和验证集，分别保存为train.json和dev.json文件，数据的格式如下：

![](../images/ADGEN数据集的数据格式.png)

### 环境安装
要进行全参数微调，您需要先安装deepspeed，还需要安装一些有监督微调需要的包。
```text
pip install deepspeed
cd ptuning
pip install rouge_chinese nltk jieba datasets
```
修改微调代码中的相关参数
```text
vim ds_train_finetune.sh
LR=1e-5
MASTER_PORT=$(shuf -n 1 -i 10000-65535)
deepspeed --num_gpus=8 --master_port $MASTER_PORT main.py \
    --deepspeed deepspeed.json \
    --do_train \
    --preprocessing_num_workers 32 \
    --train_file AdvertiseGen/train.json \
    --test_file AdvertiseGen/dev.json \
    --prompt_column content \
    --response_column summary \
    --model_name_or_path ../models/chatglm3-6b \
    --output_dir output/adgen-chatglm3-6b-ft \
    --overwrite_output_dir \
    --max_source_length 512 \
    --max_target_length 512 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --predict_with_generate \
    --logging_steps 10 \
    --save_steps 1000 \
    --learning_rate $LR \
    --fp16
```
各个参数的含义如下表所示：
| 参数 | 参数含义 |
| --- | --- |
| num_gpus | 使用的GPU数量 |
| deepspeed | deepspeed的配置文件 |
| preprocessing_num_workers | 数据预处理的线程数量 |
| train_file | 训练数据文件的路径 |
| test_file | 测试数据文件的路径 |
| prompt_column | 输入列的名称 |
| response_column | 输出列的名称 |
| model_name_or_path | 模型名称或路径 |
| output_dir | 输出模型参数的文件路径 |
| max_source_length | 最大输入长度 |
| max_target_length | 最大输出长度 |
| per_device_train_batch_size | 每个设备的训练批次大小 |
| per_device_eval_batch_size | 每个设备的评估批次大小 |
| gradient_accumulation_steps | 梯度累计步数 |
| predict_with_generate | 是否使用生成模式进行预测 |
| logging_steps | 记录日志的步数 |
| save_steps | 保存模型的步数 |
| learning_rate | 学习率 |
| fp16 | 是否使用半精度浮点数进行训练 |

执行下面命令，训练模型：
```text
bash ds_train_finetune.sh
```
当运行代码时，会遇到一个错误提示：ChatGLMTokenizer类没有build_prompt方法。这是因为ChatGLM3-6B的ChatGLMTokenizer类没有实现这个方法。要解决这个问题，您可以参考ChatGLM2-6B中ChatGLMTokenizer类的build_prompt方法，按照相同的逻辑编写代码。
```python
vim ../models/chatglm3-6b/tokenization_chatglm.py
# 在ChatGLMTokenizer类中实现build_prompt方法
def build_prompt(self, query, history=None):
    if history is None:
	history = []
	prompt = ""
	for i, (old_query, response) in enumerate(history):
	    prompt += "[Round {}]\n\n问：{}\n\n答：{}\n\n".format(
		    i + 1, old_query, response)
	prompt += "[Round {}]\n\n问：{}\n\n答：".format(len(history) + 1, query)
    return prompt
```
### 模型微调
在Tokenizer类中添加了build_prompt方法的代码后，继续执行bash ds_train_finetune.sh命令。程序就可以正常运行了。
模型微调完成后，./output/adgen-chatglm3-6b-ft目录下会生成相应的文件，包含模型的参数文件和各种配置文件。
```text
tree ./output/adgen-chatglm3-6b-ft
├── all_results.json
├── checkpoint-1000
│   ├── config.json
│   ├── configuration_chatglm.py
│   ├── generation_config.json
│   ├── global_step1000
│   │   ├── mp_rank_00_model_states.pt
│   │   ├── zero_pp_rank_0_mp_rank_00_optim_states.pt
│   │   ├── zero_pp_rank_1_mp_rank_00_optim_states.pt
│   │   ├── zero_pp_rank_2_mp_rank_00_optim_states.pt
│   │   ├── zero_pp_rank_3_mp_rank_00_optim_states.pt
│   │   ├── zero_pp_rank_4_mp_rank_00_optim_states.pt
│   │   ├── zero_pp_rank_5_mp_rank_00_optim_states.pt
│   │   ├── zero_pp_rank_6_mp_rank_00_optim_states.pt
│   │   └── zero_pp_rank_7_mp_rank_00_optim_states.pt
│   ├── ice_text.model
│   ├── latest
│   ├── modeling_chatglm.py
│   ├── pytorch_model-00001-of-00002.bin
│   ├── pytorch_model-00002-of-00002.bin
│   ├── pytorch_model.bin.index.json
│   ├── quantization.py
│   ├── rng_state_0.pth
│   ├── rng_state_1.pth
│   ├── rng_state_2.pth
│   ├── rng_state_3.pth
│   ├── rng_state_4.pth
│   ├── rng_state_5.pth
│   ├── rng_state_6.pth
│   ├── rng_state_7.pth
│   ├── special_tokens_map.json
│   ├── tokenization_chatglm.py
│   ├── tokenizer_config.json
│   ├── trainer_state.json
│   ├── training_args.bin
│   └── zero_to_fp32.py
├── trainer_state.json
└── train_results.json
```
### 模型部署
执行streamlit run web_demo2.py命令来启动新模型。

![](../images/微调之后的模型.png)

