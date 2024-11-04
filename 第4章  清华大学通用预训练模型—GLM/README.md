![](../images/图4-1GLM系列大模型发展时间线.png)

# 全参数微调 GLM
## 环境搭建
下表列出了 GPU 配置的相关环境（备注：表中是 ChatGLM-6B 全参数微调的配置，如果是 LORA 微调，单卡 A100 40GB 就可以。）。

| 配置项 | 参数 |
| --- | --- |
| 操作系统 | CentOS 7 | 
| GPU版本 | 8卡A100 80GB GPUs | 
| Python版本 | >=3.10 |
| NIVIDIA驱动程序版本 | 515.65.01 |
| CUDA工具包 | 11.7 |
| NCCL | nccl_2.14.3-1+cuda11.7 |
| cuDNN | 8.8.1.3_cuda 11 |

安装CUDA
```text
vim ~/.bashrc
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```
安装cuDNN
```text
sudo cp cudnn-linux-x86_64-8.8.0.121_cuda11-archive/include/cudnn*.h /usr/local/cuda/include/
sudo cp cudnn-linux-x86_64-8.8.0.121_cuda11-archive/lib/libcudnn* /usr/local/cuda/lib64/
sudo chmod a+r /usr/local/cuda/include/cudnn*.h /usr/local/cuda/lib64/libcudnn*
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```
安装PyTorch
```text
vim requirements.txt
protobuf>=3.19.5,<3.20.1
transformers>=4.26.1
icetk
cpm_kernels
gradio
pip install —user -r requirements.txt
pip install —user torch==1.10.0+cu111 torchvision==0.11.0+cu111 torchaudio==0.10.0 -f torch-1.10.0+cu111-cp39-cp39-linux_x86_ 64.whl -f torchvision-0.11.0+cu111-cp39-cp39-linux_x86_64.whl -f torchaudio-0.10.0+cu111-cp39-cp39- linux_x86_64.whl
```
## 全参数微调 ChatGLM-6B
下载ChatGLM-6B微调代码
```text
git clone https://github.com/THUDM/ChatGLM-6B.git .
cd ChatGLM-6B
cd ptuning
```
代码下载完毕后，接下来下载微调数据集，把文件保存到../data/目录下。
```text
mkdir ./data/; cd ./data/
wget https://huggingface.co/datasets/BelleGroup/school_math_0.25M/resolve/main/school_math_0.25M.json .
sed –n ’1,10000p’ school_math_0.25M.json > dev.json
sed –n ’100001,$p’ school_math_0.25M.json > train.json
```
数据准备好之后，下面要接着修改微调代码的相关参数：
```text
vim ds_train_finetune.sh
LR=1e-5
MASTER_PORT=$(shuf -n 1 -I 10000-65535)
deepspeed —num_gpus=8 —master_port $MASTER_PORT main.py \
    --deepspeed deepspeed.json \
    --do_train \
    --preprocessing_num_workers 32 \
    --train_file data/train.json \
    --test_file data/dev.json \
    --prompt_column content \
    --response_column summary \
    --cache_dir cache/batch16 \
    --model_name_or_path THUDM/chatglm-6b \
    --output_dir ./model/adgen-chatglm-6b-ft \
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
修改 [ChatLM-6B/ptuning/main.py](ChatLM-6B/ptuning/main.py) 文件中的num_train_epoch参数（默认num_train_epoch = 3）
```python
log_level = training_args.get_process_log_level()
logger.setLevel(log_level)
# datasets.utils.logging.set_verbosity(log_level)
transformers.utils.logging.set_verbosity(log_level)
transformers.utils.logging.enable_default_handler()
transformers.utils.logging.enable_explicit_format()

# Log on each process the small summary:

training_args.num_train_epochs = 1
logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: 
                        {training_args.n_gpu}" + f"distributed training: 
                        {bool(training_args.local_rank != -1)}, 16-bits training: 
                        {training_args.fp16}"
)
logger.info(f"Training/evaluation parameters {training_args}")
```
执行 sh ds_train_finetune.sh 命令微调模型。
```text
sh ds_train_finetune.sh
```
在训练过程中，可以观察到GPU的使用情况：

![](../images/图4-7.png)

如果想减少显存占用，可以适当减小参数batch_size、max_source_length和max_target_length的值，但是这样做会增加训练时间。

模型微调完成后，./model/目录下会生成相应的文件，包含模型的参数文件和各种配置文件。以pytorch_model开头的文件是模型的参数文件。
```text
tree ./model/adgen-chatglm-6b-ft
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
模型微调完成后，可以加载模型，测试模型效果。
```python
>> from transformers import AutoTokenizer, AutoModel
>> tokenizer = AutoTokenizer.from_pretrained("./model/adgen-chatglm-6b-ft"", trust_remote_code=True)
>> model = AutoModel.from_pretrained("./model/adgen-chatglm-6b-ft", trust_remote_code=True).half().cuda()
>> model = model.eval()
>> query = "艾德有2只狗、3只猫，鱼的数量是猫和狗加起来的两倍。艾德一共有多少只宠物？"
>> response, history = model.chat(tokenizer, query, history=[])
>> print(response)

首先,艾德有2只狗和3只猫,总共是2 + 3 = 5只宠物。
接下来,鱼的数量是猫和狗的两倍,因此鱼的数量是5 * 2 = 10只。
最后将狗、猫和鱼的数量相加,即5 + 10 = 15只。
因此一共有15只宠物。
```
## 全参数微调 GLM-10B
下载GLM-10B代码
```text
mkdir GLM-main
cd GLM-main
git clone https://github.com/THUDM/GLM.git .
```
下面是下载后的代码结构：
```text
tree ./GLM-main
├── arguments.py
├── change_mp.py
├── chinese_sentencepiece
│   ├── cog-pretrain.model
│   └── cog-pretrain.vocab
├── config_tasks
│   └── config_blocklm_10B_cnndm.json
├── configure_data.py
├── data_utils
│   ├── datasets.py
│   └── tokenization.py
├── finetune_glm.py
├── generate_samples.py
├── generation_utils.py
├── model
│   └── modeling_glm.py
├── pretrain_glm.py
├── scripts
│   └── ds_finetune_seq2seq.sh
└── tasks
    └── seq2seq
        └── dataset.py
```
上面的每个文件功能如下表所示。
| 文件名 | 作用 |
| --- | --- |
| arguments.py | 定义了一些通用的参数，如模型的大小、优化器的类型、、学习率的设置等 | 
| change_mp.py | 修改模型并行的数量，可以根据不同的硬件配置进行调整 | 
| chinese_sentencepiece | 存放了中文的分词模型和词表，用于对中文文本进行预处理 |
| configure_tasks | 存放了不同任务的配置文件，比如config_blocklm_10B_cnndm.json是用于摘要任务的配置文件，里面指定了数据集的路径、任务的类型、估指标等 |
| configure_data.py | 生成数据集的元数据，如词表大小、数据集大小、最大序列长度等 |
| data_utils | 包含一些数据处理的工具，例如datasets.py是用于加载和处理不同格式的数据集的，tokenization.py是用于对文本进行分词和编码的 |
| finetune_glm.py | 微调GLM-10B模型的主程序，可以根据不同任务的配置文件进行微调，并输出预测结果和评估结果 |
| generate_samples.py | 生成GLM-10B模型的样本的，可以根据给定的输入文本或者输入文件生成相应的输出文本或者输出文件 |
| model | 包含GLM-10B模型的定义和实现，modeling_glm.py是模型的主要代码，它定义了GLM-10B模型的结构和前向传播过程 |
| pretrain_glm.py | 预训练GLM-10B模型的主程序，可以根据给定的预训练数据集进行预训练，并保持模型参数和优化器状态 |
| scripts | 存放了一些运行脚本，比如ds_finetune_seq2seq.sh是用于在分布式环境下微调GLM-10B模型 |
| tasks | 存放了一些任务相关的代码，比如seq2seq文件夹包含序列到序列任务的数据加载和评估函数 |

要想并行训练GLM-10B模型，首先需要把模型切分成多个部分，然后用 [change_mp.py][GLM-main/change_mp.py] 文件中的函数进行调整。
```text
python change_mp.py ./model_file/glm-10b 8
ll ./model_file
```
模型切分之后，结构如下：
```text
cd ./model_file/glm-10b-MP8
tree .
├── mp_rank_00_model_states.pt
├── mp_rank_01_model_states.pt
├── mp_rank_02_model_states.pt
├── mp_rank_03_model_states.pt
├── mp_rank_04_model_states.pt
├── mp_rank_05_model_states.pt
├── mp_rank_06_model_states.pt
└── mp_rank_07_model_states.pt
```
模型的参数被分成了8个文件。为了训练模型，需要把数据转换成模型需要的格式。首先需要把 school_math_0.25M.json 文件中的问题 instruction 提取出来，作为 text 字段，保存在train.source 文件中。然后需要把答案 output 也提取出来，作为 text 字段，保存在 train.target 文件中。
```text
cd ./customization
tree .
├── test.source
├── test.target
├── train.source
├── train.target
├── val.source
└── val.target
```
训练数据有6个文件：train.source 和 train.target 是训练数据的问题和答案，test.source 和 test.target 是测试数据的问题和答案，val.source 和 val.target 是验证数据的问题和答案。
编辑 scripts/ds_finetune_seq2seq.sh 文件。其中，CHECKPOINT_PATH 指模型的路径，SAVE_PATH 指微调后的模型保存的路径，MP_SIZE 是指模型被切分成的文件个数。
```text
vim config_tasks/config_blocklm_10B_cnndm.json
{
  "train_micro_batch_size_per_gpu": 16,
  "gradient_accumulation_steps": 1,
  "steps_per_print": 50, 
  "gradient_clipping": 1.0,
  "zero_optimization": {
    "stage": 2,
    "contiguous_gradients": false,
    "overlap_comm": true,
    "reduce_scatter": true,
    "reduce_bucket_size": 5e7,
    "allgather_bucket_size": 5e7,
    "cpu_offload": true
  },  
  "zero_allow_untested_optimizer": true,
  "fp16": {
    "enabled": true,
    "loss_scale": 0,
    "loss_scale_window": 1000,
    "hysteresis": 2,
    "min_loss_scale": 1
  },  
  "optimizer": {
    "type": "Adam",
    "params": {
      "lr": 1e-5,
      "betas": [
        0.9,
        0.95
      ],  
      "eps": 1e-8,
      "weight_decay": 1e-2
    }   
  },  
  "activation_checkpointing": {
    "partition_activations": false,
    "contiguous_memory_optimization": false
  },  
  "wall_clock_breakdown": false
}
```
修改 config_tasks/model_blocklm_10B.sh 文件，这个文件用于定义分词器、任务类型等参数，需要把分词器改为中文分词器ChineseSPTokenizer，这样才能正确地处理中文文本。
