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

