![](../images/图5-2Llama2-Chat训练过程.png)

# Llama 微调
## 环境搭建
从GitHub上下载Llama 2的微调代码：
```text
git clone https://github.com/facebookresearch/llama-recipes .
cd llama-recipes-main
pip install –r requirements.txt
```
从HuggingFace上下载Llama 2的模型：
```python
>> import huggingface_hub
>> huggingface_hub.snapshot_download(
        "meta-llama/Llama-2-7b-hf",
        local_dir="./Llama-2-7b-hf",
        token="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"
    )
```
## 微调
下载 GuanacoDataset 数据集：
```text
wget https://guanaco-dataset.s3.amazonaws.com/guanaco_dataset.zip
unzip guanaco_dataset.zip
```
微调模型，读者可以根据自己的硬件条件选择单卡或者多卡微调。如果是单卡微调，可以使用如下命令：
```text
export CUDA_VISIBLE_DEVICES=0
python llama_finetuning.py  --use_peft --peft_method lora --quantization --model_name /patht_of_model_folder/7B --output_dir Path/to/save/PEFT/model
```
如果是多卡微调，可以使用如下命令：
```text
torchrun --nnodes 1 --nproc_per_node 4  llama_finetuning.py --enable_fsdp --use_peft --peft_method lora --model_name /patht_of_model_folder/7B --pure_bf16 --output_dir Path/to/save/PEFT/model
```
在线推理，可以使用 alpaca-lora 中提供的脚本 generate.py 进行推理，启动命令如下：
```text
python generate.py --base_model ./Llama-2-7b-hf --lora_weights ./lora
```

