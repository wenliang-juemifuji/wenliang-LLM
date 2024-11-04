
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

