import pandas as pd
import numpy as np
import random
import os
import sys
from numpy import arange
import math
import sys,base64,urllib,re
import multiprocessing
from sklearn.metrics import roc_auc_score
from sklearn.metrics import ndcg_score
import warnings 
from optparse import OptionParser
import logging
import logging.config
import time
from sklearn.preprocessing import normalize
# 安装tfds pip install tfds-nightly==1.0.2.dev201904090105
import tensorflow_datasets as tfds
import tensorflow as tf
import tensorflow.keras.layers as layers
import time
import numpy as np
import jieba
from tqdm import tqdm
from gpt2 import GPT2
print(tf.__version__)

MAX_LENGTH = 100
BUFFER_SIZE = 20000
BATCH_SIZE = 64

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()

        self.d_model = tf.cast(d_model, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


def evaluate(inp_sentence):
    start_token = [tokenizer_title.vocab_size]
    end_token = [tokenizer_title.vocab_size + 1]
    
    # 增加开始和结束标记
    inp_sentence = start_token + tokenizer_title.encode(inp_sentence) + end_token
    encoder_input = tf.expand_dims(inp_sentence, 0)

    decoder_input = [tokenizer_title.vocab_size]
    output = tf.expand_dims(decoder_input, 0)

    for i in range(MAX_LENGTH):
        
        combined_mask = create_mask(encoder_input)
        
        predictions, _ = gpt2(encoder_input, training=False, look_ahead_mask=combined_mask)

        # 从 seq_len 维度选择最后一个词
        predictions = predictions[: ,-1:, :]  # (batch_size, 1, vocab_size)

        predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

        # 如果 predicted_id 等于结束标记，就返回结果
        if predicted_id == tokenizer_title.vocab_size + 1:
            return tf.squeeze(encoder_input, axis=0)

        # 连接 predicted_id 与输出，作为解码器的输入传递到解码器。
        encoder_input = tf.concat([encoder_input, predicted_id], axis=-1)
        output = tf.concat([output, predicted_id], axis=-1)
    
    return tf.squeeze(encoder_input, axis=0)


def loss_fun(y_ture, y_pred):
    mask = tf.math.logical_not(tf.math.equal(y_ture, 0))  # 为0掩码标1
    loss_ = loss_object(y_ture, y_pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    return tf.reduce_mean(loss_)

def train_step(targets):
    tar_inp = targets[:, :-1]
    tar_real = targets[:, 1:]
    
    # 构造掩码
    combined_mask = create_mask(tar_inp)

    with tf.GradientTape() as tape:
        predictions, predict_fine_tuning = gpt2(tar_inp, training=True, look_ahead_mask=combined_mask)
        loss = loss_fun(tar_real, predictions)
        
    # 求梯度
    gradients = tape.gradient(loss, gpt2.trainable_variables)
    
    # 反向传播
    optimizer.apply_gradients(zip(gradients, gpt2.trainable_variables))

    # 记录loss和准确率
    train_loss(loss)
    train_accuracy(tar_real, predictions)

def translate(sentence, plot=''):
    result = evaluate(sentence)

    predicted_sentence = tokenizer_title.decode([i for i in result if i < tokenizer_title.vocab_size]) 
    predicted_sentence = predicted_sentence.replace(" ", "")
    sentence = sentence.replace(" ", "")

    print('输入: {}'.format(sentence))
    print('预测输出: {}'.format(predicted_sentence))

def encode(lang):
    lang = [tokenizer_title.vocab_size] + tokenizer_title.encode(lang) + [tokenizer_title.vocab_size + 1]
    return lang

def pad_with_zero(lang, max_length=MAX_LENGTH):
    n = MAX_LENGTH - len(lang)
    lang = lang + [0 for k in range(n)]
    return lang

def create_padding_mark(seq):
    # 获取为0的padding项
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

    # 扩充维度以便用于attention矩阵
    return seq[:, np.newaxis, np.newaxis, :] # (batch_size,1,1,seq_len)

def create_look_ahead_mark(size):
    # 1 - 对角线和取下三角的全部对角线（-1->全部）
    # 这样就可以构造出每个时刻未预测token的掩码
    mark = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mark  # (seq_len, seq_len)

# 构建掩码
def create_mask(targets):

    # look_ahead 掩码， 掩掉未预测的词
    look_ahead_mask = create_look_ahead_mark(tf.shape(targets)[1])

    # 解码层第一层得到padding掩码
    decode_targets_padding_mask = create_padding_mark(targets)

    # 合并解码层第一层掩码
    combine_mask = tf.maximum(decode_targets_padding_mask, look_ahead_mask)

    return combine_mask

def main():

    global cat_name_dic, tokenizer_title, optimizer, loss_object, train_loss, train_accuracy, gpt2

    # 加载分类数据
    corpus = []
    cat_name_dic = {
        '100': '民生',
        '101': '文化',
        '102': '娱乐',
        '103': '体育',
        '104': '财经',
        '106': '房产',
        '107': '汽车',
        '108': '教育',
        '109': '科技',
        '110': '军事',
        '112': '旅游',
        '113': '国际',
        '114': '证券',
        '115': '农业',
        '116': '电竞'
    }
    with open('./data/toutiao_cat_data.txt', 'r', encoding='utf-8') as f:
        for k in tqdm(f):
            new_id, cat, cat_n, title, title_kws = k.strip("").split("_!_")
            cat_name = cat_name_dic.get(cat, '')
            if cat_name == "":
                continue
            title = title.replace("|", "")
            title = title + "|" + cat_name
            if len(title) > MAX_LENGTH:
                continue
            title = " ".join(jieba.cut(title, cut_all=False))
            corpus.append(title)

    # 过滤词频特别低的数据
    kws_dic = {}
    for k in tqdm(corpus):
        k = k.split(" ")
        for c in k:
            n = kws_dic.get(c, -1)
            if n == -1:
                n = 1
            else:
                n += 1
            kws_dic[c] = n

    filter_kws = []
    for k in kws_dic.keys():
        filter_kws.append([k, kws_dic[k]])
    filter_kws = pd.DataFrame(filter_kws, columns=['kws', 'num'])
    filter_kws = filter_kws[filter_kws.num >= 1]
    filter_kws = list(filter_kws.kws.values)
    filter_kws_dic = {}
    for k in filter_kws:
        filter_kws_dic[k] = 1

    corpus_format = []
    for k in tqdm(corpus):
        k = k.split(" ")
        k = [c for c in k if filter_kws_dic.get(c, -1) == 1]
        corpus_format.append(" ".join(k))

    random.shuffle(corpus_format)
    train_examples = corpus_format
    tokenizer_title = tfds.features.text.SubwordTextEncoder.build_from_corpus(
        (k for k in train_examples), target_vocab_size=20000)

    train_examples = corpus_format[:300000]
    train_examples = [encode(k) for k in train_examples]
    train_examples = [k for k in train_examples if len(k) <= MAX_LENGTH]
    train_examples = [pad_with_zero(k) for k in train_examples]
    train_dataset = tf.data.Dataset.from_tensor_slices(train_examples)

    # 使用缓存数据加速读入
    train_dataset = train_dataset.cache()

    # 打乱并获取批数据
    train_dataset = train_dataset.batch(BATCH_SIZE)

    # 设置预取数据
    train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)

    # 定义模型参数和优化器
    num_layers = 4
    d_model = 128
    dff = num_layers * d_model
    num_heads = 8
    target_vocab_size = tokenizer_title.vocab_size + 2
    max_seq_len = MAX_LENGTH
    dropout_rate = 0.1
    # 定义优化器
    learing_rate = CustomSchedule(d_model)
    optimizer = tf.keras.optimizers.Adam(learing_rate, beta_1=0.9, 
                                         beta_2=0.98, epsilon=1e-9)

    # 定义目标函数和评估指标
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True,
                                                               reduction='none')

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

    # 定义模型
    gpt2 = GPT2(num_layers, d_model, num_heads, dff,
                target_vocab_size,
                max_seq_len, 
                dropout_rate)

    checkpoint_path = './checkpoint/train_gpt2_exp1'
    ckpt = tf.train.Checkpoint(gpt2=gpt2,
                              optimizer=optimizer)
    # ckpt管理器
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=3)

    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print('last checkpoit restore')

    # 训练模型
    EPOCHS = 20
    step_list = []
    loss_list = []
    step = 0

    for epoch in range(EPOCHS):
        start = time.time()

        # 重置记录项
        train_loss.reset_state()
        train_accuracy.reset_state()

        for batch, all_inputs in tqdm(enumerate(train_dataset)):
            
            # 训练
            train_step(all_inputs)

            if batch % 100 == 0:
                loss = train_loss.result()
                print('epoch {}, batch {}, loss:{:.4f}, acc:{:.4f}'.format(
                    epoch+1, batch, loss, train_accuracy.result()
                ))
                step_list.append(step)
                loss_list.append(loss)
            step += 1

        if (epoch + 1) % 2 == 0:
            ckpt_save_path = ckpt_manager.save()
            print('epoch {}, save model at {}'.format(
            epoch+1, ckpt_save_path
            ))


        print('epoch {}, loss:{:.4f}, acc:{:.4f}'.format(
            epoch+1, train_loss.result(), train_accuracy.result()
        ))

        print('time in 1 epoch:{} secs\n'.format(time.time()-start))
    plt.plot(step_list, loss_list)
    plt.xlabel('train step')
    plt.ylabel('loss')

    # 文本分类效果测试
    test = corpus_format[300000:]
    i = 0
    print("真实数据：", test[i].replace(" ", ""))
    translate(test[i][:-4])
    print("============================")
    i = 1
    print("真实数据：", test[i].replace(" ", ""))
    translate(test[i][:-4])
    print("============================")
    i = 2
    print("真实数据：", test[i].replace(" ", ""))
    translate(test[i][:-4])
    print("============================")
    i = 3
    print("真实数据：", test[i].replace(" ", ""))
    translate(test[i][:-4])

    # 文本生成效果测试
    s = " ".join(list(jieba.cut("杨幂景甜")))
    translate(s)
    print("============================")
    s = " ".join(list(jieba.cut("整容狂人")))
    translate(s)
    print("============================")
    s = " ".join(list(jieba.cut("北大校长口误")))
    translate(s)
    print("============================")

if __name__ == "__main__":
    main()
