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
import re
from tqdm import tqdm
from gpt1 import GPT1

MAX_LENGTH = 40
BUFFER_SIZE = 20000
BATCH_SIZE = 64

def load_data(file_path):
    corpus = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for k in f:
            new_id, cat, cat_n, title, title_kws = k.strip("").split("_!_")
            cat_name = cat_name_dic.get(cat, '')
            if cat_name == '':
                continue
            if len(title) > MAX_LENGTH:
                continue
                
            label = [0 for i in range(len(cat_name_all))]
            index = cat_name_label[cat_name]
            label[index] = 1
            corpus.append([title, label])
    return corpus

def encode(lang):
    lang1, lang2 = lang
    lang1 = [tokenizer_title.vocab_size] + tokenizer_title.encode(lang1) + [tokenizer_title.vocab_size + 1]
    return [lang1, lang2]

def filter_long_sent(x, y, max_length=MAX_LENGTH):
    return tf.logical_and(tf.size(x) <= max_length, tf.size(y) <= max_length)

def pad_with_zero(lang, max_length=MAX_LENGTH):
    lang1, lang2 = lang
    n1 = MAX_LENGTH - len(lang1)
    lang1 = lang1 + [0 for k in range(n1)]
    return [lang1, lang2]

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


def create_look_ahead_mark(size):
    # 1 - 对角线和取下三角的全部对角线（-1->全部）
    # 这样就可以构造出每个时刻未预测token的掩码
    mark = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mark  # (seq_len, seq_len)

def create_padding_mark(seq):
    # 获取为0的padding项
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

    # 扩充维度以便用于attention矩阵
    return seq[:, np.newaxis, np.newaxis, :] # (batch_size,1,1,seq_len)

# 构建掩码
def create_mask(targets):

    # look_ahead 掩码， 掩掉未预测的词
    look_ahead_mask = create_look_ahead_mark(tf.shape(targets)[1])
    
    # 解码层第一层得到padding掩码
    decode_targets_padding_mask = create_padding_mark(targets)

    # 合并解码层第一层掩码
    combine_mask = tf.maximum(decode_targets_padding_mask, look_ahead_mask)

    return combine_mask

def loss_fun(y_ture, y_pred):
    mask = tf.math.logical_not(tf.math.equal(y_ture, 0))  # 为0掩码标1
    loss_ = loss_object(y_ture, y_pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    return tf.reduce_mean(loss_)

def loss_fun_fine_tuning(y_ture, y_pred):
    loss_ = loss_object_fine_tuning(y_ture, y_pred)
    return tf.reduce_mean(loss_)

def train_step(targets):
    tar_inp = targets['title'][:, :-1]
    tar_real = targets['title'][:, 1:]
    cat_name = targets['cat']
    
    # 构造掩码
    combined_mask = create_mask(tar_inp)

    with tf.GradientTape() as tape:
        predictions, predict_fine_tuning, _ = gpt1(tar_inp, training=True, look_ahead_mask=combined_mask)
        loss = loss_fun(tar_real, predictions)
        loss_fine_tuning = loss_fun_fine_tuning(cat_name, predict_fine_tuning)
        loss_combine = loss + loss_fine_tuning
        
    # 求梯度
    gradients = tape.gradient(loss_combine, gpt1.trainable_variables)
    
    # 反向传播
    optimizer.apply_gradients(zip(gradients, gpt1.trainable_variables))

    # 记录loss和准确率
    train_loss(loss)
    train_accuracy(tar_real, predictions)
    train_loss_fine_tuning(loss_fine_tuning)
    train_accuracy_fine_tuning(cat_name, predict_fine_tuning)

def evaluate(inp_sentence):
    
    start_token = [tokenizer_title.vocab_size]
    end_token = [tokenizer_title.vocab_size + 1]
    inp_sentence = start_token + tokenizer_title.encode(inp_sentence) + end_token
    n = MAX_LENGTH - len(inp_sentence)
    inp_sentence = inp_sentence + [0 for k in range(n)]
    inp_sentence = inp_sentence[:-1]
    inp_sentence = tf.expand_dims(inp_sentence, 0)
    
    combined_mask = create_mask(inp_sentence)
    predictions, predict_fine_tuning, _ = gpt1(inp_sentence, training=False, look_ahead_mask=combined_mask)
    predicted_id = tf.cast(tf.argmax(predict_fine_tuning, axis=-1), tf.int32)
    return predicted_id

def evaluate_func(val_dataset):
    predict = []
    real = []
    for k in tqdm(val_dataset):
        inp = tf.expand_dims(k['title'][:-1], 0)
        combined_mask = create_mask(inp)
        predictions, predict_fine_tuning, _ = gpt1(inp, training=False, look_ahead_mask=combined_mask)
        predicted_id = tf.cast(tf.argmax(predict_fine_tuning, axis=-1), tf.int32)
        
        real_ = k['cat']
        s = list(real_.numpy()).index(1)
        real.append(s)
        predict += list(predicted_id.numpy())
    
    return predict, real
    
def get_cat_name(sentence, plot=''):
    result = evaluate(sentence)[0]
    result = cat_name_all[result]

    print('输入: {}'.format(sentence).replace(" ", ""))
    print('预测输出: {}'.format(result))

def get_real_cat(label):
    index = label.index(1)
    return cat_name_all[index]


def main():

    global cat_name_dic, cat_name_all, cat_name_label, tokenizer_title, optimizer, loss_object, train_loss, train_accuracy
    global gpt1, loss_object_fine_tuning, train_loss_fine_tuning, train_accuracy_fine_tuning

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
    cat_name_all = list(cat_name_dic.values())
    cat_name_label = dict([(cat_name_all[k], k) for k in range(len(cat_name_all))])

    corpus = load_data('./data/toutiao_cat_data.txt') # 数据集https://link.zhihu.com/?target=https%3A//github.com/BenDerPan/toutiao-text-classfication-dataset
    random.shuffle(corpus)
    print(corpus[0])

    ### 分词
    corpus_format = []
    for k in corpus:
        title = k[0]
        cat = k[1]
        title = " ".join(jieba.cut(title, cut_all=False))
        corpus_format.append([title, cat])
    print(corpus_format[5])

    random.shuffle(corpus_format)
    train_examples, val_examples = corpus_format[:300000], corpus_format[300000:]
    tokenizer_title = tfds.features.text.SubwordTextEncoder.build_from_corpus(
        (k[0] for k in train_examples), target_vocab_size=2**13)

    sample_str = '为什么 商用 客机 一般 先要 卖 给 银行'
    tokenized_str = tokenizer_title.encode(sample_str)
    print(tokenized_str)
    original_str = tokenizer_title.decode(tokenized_str)
    print(original_str)

    train_examples, val_examples = corpus_format[:300000], corpus_format[300000:]
    train_examples = [encode(k) for k in train_examples]
    train_examples = [k for k in train_examples if len(k[0]) <= MAX_LENGTH]
    train_examples = [pad_with_zero(k) for k in train_examples]
    dic = {}
    dic['title'] = [k[0] for k in train_examples]
    dic['cat'] = [k[1] for k in train_examples]
    train_examples = dic
    train_dataset = tf.data.Dataset.from_tensor_slices(train_examples)

    # 使用缓存数据加速读入
    train_dataset = train_dataset.cache()

    # 打乱并获取批数据
    train_dataset = train_dataset.batch(BATCH_SIZE)

    # 设置预取数据
    train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)

    # # 验证集数据
    val_examples = [encode(k) for k in val_examples]
    val_examples = [k for k in val_examples if len(k[0]) <= MAX_LENGTH]
    val_examples = [pad_with_zero(k) for k in val_examples]
    dic['title'] = [k[0] for k in val_examples]
    dic['cat'] = [k[1] for k in val_examples]
    val_examples = dic
    val_dataset = tf.data.Dataset.from_tensor_slices(val_examples)

    num_layers = 4
    d_model = 128
    dff = 512
    num_heads = 8
    target_vocab_size = tokenizer_title.vocab_size + 2
    max_seq_len = MAX_LENGTH
    dropout_rate = 0.1
    n_class = len(cat_name_dic)

    # 定义优化器
    learing_rate = CustomSchedule(d_model)
    optimizer = tf.keras.optimizers.Adam(learing_rate, beta_1=0.9, 
                                         beta_2=0.98, epsilon=1e-9)

    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True,
                                                               reduction='none')
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

    loss_object_fine_tuning = tf.keras.losses.CategoricalCrossentropy(from_logits=True, reduction='none')
    train_loss_fine_tuning = tf.keras.metrics.Mean(name='train_loss_fine_tuning')
    train_accuracy_fine_tuning = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy_fine_tuning')

    # 定义模型
    gpt1 = GPT1(num_layers, d_model, num_heads, dff,
                target_vocab_size,
                max_seq_len, 
                n_class,
                dropout_rate)

    checkpoint_path = './checkpoint/train_cat'
    ckpt = tf.train.Checkpoint(gpt1=gpt1,
                              optimizer=optimizer)
    # ckpt管理器
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=3)

    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print('last checkpoit restore')


    # 开始训练模型
    EPOCHS = 20
    step_list = []
    loss_list = []
    loss_list_fine_tuning = []
    step = 0

    for epoch in list(range(EPOCHS)):
        start = time.time()

        # 重置记录项
        train_loss.reset_state()
        train_accuracy.reset_state()
        train_loss_fine_tuning.reset_state()
        train_accuracy_fine_tuning.reset_state()

        for batch, all_inputs in tqdm(enumerate(train_dataset)):
            
            # 训练
            train_step(all_inputs)

            if batch % 1000 == 0:
                loss = train_loss.result()
                loss_fine_tuning = train_loss_fine_tuning.result()
                print('epoch {}, batch {}, loss:{:.4f}, loss_fine:{:.4f}, acc:{:.4f}'.format(
                    epoch+1, batch, loss, loss_fine_tuning, train_accuracy_fine_tuning.result()
                ))
                step_list.append(step)
                loss_list.append(loss)
                loss_list_fine_tuning.append(loss_fine_tuning)
            step += 1

        if (epoch + 1) % 2 == 0:
            ckpt_save_path = ckpt_manager.save()
            print('epoch {}, save model at {}'.format(epoch+1, ckpt_save_path))

        print('epoch {}, loss:{:.4f}, acc:{:.4f}'.format(epoch+1, train_loss.result(), train_accuracy.result()))
        print('time in 1 epoch:{} secs\n'.format(time.time()-start))
        
    # 测试准确率
    predict, real = evaluate_func(val_dataset)
    acc = np.sum(np.array(predict) == np.array(real)) / len(real)
    print("验证集上的准确率：", acc) # 0.8550686378035903

    # 用于文本分类
    s = "《狂飙》结局后，张译终于发声了，剧中演员回应一辈子不想见张译"
    s = " ".join(jieba.cut(s))
    get_cat_name(s)
    print("==============================================================")
    s = "教育部下发新通知，将调整今年的高考方向，家长看完心态“崩”了"
    s = " ".join(jieba.cut(s))
    get_cat_name(s)
    print("==============================================================")
    s = "俄罗斯学会了，发射大批气球飞向乌克兰，乌军导弹快不够用了"
    s = " ".join(jieba.cut(s))
    get_cat_name(s)


if __name__ == "__main__":
    main()
