import pandas as pd
import numpy as np
import os
import numpy as np
import sys
import pandas as pd
from numpy import arange
import math
import pyecharts
import sys,base64,urllib,re
import multiprocessing
from sklearn.metrics import roc_auc_score
from sklearn.metrics import ndcg_score
import warnings 
from optparse import OptionParser
import logging
import logging.config
import time
import tensorflow as tf
from sklearn.preprocessing import normalize
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import LeakyReLU, Conv2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import datasets
from tensorflow import keras
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

"""
U-Net model
as proposed in https://arxiv.org/pdf/1505.04597v1.pdf
"""

# use sinusoidal position embedding to encode time step (https://arxiv.org/abs/1706.03762)   
def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = tf.exp(
        -math.log(max_period) * tf.experimental.numpy.arange(start=0, stop=half, step=1, dtype=tf.float32) / half
    )
    args = timesteps[:, ] * freqs
    embedding = tf.concat([tf.cos(args), tf.sin(args)], axis=-1)
    if dim % 2:
        embedding = tf.concat([embedding, tf.zeros_like(embedding[:, :1])], axis=-1)
    return embedding

# upsample
class Upsample(keras.layers.Layer):
    def __init__(self, channels, use_conv=False, name='Upsample', **kwargs):
        super(Upsample, self).__init__(name=name, **kwargs)
        self.use_conv = use_conv
        self.channels = channels
    
    def get_config(self):
        config = super(Upsample, self).get_config()
        config.update({'channels': self.channels, 'use_conv': self.use_conv})
        return config
    
    @classmethod
    def from_config(cls, config, custom_objects=None):
        return cls(**config)
    
    def build(self, input_shape):
        if self.use_conv:
            self.conv = keras.layers.Conv2D(filters=self.channels, kernel_size=3, padding='same')

    def call(self, inputs_all, dropout=None, **kwargs):
        x, t = inputs_all
        x = tf.image.resize_with_pad(x, target_height=x.shape[1]*2, target_width=x.shape[2]*2, method='nearest')
#         if self.use_conv:
#             x = self.conv(x)
        return x

# downsample
class Downsample(keras.layers.Layer):
    def __init__(self, channels, use_conv=True, name='Downsample', **kwargs):
        super(Downsample, self).__init__(name=name, **kwargs)
        self.use_conv = use_conv
        self.channels = channels
    
    def get_config(self):
        config = super(Downsample, self).get_config()
        config.update({'channels': self.channels, 'use_conv': self.use_conv})
        return config
    
    @classmethod
    def from_config(cls, config, custom_objects=None):
        return cls(**config)
    
    def build(self, input_shape):
        if self.use_conv:
            self.op = keras.layers.Conv2D(filters=self.channels, kernel_size=3, strides=2, padding='same')
        else:
            self.op = keras.layers.AveragePooling2D(strides=(2, 2))

    def call(self, inputs_all, dropout=None, **kwargs):
        x, t = inputs_all
        return self.op(x)

# Residual block
class ResidualBlock(keras.layers.Layer):
    
    def __init__(
        self, 
        in_channels, 
        out_channels, 
        time_channels, 
        use_time_emb=True,
        name='residul_block', **kwargs
    ):
        super(ResidualBlock, self).__init__(name=name, **kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.time_channels = time_channels
        self.use_time_emb = use_time_emb
    
    def get_config(self):
        config = super(ResidualBlock, self).get_config()
        config.update({
            'time_channels': self.time_channels, 
            'in_channels': self.in_channels, 
            'out_channels': self.out_channels,
            'use_time_emb': self.use_time_emb
        })
        return config
    
    @classmethod
    def from_config(cls, config, custom_objects=None):
        return cls(**config)
    
    def build(self, input_shape):
        self.dense_ = keras.layers.Dense(units=self.out_channels, activation=None)
        self.dense_short = keras.layers.Dense(units=self.out_channels, activation=None)
        
        self.conv1 = [
            keras.layers.LeakyReLU(),
            keras.layers.Conv2D(filters=self.out_channels, kernel_size=3, padding='same')
        ]
        self.conv2 = [
            keras.layers.LeakyReLU(),
            keras.layers.Conv2D(filters=self.out_channels, kernel_size=3, padding='same', name='conv2')
        ]
        self.conv3 = [
            keras.layers.LeakyReLU(),
            keras.layers.Conv2D(filters=self.out_channels, kernel_size=1, name='conv3')
        ]
        
        self.activate = keras.layers.LeakyReLU()

    def call(self, inputs_all, dropout=None, **kwargs):
        """
        `x` has shape `[batch_size, height, width, in_dim]`
        `t` has shape `[batch_size, time_dim]`
        """
        x, t = inputs_all
        h = x
        for module in self.conv1:
            h = module(x)
        
        # Add time step embeddings
        if self.use_time_emb:
            time_emb = self.dense_(self.activate(t))[:, None, None, :]
            h += time_emb
        for module in self.conv2:
            h = module(h)
        
        if self.in_channels != self.out_channels:
            for module in self.conv3:
                x = module(x)
            return h + x
        else:
            return h + x

# Attention block with shortcut
class AttentionBlock(keras.layers.Layer):
    
    def __init__(self, channels, num_heads=1, name='attention_block', **kwargs):
        super(AttentionBlock, self).__init__(name=name, **kwargs)
        self.channels = channels
        self.num_heads = num_heads
        self.dense_layers = []
        
    def get_config(self):
        config = super(AttentionBlock, self).get_config()
        config.update({'channels': self.channels, 'num_heads': self.num_heads})
        return config
    
    @classmethod
    def from_config(cls, config, custom_objects=None):
        return cls(**config)
    
    def build(self, input_shape):
        for i in range(3):
            dense_ = keras.layers.Conv2D(filters=self.channels, kernel_size=1)
            self.dense_layers.append(dense_)
        self.proj = keras.layers.Conv2D(filters=self.channels, kernel_size=1)
    
    def call(self, inputs_all, dropout=None, **kwargs):
        inputs, t = inputs_all
        H = inputs.shape[1]
        W = inputs.shape[2]
        C = inputs.shape[3]
        qkv = inputs
        q = self.dense_layers[0](qkv)
        k = self.dense_layers[1](qkv)
        v = self.dense_layers[2](qkv)
        attn = tf.einsum("bhwc,bHWc->bhwHW", q, k)* (int(C) ** (-0.5))
        attn = tf.reshape(attn, [-1, H, W, H * W])
        attn = tf.nn.softmax(attn, axis=-1)
        attn = tf.reshape(attn, [-1, H, W, H, W])
        
        h = tf.einsum('bhwHW,bHWc->bhwc', attn, v)
        h = self.proj(h)
        
        return h + inputs

# upsample
class UNetModel(keras.layers.Layer):
    def __init__(
        self,
        in_channels=3,
        model_channels=128,
        out_channels=3,
        num_res_blocks=2,
        attention_resolutions=(8, 16),
        dropout=0,
        channel_mult=(1, 2, 2, 2),
        conv_resample=True,
        num_heads=4,
        name='UNetModel',
        **kwargs
    ):
        super(UNetModel, self).__init__(name=name, **kwargs)
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_heads = num_heads
        self.time_embed_dim = self.model_channels * 4
    
    def build(self, input_shape):
        
        # time embedding
        self.time_embed = [
            keras.layers.Dense(self.time_embed_dim, activation=None),
            keras.layers.LeakyReLU(),
            keras.layers.Dense(self.time_embed_dim, activation=None)
        ]
        
        # down blocks
        self.conv = keras.layers.Conv2D(filters=self.model_channels, kernel_size=3, padding='same')
        self.down_blocks = []
        down_block_chans = [self.model_channels]
        ch = self.model_channels
        ds = 1
        index = 0
        for level, mult in enumerate(self.channel_mult):
            for _ in range(self.num_res_blocks):
                
                layers = [
                    ResidualBlock(
                        in_channels=ch, 
                        out_channels=mult * self.model_channels, 
                        time_channels=self.time_embed_dim,
                        name='resnet_'+str(index)
                    )
                ]
                index += 1
                ch = mult * self.model_channels
                if ds in self.attention_resolutions:
                    layers.append(AttentionBlock(ch, num_heads=self.num_heads))
                self.down_blocks.append(layers)
                down_block_chans.append(ch)
        
            if level != len(self.channel_mult) - 1: # don't use downsample for the last stage
                self.down_blocks.append(Downsample(ch, self.conv_resample))
                down_block_chans.append(ch)
                ds *= 2
                
        # middle block
        self.middle_block = [
            ResidualBlock(ch, ch, self.time_embed_dim, name='res1'),
            AttentionBlock(ch, num_heads=self.num_heads),
            ResidualBlock(ch, ch, self.time_embed_dim, name='res2')
        ]
        
        # up blocks
        self.up_blocks = []
        index = 0
        for level, mult in list(enumerate(self.channel_mult))[::-1]:
            for i in range(self.num_res_blocks + 1):
                layers = []
                layers.append(
                    ResidualBlock(
                        in_channels=ch + down_block_chans.pop(), 
                        out_channels=self.model_channels * mult, 
                        time_channels=self.time_embed_dim,
                        name='up_resnet_'+str(index)
                    )
                )
                
                layer_num = 1
                ch = self.model_channels * mult
                if ds in self.attention_resolutions:
                    layers.append(AttentionBlock(ch, num_heads=self.num_heads))
                if level and i == self.num_res_blocks:
                    layers.append(Upsample(ch, self.conv_resample))
                    ds //= 2
                self.up_blocks.append(layers)
                
                index += 1
            
        
        self.out = Sequential([
            keras.layers.LeakyReLU(),
            keras.layers.Conv2D(filters=self.out_channels, kernel_size=3, padding='same')
        ])

    def call(self, inputs, dropout=None, **kwargs):
        """
        Apply the model to an input batch.
        :param x: an [N x H x W x C] Tensor of inputs. N, H, W, C
        :param timesteps: a 1-D batch of timesteps.
        :return: an [N x C x ...] Tensor of outputs.
        """
        x, timesteps = inputs
        hs = []
        
        # time step embedding
        emb = timestep_embedding(timesteps, self.model_channels)
        for module in self.time_embed:
            emb = module(emb)
        
        # down stage
        h = x
        h = self.conv(h)
        hs = [h]
        for module_list in self.down_blocks:
            if isinstance(module_list, list):
                for module in module_list:
                    h = module([h, emb])
            else:
                h = module_list([h, emb])
            hs.append(h)
            
        # middle stage
        for module in self.middle_block:
            h = module([h, emb])
        
        # up stage
        for module_list in self.up_blocks:
            cat_in = tf.concat([h, hs.pop()], axis=-1)
            h = cat_in
            for module in module_list:
                h = module([h, emb])
        
        return self.out(h)
