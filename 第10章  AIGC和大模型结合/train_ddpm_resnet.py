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
from ddpm import GaussianDiffusion


# ResNet model
class ResNet(keras.layers.Layer):
    
    def __init__(self, in_channels, out_channels, name='ResNet', **kwargs):
        super(ResNet, self).__init__(name=name, **kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
    
    def get_config(self):
        config = super(ResNet, self).get_config()
        config.update({'in_channels': self.in_channels, 'out_channels': self.out_channels})
        return config
    
    @classmethod
    def from_config(cls, config, custom_objects=None):
        return cls(**config)
    
    def build(self, input_shape):
        self.conv1 = Sequential([
            keras.layers.LeakyReLU(),
            keras.layers.Conv2D(filters=self.out_channels, kernel_size=3, padding='same')
        ])
        self.conv2 = Sequential([
            keras.layers.LeakyReLU(),
            keras.layers.Conv2D(filters=self.out_channels, kernel_size=3, padding='same', name='conv2')
        ])

    def call(self, inputs_all, dropout=None, **kwargs):
        """
        `x` has shape `[batch_size, height, width, in_dim]`
        """
        x, t = inputs_all
        h = self.conv1(x)
        h = self.conv2(h)
        h += x
        
        return h

def build_DDPM(nn_model):
    nn_model.trainablea = True
    inputs = Input(shape=(28, 28, 1,))
    timesteps=Input(shape=(1,))
    outputs = nn_model([inputs, timesteps])
    ddpm = Model(inputs=[inputs, timesteps], outputs=outputs)
    ddpm.compile(loss=keras.losses.mse, optimizer=Adam(5e-4))
    return ddpm

# train ddpm
def train_ddpm(ddpm, gaussian_diffusion, epochs=1, batch_size=128, timesteps=500):
    
    #Loading the data
    X_train, y_train = load_data()
    step_cont = len(y_train) // batch_size
    
    step = 1
    for i in range(1, epochs + 1):
        for s in range(step_cont):
            if (s+1)*batch_size > len(y_train):
                break
            images = X_train[s*batch_size:(s+1)*batch_size]
            images = tf.reshape(images, [-1, 28, 28 ,1])
            t = tf.random.uniform(shape=[batch_size], minval=0, maxval=timesteps, dtype=tf.int32)
            loss = gaussian_diffusion.train_losses(ddpm, images, t)
            if step == 1 or step % 100 == 0:
                print("[step=%s]\tloss: %s" %(step, str(tf.reduce_mean(loss).numpy())))
            step += 1

# Load the dataset
def load_data():
    (x_train, y_train), (_, _) = mnist.load_data()
    x_train = (x_train.astype(np.float32) - 127.5)/127.5
    return (x_train, y_train)

def main():
	timesteps = 500
  X_train, y_train = load_data()
  gaussian_diffusion = GaussianDiffusion(timesteps)

  print("[ResNet] train ddpm")
	nn_model = ResNet(in_channels=1, out_channels=1)
	ddpm = build_DDPM(nn_model)
	gaussian_diffusion = GaussianDiffusion(timesteps=500)
	train_ddpm(ddpm, gaussian_diffusion, epochs=10, batch_size=64, timesteps=500)

	print("[ResNet] generate new images")
	generated_images = gaussian_diffusion.sample(ddpm, 28, batch_size=64, channels=1)
	fig = plt.figure(figsize=(12, 12), constrained_layout=True)
	gs = fig.add_gridspec(8, 8)

	imgs = generated_images[-1].reshape(8, 8, 28, 28)
	for n_row in range(8):
	    for n_col in range(8):
	        f_ax = fig.add_subplot(gs[n_row, n_col])
	        f_ax.imshow((imgs[n_row, n_col]+1.0) * 255 / 2, cmap="gray")
	        f_ax.axis("off")

	print("[ResNet] show the denoise steps")
	fig = plt.figure(figsize=(12, 12), constrained_layout=True)
	gs = fig.add_gridspec(16, 16)

	for n_row in range(16):
	    for n_col in range(16):
	        f_ax = fig.add_subplot(gs[n_row, n_col])
	        t_idx = (timesteps // 16) * n_col if n_col < 15 else -1
	        img = generated_images[t_idx][n_row].reshape(28, 28)
	        f_ax.imshow((img+1.0) * 255 / 2, cmap="gray")
	        f_ax.axis("off")


if __name__ == "__main__":
	main()
