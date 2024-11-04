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
from ddpm import *
from unet import *

# Load the dataset
def load_data():
    (x_train, y_train), (_, _) = mnist.load_data()
    x_train = (x_train.astype(np.float32) - 127.5)/127.5
    return (x_train, y_train)

def main():
	timesteps = 500
	X_train, y_train = load_data()
	gaussian_diffusion = GaussianDiffusion(timesteps)

	print("[U-Net] train ddpm")
	nn_model = UNetModel(
	    in_channels=1,
	    model_channels=96,
	    out_channels=1,
	    channel_mult=(1, 2, 2),
	    attention_resolutions=[]
	)
	ddpm = build_DDPM(nn_model)
	gaussian_diffusion = GaussianDiffusion(timesteps=500)
	train_ddpm(ddpm, gaussian_diffusion, epochs=10, batch_size=64, timesteps=500)

	print("[U-Net] generate new images")
	generated_images = gaussian_diffusion.sample(ddpm, 28, batch_size=64, channels=1)
	fig = plt.figure(figsize=(12, 12), constrained_layout=True)
	gs = fig.add_gridspec(8, 8)

	imgs = generated_images[-1].reshape(8, 8, 28, 28)
	for n_row in range(8):
	    for n_col in range(8):
	        f_ax = fig.add_subplot(gs[n_row, n_col])
	        f_ax.imshow((imgs[n_row, n_col]+1.0) * 255 / 2, cmap="gray")
	        f_ax.axis("off")

	print("[U-Net] show the denoise steps")
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
