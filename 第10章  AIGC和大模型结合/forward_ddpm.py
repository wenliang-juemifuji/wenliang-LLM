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
from ddpm import GaussianDiffusion
%matplotlib inline

# Load the dataset
def load_data():
    (x_train, y_train), (_, _) = mnist.load_data()
    x_train = (x_train.astype(np.float32) - 127.5)/127.5
    return (x_train, y_train)

def main():
    print("forward diffusion: q(x_t | x_0)")
    timesteps = 500
    X_train, y_train = load_data()
    gaussian_diffusion = GaussianDiffusion(timesteps)
    plt.figure(figsize=(16, 8))
    x_start = X_train[7:8]
    for idx, t in enumerate([0, 50, 100, 200, 499]):
        x_noisy = gaussian_diffusion.q_sample(x_start, t=tf.convert_to_tensor([t]))
        x_noisy = x_noisy.numpy()
        x_noisy = x_noisy.reshape(28, 28)
        plt.subplot(1, 5, 1 + idx)
        plt.imshow(x_noisy, cmap="gray")
        plt.axis("off")
        plt.title(f"t={t}")


if __name__ == "__main__":
	main()

