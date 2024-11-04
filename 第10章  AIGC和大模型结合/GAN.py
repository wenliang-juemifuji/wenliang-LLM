import pandas as pd
import numpy as np
import os
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.optimizers import Adam
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

# 加载数据集
def load_data():
    (x_train, y_train), (_, _) = mnist.load_data()
    x_train = (x_train.astype(np.float32) - 127.5)/127.5
    # Convert shape from (60000, 28, 28) to (60000, 784)
    x_train = x_train.reshape(60000, 784)
    return (x_train, y_train)

def draw_images(generator, epoch, examples=5, dim=(1, 5), figsize=(5, 5)):
    noise= np.random.normal(loc=0, scale=1, size=[examples, 100])
    generated_images = generator.predict(noise)
    generated_images = generated_images.reshape(5, 28, 28)
    plt.figure(figsize=figsize)
    for i in range(generated_images.shape[0]):
        plt.subplot(dim[0], dim[1], i+1)
        plt.imshow(generated_images[i], interpolation='nearest', cmap='Greys')
        plt.axis('off')
    plt.tight_layout()

###构建生成器网络
def build_generator():
    model = Sequential()
    
    model.add(Dense(units=256, input_dim=100))
    model.add(LeakyReLU(alpha=0.2))
    
    model.add(Dense(units=512))
    model.add(LeakyReLU(alpha=0.2))
    
    model.add(Dense(units=1024))
    model.add(LeakyReLU(alpha=0.2))
    
    model.add(Dense(units=784, activation='tanh'))
    
    model.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5))
    return model

###构建判别器网络
def build_discriminator():
    model = Sequential()
    
    model.add(Dense(units=1024 ,input_dim=784))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.3))
       
    model.add(Dense(units=512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.3))
       
    model.add(Dense(units=256))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.3))
      
    model.add(Dense(units=1, activation='sigmoid'))
    
    model.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5))
    return model

###构建整个GAN网络
def build_GAN(discriminator, generator):
    discriminator.trainable=False
    GAN_input = Input(shape=(100,))
    x = generator(GAN_input)
    GAN_output= discriminator(x)
    GAN = Model(inputs=GAN_input, outputs=GAN_output)
    GAN.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5))
    return GAN

###训练GAN
def train_GAN(epochs=1, batch_size=128):
    
    #加载真实的手写体图片
    X_train, y_train = load_data()

    # 创建GAN网络
    generator= build_generator()
    discriminator= build_discriminator()
    GAN = build_GAN(discriminator, generator)
    
    for i in range(1, epochs+1):
        
        for _ in range(batch_size):
            # 输出生成器图片
            noise= np.random.normal(0,1, (batch_size, 100))
            fake_images = generator.predict(noise)

            # 从真实数据集中选择图片
            real_images = X_train[np.random.randint(0, X_train.shape[0], batch_size)]

            # 构建判别器的正负label           
            label_fake = np.zeros(batch_size)
            label_real = np.ones(batch_size) 

            # 创建训练集
            X = np.concatenate([fake_images, real_images])
            y = np.concatenate([label_fake, label_real])

            # 使用正负样本训练判别器
            discriminator.trainable=True
            discriminator.train_on_batch(X, y)

            # 使用生成器输出的图片作为正样本，训练生成器网络
            discriminator.trainable=False
            GAN.train_on_batch(noise, label_real)

        # Draw generated images every 15 epoches     
        if i == 1 or i % 100 == 0:
            draw_images(generator, i)
        
        if i > 1 and i % 100 == 0:
            print("Epoch %d" %i)

def main():
	train_GAN(epochs=400, batch_size=128)


if __name__ == "__main__":
	main()
