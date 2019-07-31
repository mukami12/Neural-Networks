# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 11:54:02 2019

@author: sxd170023
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
import keras
from keras.layers import Dense, Dropout, Input, Flatten, Reshape
from keras.models import Model,Sequential
from keras.datasets import mnist
from tqdm import tqdm
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam
import pickle
import tensorflow as tf


epochs = 10000
batch_size = 32
img_shape = (28,28,1)
d_out = []
g_out = []

def load_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = (x_train.astype(np.float32) - 127.5)/127.5
    
    # convert shape of x_train from (60000, 28, 28) to (60000, 784) 
    # 784 columns per row
    x_train = x_train.reshape(60000, 784)
    return (x_train, y_train, x_test, y_test)

def adam_optimizer():
    return Adam(lr=0.0002, beta_1=0.5)

def create_generator():
    gmodel = Sequential()
    gmodel.add(Dense(256, input_dim=100))
    gmodel.add(LeakyReLU(alpha=0.2))
    gmodel.add(BatchNormalization(momentum=0.8))
    gmodel.add(Dense(512))
    gmodel.add(LeakyReLU(alpha=0.2))
    gmodel.add(BatchNormalization(momentum=0.8))
    gmodel.add(Dense(1024))
    gmodel.add(LeakyReLU(alpha=0.2))
    gmodel.add(BatchNormalization(momentum=0.8))
    gmodel.add(Dense(units=784, activation='tanh'))
    return gmodel

def create_discriminator():
    dmodel = Sequential()
    dmodel.add(Dense(512,input_dim=784))
    dmodel.add(LeakyReLU(alpha=0.2))
    dmodel.add(Dense(256))
    dmodel.add(LeakyReLU(alpha=0.2))
    dmodel.add(Dense(1, activation='sigmoid'))
    return dmodel 


def create_gan(discriminator, generator):
    discriminator.trainable=False
    gan_input = Input(shape=(100,))
    x = generator(gan_input)
    gan_output = discriminator(x)
    gan= Model(inputs=gan_input, outputs=gan_output)
    return gan


def plot_generated_images(epoch, generator, examples=100, dim=(10,10), figsize=(10,10)):
    noise= np.random.normal(loc=0, scale=1, size=[examples, 100])
    generated_images = generator.predict(noise)
    generated_images = generated_images.reshape(100,28,28)
    plt.figure(figsize=figsize)
    for i in range(generated_images.shape[0]):
        plt.subplot(dim[0], dim[1], i+1)
        plt.imshow(generated_images[i], interpolation='nearest')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig("images/%d.png" % epoch)
  
def training(epochs,batch_size):    
    (X_train, y_train,X_test, y_test)=load_data()
    print(X_train.shape)
    X_test = X_test.reshape(10000,784)
    print(X_test.shape,y_test.shape)
    generator= create_generator()
    discriminator= create_discriminator()
    generator.compile(loss='binary_crossentropy', optimizer=adam_optimizer(),metrics=['accuracy'])   
    discriminator.compile(loss='binary_crossentropy', optimizer=adam_optimizer(),metrics=['accuracy'])   
 
    gan = create_gan(discriminator, generator)
    gan.compile(loss='binary_crossentropy', optimizer='adam')
    discriminator.summary()
    generator.summary()
    gan.summary()
    
    for e in range(1,epochs+1 ):
            print("Epoch %d" %e)
            for _ in tqdm(range(batch_size)):
            #generate  random noise as an input  to  initialize the  generator
                noise= np.random.normal(0,1, [batch_size, 100])
                
                # Generate fake MNIST images from noised input
                generated_images = generator.predict(noise)
                
                # Get a random set of  real images
                image_batch = X_train[np.random.randint(low=0,high=X_train.shape[0],size=batch_size)]
                #Construct different batches of  real and fake data 
                X= np.concatenate([image_batch, generated_images])
                
                # Labels for generated and real data
                y_dis=np.zeros(2*batch_size)
                y_dis[:batch_size]=0.9
                
                #Pre train discriminator on  fake and real data  before starting the gan. 
                discriminator.trainable=True
               
                #d_loss = discriminator.train_on_batch(X, y_dis)
                d_loss = discriminator.fit(X, y_dis)
                print("Discriminator LOSS",d_loss)
                d_out.append(np.array(d_loss))
                
                # Fixing Discriminator weights
                discriminator.trainable=False
                  #Tricking the noised input of the Generator as real data
                noise= np.random.normal(0,1, [batch_size, 100])
                y_gen = np.ones(batch_size)
           
                #Train the GAN     
                g_loss= gan.fit(noise,y_gen)
                
            if e == 1 or e % 20 == 0:
                plot_generated_images(e,generator)
               
    generator.model.save("g_images_model.h5")
    discriminator.model.save("d_images_model.h5")
    discriminator.trainable = False
    
    #Evaluate the Generator
    scores = discriminator.evaluate(X_test,y_test,batch_size=32)
    
    print("%s: %.2f%%" % (discriminator.metrics_names[1], scores[1]*100))
    print("%.2f%% (+/- %.2f%%)" % (np.mean(scores), np.std(scores)))           
    print("%s: %.2f%%" % (discriminator.metrics_names[0], scores[0]*100))          
  
    
training(epochs,batch_size)
