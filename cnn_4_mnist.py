# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 11:48:06 2019

@author: sxd170023
"""
import numpy as np
import pickle
from keras import applications
from keras.models import Sequential 
from keras.layers import Dense, Activation, Dropout, BatchNormalization, Conv2D, MaxPool2D, Flatten
from sklearn.model_selection import train_test_split
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.optimizers import Adam
import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
import keras
from keras.layers import Dense, Dropout, Input, Flatten, Reshape, Conv2D, MaxPooling2D
from keras.models import Model,Sequential
from keras.datasets import mnist
from tqdm import tqdm
from keras.layers.advanced_activations import LeakyReLU, ReLU
from keras.optimizers import Adam
import pickle
import tensorflow as tf

epochs = 10
batch_size = 32
img_shape = (28,28,1)
d_out = []
num_category = 10
def load_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = (x_train.astype(np.float32) - 127.5)/127.5
   
    # convert shape of x_train from (60000, 28, 28) to (60000, 784) 
    # 784 columns per row
    #x_train = x_train.reshape(60000, 784)
    return (x_train, y_train, x_test, y_test)

def create_discriminator():
    
         ##model building
        model = Sequential()
        #convolutional layer with rectified linear unit activation
        model.add(Conv2D(32, kernel_size=(3, 3),
                         activation='relu',
                         input_shape = img_shape))
        #32 convolution filters used each of size 3x3
        #again
        model.add(Conv2D(64, (3, 3), activation='relu'))
        #64 convolution filters used each of size 3x3
        #choose the best features via pooling
        model.add(MaxPooling2D(pool_size=(2, 2)))
        #randomly turn neurons on and off to improve convergence
        model.add(Dropout(0.25))
        #flatten since too many dimensions, we only want a classification output
        model.add(Flatten())
        #fully connected to get all relevant data
        model.add(Dense(128, activation='relu'))
        #one more dropout for convergence' sake :) 
        model.add(Dropout(0.5))
        #output a softmax to squash the matrix into output probabilities
        model.add(Dense(num_category, activation='softmax'))
        return model

def adam_optimizer():
    return Adam(lr=0.0002, beta_1=0.5)

def training(epochs,batch_size):    
    (X_train, y_train,X_test, y_test)=load_data()
    X_train = np.expand_dims(X_train, axis=3)
    print(X_train.shape)
    #X_test = X_test.reshape(10000,784)
    X_test = np.expand_dims(X_test, axis=3)
    y_train = keras.utils.to_categorical(y_train, num_category)
    y_test = keras.utils.to_categorical(y_test, num_category)
    print(X_test.shape,y_test.shape)
    model_aug= create_discriminator()
    model_aug.compile(loss='binary_crossentropy', optimizer=adam_optimizer(),metrics=['accuracy'])
    model_aug.summary()
    
  #d_loss = discriminator.train_on_batch(X, y_dis)
    d_loss = model_aug.fit(X_train, y_train,batch_size=batch_size,epochs=epochs)
    d_out.append(np.array(d_loss))

                #print("%d %f %f %f " % (epochs, d_loss[0], d_loss[1],g_loss))
               # print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epochs, d_loss[0], 100*d_loss[1], g_loss))
                
    #discriminator.model.save("d_images_model.h5")
    scores = model_aug.evaluate(X_test,y_test,batch_size=32)
    print(model_aug.evaluate(X_test,y_test,batch_size=32))
    
    print("%s: %.2f%%" % (model_aug.metrics_names[1], scores[1]))
   # print("%.2f%% (+/- %.2f%%)" % (np.mean(scores), np.std(scores)))           
    print("%s: %.2f%%" % (model_aug.metrics_names[0], scores[0]))          
  
   
training(epochs,batch_size)
