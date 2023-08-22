import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input,Dense,Conv2D,Flatten,BatchNormalization,UpSampling2D,Conv1D,MaxPooling1D
from tensorflow.keras.layers import PReLU,Add,Concatenate,LeakyReLU,MaxPooling2D,UpSampling2D,Activation,Conv2DTranspose
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam,RMSprop
from tensorflow.keras import optimizers
import tensorflow.keras.backend as K
import tensorflow as tf
import numpy as np
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from tensorflow.python.keras.backend import set_session
from copy import deepcopy
from tensorflow.keras.losses import mean_squared_error as mse
from tensorflow.keras.applications import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input
import os


def convBlock(x,filters1,filters2):
    '''
    Building a Convolutional Block
    :param x:Input layer
    :param filters1: Number of convolutional kernel 1
    :param filters2: Number of convolutional kernel 2
    '''
    conv = Conv2D(filters1,3,padding='same')(x)
    conv = BatchNormalization()(conv)
    conv = LeakyReLU()(conv)
    conv = Conv2D(filters2,3,padding='same')(conv)
    conv = BatchNormalization()(conv)
    conv = LeakyReLU()(conv)
    return conv

def upBlock(upConv,addConv,filters1,filters2):
    '''
    Upsampling block
    :param upConv:
    :param addConv: Upsampling convolutional layer input
    :param filters1: Number of convolutional kernel 1
    :param filters2: Number of convolutional kernel 2
    '''
    #up = UpSampling2D(2)(upConv)
    up = Conv2DTranspose(filters1,3,strides=2,padding='same')(upConv)
    up = Concatenate(axis=-1)([up,addConv])
    up = Conv2D(filters1,3,padding='same')(up)
    up = LeakyReLU()(up)
    up = Conv2D(filters2, 3, padding='same')(up)
    up = LeakyReLU()(up)
    return up


def buildGenerator():
    #input layer
    inputLayer = Input(shape=(48,48,3))

    #down-sampling 1 48x48->24x24
    conv11 = convBlock(inputLayer,filters1=16,filters2=32)
    pool11 = MaxPooling2D(2, strides=2)(conv11)

    #down-sampling 2 24x24->12x12
    conv21 = convBlock(pool11,filters1=32,filters2=64)
    pool21 = MaxPooling2D(2, strides=2)(conv21)

    #down-sampling 3 12x12->6x6
    conv31 = convBlock(pool21,filters1=64,filters2=128,)
    pool31 = MaxPooling2D(2, strides=2)(conv31)

    #down-sampling 4 6x6->3x3
    conv41 = convBlock(pool31,filters1=128,filters2=256)
    pool41 = MaxPooling2D(2, strides=2)(conv41)

    #down-sampling 5 3x3->3x3
    conv51 = convBlock(pool41, filters1=256, filters2=256)
    
    up12 = upBlock(upConv=conv21, addConv=conv11, filters1=32, filters2=32)
    up22 = upBlock(upConv=conv31, addConv=conv21, filters1=64, filters2=64)
    up32 = upBlock(upConv=conv41, addConv=conv31, filters1=128, filters2=128)
    up42 = upBlock(upConv=conv51,addConv=conv41,filters1=128,filters2=128)
    
    up13 = upBlock(upConv=up22, addConv=up12, filters1=32, filters2=32)
    up23 = upBlock(upConv=up32, addConv=up22, filters1=64, filters2=64)
    up33 = upBlock(upConv=up42, addConv=up32, filters1=128, filters2=128)
    
    up14 = upBlock(upConv=up23, addConv=up13, filters1=32, filters2=32)
    up24 = upBlock(upConv=up33, addConv=up23, filters1=64, filters2=64)

    up15 = upBlock(upConv=up24, addConv=up14, filters1=32, filters2=32)

    #Output convolution
    outLayer = convBlock(up15,16,3)
    outLayer = Activation(tf.nn.sigmoid)(outLayer)

    #Modeling
    model = Model(inputs=inputLayer, outputs=outLayer)
    return model


def block(xIn,filterNum):
    '''Convolution+Standardization+Activation Block'''
    x = Conv2D(filterNum,kernel_size=3,strides=3,padding='same')(xIn)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    return x


def buildDiscriminator():
    '''Build discriminator'''
    # Input layer
    inputLayer = Input(shape=(48, 48, 3))

    # Middle layer
    middle = inputLayer
    for num in range(4):
        middle = block(middle, 256)
    middle = Flatten()(middle)
    middle = Dense(1000)(middle)
    middle = LeakyReLU()(middle)

    # Output layer
    outputLayer = Dense(1, activation='sigmoid')(middle)

    # Modeling
    model = Model(inputs=inputLayer, outputs=outputLayer)
    # Optimizer
    optimizer = RMSprop(lr=1e-4)
    model.compile(optimizer=optimizer, loss='binary_crossentropy')
    return model

def normalizeImg(img):
    '''Normalize the image to (0,1), where there can be decimals'''
    img = img/255
    return img

def reverseImg(img):
    '''Restore the image to its original order of magnitude, there cannot be decimals here'''
    img = img*255
    #img = img.astype(np.uint8)
    return img

#feature extractor
vgg19 = VGG19(include_top=False, weights='imagenet')
vgg19 = Model(vgg19.input, vgg19.output)

def contentLoss(y_true, y_pred):
    '''Content loss'''
    y_true = reverseImg(y_true)
    y_pred = reverseImg(y_pred)
    y_true = preprocess_input(y_true)
    y_pred = preprocess_input(y_pred)
    return mse(y_true, y_pred)

def buildGAN(generator,discriminator):
    '''Build GAN'''
    discriminator.trainable = False
    #input of generator
    lowImg = generator.input
    #output of generator
    fakeImg = generator(lowImg)
    #output of discriminator
    judge = discriminator(fakeImg)
    model = Model(inputs=lowImg,outputs=[judge,fakeImg])
    optimizer = RMSprop(lr=1e-4)
    model.compile(optimizer=optimizer, loss=['binary_crossentropy', contentLoss],loss_weights=[1, 1e-1])
    model.summary()
    return model
