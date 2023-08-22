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
    搭建一个卷积块
    :param x:输入层
    :param filters1: 卷积核数量1
    :param filters2: 卷积核数量2
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
    上采样块
    :param upConv:上采样卷积层输入
    :param addConv: 相加卷积层输入
    :param filters1: 卷积核数1
    :param filters2: 卷积核数2
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
    '''搭建Unet'''
    #输入层
    inputLayer = Input(shape=(48,48,3))

    #下采样1 48x48->24x24
    conv11 = convBlock(inputLayer,filters1=16,filters2=32)
    pool11 = MaxPooling2D(2, strides=2)(conv11)

    #下采样2 24x24->12x12
    conv21 = convBlock(pool11,filters1=32,filters2=64)
    pool21 = MaxPooling2D(2, strides=2)(conv21)

    #下采样3 12x12->6x6
    conv31 = convBlock(pool21,filters1=64,filters2=128,)
    pool31 = MaxPooling2D(2, strides=2)(conv31)

    #下采样4 6x6->3x3
    conv41 = convBlock(pool31,filters1=128,filters2=256)
    pool41 = MaxPooling2D(2, strides=2)(conv41)

    #下采样 3x3->3x3
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

    #输出卷积
    outLayer = convBlock(up15,16,3)
    outLayer = Activation(tf.nn.sigmoid)(outLayer)

    #建模
    model = Model(inputs=inputLayer, outputs=outLayer)
    return model


def block(xIn,filterNum):
    '''卷积+标准化+激活块'''
    x = Conv2D(filterNum,kernel_size=3,strides=3,padding='same')(xIn)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    return x


def buildDiscriminator():
    '''创建判别器'''
    # 输入层
    inputLayer = Input(shape=(48, 48, 3))

    # 中间层
    middle = inputLayer
    for num in range(4):
        middle = block(middle, 256)
    middle = Flatten()(middle)
    middle = Dense(1000)(middle)
    middle = LeakyReLU()(middle)

    # 输出层
    outputLayer = Dense(1, activation='sigmoid')(middle)

    # 建模
    model = Model(inputs=inputLayer, outputs=outputLayer)
    #优化器
    optimizer = RMSprop(lr=1e-4)
    model.compile(optimizer=optimizer, loss='binary_crossentropy')
    return model

def normalizeImg(img):
    '''将图片归一化到0,1，这里可以有小数'''
    img = img/255
    return img

def reverseImg(img):
    '''将图片还原到原数量级，这里不能有小数'''
    img = img*255
    #img = img.astype(np.uint8)
    return img

#特征提取器
vgg19 = VGG19(include_top=False, weights='imagenet')
vgg19 = Model(vgg19.input, vgg19.output)

def contentLoss(y_true, y_pred):
    '''内容损失'''
    y_true = reverseImg(y_true)
    y_pred = reverseImg(y_pred)
    y_true = preprocess_input(y_true)
    y_pred = preprocess_input(y_pred)
    return mse(y_true, y_pred)

def buildGAN(generator,discriminator):
    '''构建对抗网'''
    discriminator.trainable = False
    #生成器输入
    lowImg = generator.input
    #生成器输出
    fakeImg = generator(lowImg)
    #生成器判断
    judge = discriminator(fakeImg)
    model = Model(inputs=lowImg,outputs=[judge,fakeImg])
    optimizer = RMSprop(lr=1e-4)
    model.compile(optimizer=optimizer, loss=['binary_crossentropy', contentLoss],loss_weights=[1, 1e-1])
    model.summary()
    return model
