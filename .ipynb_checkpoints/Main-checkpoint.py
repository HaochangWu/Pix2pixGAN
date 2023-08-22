from Module.BuildModel import *
from Module.Common import *
from glob import glob
import numpy as np
import pandas as pd
import cv2
import os
#整理输入输出路径
inputDir = staticDir+"/png/raw(png)/raw/"
outputDir = staticDir+"/png/real(png)jet/real/"
names = os.listdir(inputDir)
batchSize = 100#批处理量
#整理测试训练集
data = pd.DataFrame({"name":names,"input":[inputDir+name for name in names],"output":[outputDir+name for name in names]})
data.sample(frac=1.0,random_state=0)#打乱数据
trainData = data.iloc[:int(data.shape[0]*0.8)]
valiData = data.iloc[int(data.shape[0]*0.8):int(data.shape[0]*0.9)]
testData = data.iloc[int(data.shape[0]*0.9):]
del data

def collect(data,indexs,augment=0):
    '''
    根据索引从数据集抽取数据，形成输入输出
    :param data: 数据集
    :param indexs:索引
    :param augment:数据增强
    '''
    subData = data.iloc[indexs]
    xAll = []
    yAll = []
    for i in range(subData.shape[0]):
        row = subData.iloc[i]
        x = cv2.imdecode(np.fromfile(row["input"], dtype=np.uint8),-1)
        x = normalizeImg(x)
        y = cv2.imdecode(np.fromfile(row["output"], dtype=np.uint8),-1)
        y = normalizeImg(y)
        #翻转
        if augment:
            k = np.random.randint(4)
            x = np.rot90(x,k=k)
            y = np.rot90(y,k=k)
        xAll.append(x)
        yAll.append(y)
    xAll = np.stack(xAll)
    yAll = np.stack(yAll)
    return xAll,yAll

def calAcc(yTrain,yPred):
    '''计算像素准确率'''
    yTrain = reverseImg(yTrain)
    yPred = reverseImg(yPred)
    acc = 1-np.sum(np.abs(yTrain - yPred))/(48*48*3*255*batchSize)
    return acc

def calDataAcc(data,batchSize):
    '''计算一个数据集的像素准确率'''
    acc = []
    for j in range(0, data.shape[0], batchSize):
        x, y = collect(data,range(j,min(j+batchSize,data.shape[0])),augment=0)
        yPred = generator.predict(x)
        acc.append(calAcc(y,yPred))
    acc = np.mean(acc)
    return acc

#实例化网络
generator = buildGenerator()
discriminator = buildDiscriminator()
gan = buildGAN(generator,discriminator)
gan.summary()

#开始训练
steps = 0#迭代次数

history = {"discriminator_loss":[],"GAN_loss":[],"content_loss":[]}
testHistory = {"step":[],"train-acc":[],"validate-acc":[],"test-acc":[]}
trainAcc = []
bestValAcc = 0
while True:

    #随机抽取训练集
    indexs = np.random.choice(trainData.shape[0],batchSize,replace=False)
    xTrain,yTrain = collect(trainData,indexs,0)
    yPred = generator.predict(xTrain)
    #真伪标签
    realBool = np.random.uniform(0.7,1,size=(batchSize,))
    fakeBool = np.random.uniform(0,0.3,size=(batchSize,))
    #鉴别器训练
    discriminator.trainable = True
    dRealLoss = discriminator.train_on_batch(x=yTrain, y=realBool)
    dFakeLoss = discriminator.train_on_batch(x=yPred, y=fakeBool)
    #判别损失
    history['discriminator_loss'].append(0.5 * (dRealLoss + dFakeLoss))

    #生成器训练
    discriminator.trainable = False
    ganLoss = gan.train_on_batch(x=xTrain, y=[realBool, yTrain])
    history['GAN_loss'].append(ganLoss[1])#对抗损失
    history['content_loss'].append(ganLoss[2])#内容损失

    trainAcc.append(calAcc(yTrain, yPred))
    if steps%1000 == 0:
        #打印损失
        dLoss = np.array(history['discriminator_loss'][-100:]).mean()
        gLoss = np.array(history['GAN_loss'][-100:]).mean()
        cLoss = np.array(history['content_loss'][-100:]).mean()
        print("----------------------------------------------------")
        print('step:%d dLoss:%.4f gLoss:%.4f cLoss:%.4f' % (steps, dLoss, gLoss, cLoss))
        trainAcc = np.mean(trainAcc)
        valiAcc = calDataAcc(valiData,batchSize)
        testAcc = calDataAcc(testData,batchSize)
        print("train acc:%.4f vali acc:%.4f test acc:%.4f"%(trainAcc,valiAcc,testAcc))
        testHistory["step"].append(steps)
        testHistory["train-acc"].append(trainAcc)
        testHistory["validate-acc"].append(valiAcc)
        testHistory["test-acc"].append(testAcc)
        trainAcc = []
        #保存历史信息
        historyDF = pd.DataFrame(history)
        historyDF["step"] = range(1,historyDF.shape[0]+1)
        historyDF.to_excel(resultDir+"/训练损失.xlsx",index=None)
        testHistoryDF = pd.DataFrame(testHistory)
        testHistoryDF.to_excel(resultDir+"/准确率.xlsx",index=None)
        #随机抽一张图进行测试保存
        index = np.random.randint(0,testData.shape[0])
        row = testData.iloc[index]
        x = cv2.imdecode(np.fromfile(row["input"], dtype=np.uint8),-1)
        x = normalizeImg(x)
        realImg = cv2.imdecode(np.fromfile(row["output"], dtype=np.uint8),-1)
        fakeImg = generator(np.stack([x]))
        fakeImg = reverseImg(fakeImg).numpy().astype(np.uint8)[0]
        #拼接 保存
        img = np.concatenate([fakeImg,realImg],axis=1)
        cv2.imencode('.png', img)[1].tofile(resultDir + '/test/%d.png'%steps)
        #保存模型
        if bestValAcc < valiAcc:
            generator.save_weights(staticDir + '/best_generator.h5')
            discriminator.save_weights(staticDir + '/best_discriminator.h5')
        generator.save_weights(staticDir + '/last_generator.h5')
        discriminator.save_weights(staticDir + '/last_discriminator.h5')
    steps += 1
