from Module.BuildModel import *
from Module.Common import *
from sklearn.utils import shuffle
from glob import glob
import numpy as np
import pandas as pd
import cv2
import os
#Organize input and output paths
inputDir = staticDir+"/png/raw(png)/raw/"
outputDir = staticDir+"/png/real(png)jet/real/"
names = os.listdir(inputDir)
batchSize = 50#Batch processing volume
#Organize testing training sets
data = pd.DataFrame({"name":names,"input":[inputDir+name for name in names],"output":[outputDir+name for name in names]})
data = shuffle(data)#shuffle data
trainData = data.iloc[:int(data.shape[0]*0.8)]
valiData = data.iloc[int(data.shape[0]*0.8):int(data.shape[0]*0.9)]
testData = data.iloc[int(data.shape[0]*0.9):]
del data

def collect(data,indexs,augment=0):
    '''
    Extract data from the dataset based on the index to form input and output
    :param data: data set
    :param indexs:index
    :param augment:data enhancement
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
        # rotation
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
    '''Calculate pixel accuracy'''
    yTrain = reverseImg(yTrain)
    yPred = reverseImg(yPred)
    acc = 1-np.sum(np.abs(yTrain - yPred))/(48*48*3*255*batchSize)
    return acc

def calDataAcc(data,batchSize):
    '''Calculate the pixel accuracy of a dataset'''
    acc = []
    for j in range(0, data.shape[0], batchSize):
        x, y = collect(data,range(j,min(j+batchSize,data.shape[0])),augment=0)
        yPred = generator.predict(x)
        acc.append(calAcc(y,yPred))
    acc = np.mean(acc)
    return acc

#Instantiating Networks
generator = buildGenerator()
discriminator = buildDiscriminator()
gan = buildGAN(generator,discriminator)

#started training
steps = 0#iterations

history = {"discriminator_loss":[],"GAN_loss":[],"content_loss":[]}
testHistory = {"step":[],"train-acc":[],"validate-acc":[],"test-acc":[]}
trainAcc = []
bestValAcc = 0
while True:

    #Random extraction of training sets
    indexs = np.random.choice(trainData.shape[0],batchSize,replace=False)
    xTrain,yTrain = collect(trainData,indexs,0)
    yPred = generator.predict(xTrain)
    #Authentic labels
    realBool = np.random.uniform(0.7,1,size=(batchSize,))
    fakeBool = np.random.uniform(0,0.3,size=(batchSize,))
    #Discriminator training
    discriminator.trainable = True
    dRealLoss = discriminator.train_on_batch(x=yTrain, y=realBool)
    dFakeLoss = discriminator.train_on_batch(x=yPred, y=fakeBool)
    #Discriminatory loss
    history['discriminator_loss'].append(0.5 * (dRealLoss + dFakeLoss))

    #Generator training
    discriminator.trainable = False
    ganLoss = gan.train_on_batch(x=xTrain, y=[realBool, yTrain])
    history['GAN_loss'].append(ganLoss[1])#Adversarial loss
    history['content_loss'].append(ganLoss[2])#Content loss

    trainAcc.append(calAcc(yTrain, yPred))
    if steps%23 == 0:
        #Printing loss
        dLoss = np.array(history['discriminator_loss'][-23:]).mean()
        gLoss = np.array(history['GAN_loss'][-23:]).mean()
        cLoss = np.array(history['content_loss'][-23:]).mean()
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
        #Save Historical Information
        historyDF = pd.DataFrame(history)
        historyDF["step"] = range(1,historyDF.shape[0]+1)
        historyDF.to_excel(resultDir+"/Training_loss.xlsx",index=None)
        testHistoryDF = pd.DataFrame(testHistory)
        testHistoryDF.to_excel(resultDir+"/Accuracy.xlsx",index=None)
        #Randomly select one image for testing and saving
        index = np.random.randint(0,testData.shape[0])
        row = testData.iloc[index]
        x = cv2.imdecode(np.fromfile(row["input"], dtype=np.uint8),-1)
        raw = x
        x = normalizeImg(x)
        realImg = cv2.imdecode(np.fromfile(row["output"], dtype=np.uint8),-1)
        fakeImg = generator(np.stack([x]))
        fakeImg = reverseImg(fakeImg).numpy().astype(np.uint8)[0]
        #Splice+Save
        img = np.concatenate([raw,fakeImg,realImg],axis=1)
        cv2.imencode('.png', img)[1].tofile(resultDir + '/test/%d.png'%(steps/23))
        #Save a model
        if bestValAcc < valiAcc:
            generator.save_weights(staticDir + '/best_generator.h5')
            discriminator.save_weights(staticDir + '/best_discriminator.h5')
        generator.save_weights(staticDir + '/last_generator.h5')
        discriminator.save_weights(staticDir + '/last_discriminator.h5')
    steps += 1
