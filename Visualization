import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import os
from Module.BuildModel import *
from Module.Common import *
from sklearn.utils import shuffle
from glob import glob
import numpy as np
import pandas as pd
import cv2
import os

plt.rcParams['font.sans-serif']="KaiTi"
plt.rcParams['axes.unicode_minus']=False

plt.figure(dpi=3000)
df = pd.read_excel(resultDir+'\\Accuracy.xlsx', engine='openpyxl')

y_data1 = np.array(df[['train-acc']])
y1 = tf.cast(y_data1, tf.float32)
y_data2 = np.array(df[['validate-acc']])
y2 = tf.cast(y_data2, tf.float32)
y_data3 = np.array(df[['test-acc']])
y3 = tf.cast(y_data3, tf.float32)


plt.plot(y1,label='training-accuracy',color='green',alpha=1)
plt.plot(y2,label='validation-accuracy',color='red',alpha=1)
plt.plot(y3,label='test-accuracy',color='blue',alpha=1)

plt.xlim(0,3000)
plt.ylim(0.6,1)
plt.xlabel('Epoches',fontsize=10)
plt.ylabel('Pixel-level Accuracy',fontsize=10)

plt.legend()
plt.grid(axis='both',color='black',linestyle=':',linewidth=0.3)
plt.show()