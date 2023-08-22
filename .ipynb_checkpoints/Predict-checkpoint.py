from Module.BuildModel import *
from Module.Common import *
from glob import glob
import numpy as np
import pandas as pd
import cv2
import os

generator = buildGenerator()
generator.load_weights(staticDir + '/best_generator.h5')
path = input("请拖入图片:")
x = cv2.imdecode(np.fromfile(path, dtype=np.uint8), -1)
x = normalizeImg(x)
fakeImg = generator(np.stack([x]))
fakeImg = reverseImg(fakeImg).numpy().astype(np.uint8)[0]
cv2.imshow("",fakeImg)
cv2.waitKey(0)