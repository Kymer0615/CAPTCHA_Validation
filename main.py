from os import listdir, remove, getcwd
from os.path import isdir, join, isfile

import numpy as np
import requests
from torch.utils.data import Dataset, DataLoader

import torch
from torchvision.transforms import transforms

from DataGathering.CAPTCHAgenerator import CAPTCHAgenerator
from matplotlib import pyplot as plt
from DataGathering.Preprocess import Preprocess
from Train import gb688Dataset
from pathlib import Path
import cv2 as cv

from Train.Varify import Varify

# 生成数据
a = CAPTCHAgenerator(200)
b = Preprocess()
images = b.preprocess(inputPath=getcwd()+"/DataGathering/RawImage/")
b.save_imgs(outputPath=getcwd() + "/DataGathering/PreprocessedImage/")

# for i in range(len(images)):
#     plt.subplot(10, 4, i + 1)
#     print(images[i].shape)
#     plt.imshow(images[i], cmap='gray')
#
# plt.show()


#测试
# r = requests.get('http://c.gb688.cn/bzgk/gb/gc?_1627954865562', params={
#             "USER_AGENT": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.164 Safari/537.36"})
# v = Varify('gb688')
# nparr = np.fromstring(r.content, np.uint8)
# img = cv.imdecode(nparr, cv.IMREAD_COLOR)
# plt.imshow(img, cmap='gray')
# plt.show()
# res=v.getResult(r.content)
# print(res)
