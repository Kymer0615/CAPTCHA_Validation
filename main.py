from os import listdir, remove
from os.path import isdir, join, isfile
from torch.utils.data import Dataset, DataLoader

import torch

from DataGathering.CAPTCHAgenerator import CAPTCHAgenerator
from matplotlib import pyplot as plt
from DataGathering.Preprocess import Preprocess
from Train import gb688Dataset
from pathlib import Path
import cv2 as cv

# 生成数据
# a = CAPTCHAgenerator(1)
# b = Preprocess(a.get_path())
# images = b.preprocess()
# b.save_imgs()

# for i in range(len(images)):
    # plt.subplot(10, 4, i + 1)
    # print(images[i].shape)
    # plt.imshow(images[i], cmap='gray')

# plt.show()





root_dir = "/Users/chenziyang/Documents/Ziyang/Crawler/中石化.nosync/验证码/CAPTCHA_Validation/Data"
for i in listdir(root_dir):
    if not isdir(join(root_dir, i)):
        continue
    for j in listdir(join(root_dir, i)):
        path = join(root_dir, i, j)
        if isfile(path) and ".png" in path:
            temp = cv.imread(path)
            if temp is not None:
                # remove(path)
                # temp = cv.resize(temp, (38, 60))
                # temp.reshape(3,38,60)
                print(temp.shape)
                # cv.imwrite(path,temp)
        elif isfile(path) and ".png" not in path:
            continue


# c = gb688Dataset.gb688Dataset(Path("Data").absolute())
# d = DataLoader(c, 5, shuffle=True)
# for batch_idx, (data, labels) in enumerate(d):
#         print(data.shape)
# print(torch.Size(c[0][0]))
# print(len(c))