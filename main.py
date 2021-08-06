from time import time

import numpy as np
import requests
from matplotlib import pyplot as plt
import cv2 as cv

from CAPTCHA_Validation.Utils.CAPTCHAgenerator import CAPTCHAgenerator
from CAPTCHA_Validation.Utils.Preprocess import Preprocess
from Utils.Varify import Varify

from CAPTCHA_Validation.Train.Train import Train
# 生成数据
# a = CAPTCHAgenerator(num=1, sourceUrl=["http://c.gb688.cn/bzgk/gb/gc?_1627954865562"], name='gb688')
# b = Preprocess('gb688')
# images = b.preprocess()
# b.save_imgs()

# 训练
t = Train('gb688')
t.training_loop()


# 测试
# r = requests.get('http://c.gb688.cn/bzgk/gb/gc?_1627954865562', params={
#             "USER_AGENT": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.164 Safari/537.36"})
# print(Varify('gb688').getResult(r.content))
# nparr = np.frombuffer(r.content, np.uint8)
# img = cv.imdecode(nparr, cv.IMREAD_COLOR)
# plt.imshow(img)
# plt.show()

# 遍历所有图片
# root_dir = "/Users/chenziyang/Documents/Ziyang/Crawler/中石化.nosync/验证码/CAPTCHA_Validation/Data"
# for i in listdir(root_dir):
#     if not isdir(join(root_dir, i)):
#         continue
#     for j in listdir(join(root_dir, i)):
#         path = join(root_dir, i, j)
#         if isfile(path) and ".png" in path:
#             temp = cv.imread(path)
#             if temp is not None:
#                 remove(path)
#                 temp = cv.resize(temp, (32, 32))
#                 temp = temp[:,:,1]
#                 # print(temp.shape)
#                 cv.imwrite(path,temp)
#         elif isfile(path) and ".png" not in path:
#             continue

