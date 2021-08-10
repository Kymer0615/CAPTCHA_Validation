from io import BytesIO

import numpy as np
import requests
from matplotlib import pyplot as plt
import cv2 as cv
from main.Utils.Varify import Varify
from main.Utils.FontsGenerator import FontsGenerator
# 生成数据
#   -从网页获取
# a = CAPTCHAgenerator(num=1, sourceUrl=["http://c.gb688.cn/bzgk/gb/gc?_1627954865562"], name='gb688')
# b = Preprocess('gb688')
# images = b.preprocess()
# b.save_imgs()
#   -通过字体生成
c=FontsGenerator(4,name="fontSet0",num=1).generateDataset()
# nparr = np.frombuffer(c, np.uint8)
# img = cv.imdecode(nparr, cv.IMREAD_COLOR)
# plt.imshow(img)
# plt.show()



# 训练
# t = Train('gb688')
# t.training_loop()


# 测试
# r = requests.get('http://c.gb688.cn/bzgk/gb/gc?_1627954865562', params={
#             "USER_AGENT": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.164 Safari/537.36"})
# print(Varify('gb688').getResult(r.content))
# nparr = np.frombuffer(r.content, np.uint8)
# img = cv.imdecode(nparr, cv.IMREAD_COLOR)
# plt.imshow(img)
# plt.show()

# 遍历所有图片
# root_dir = "/Users/chenziyang/Documents/Ziyang/Crawler/中石化.nosync/CAPTCHA/CAPTCHA_Validation/Data"
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

