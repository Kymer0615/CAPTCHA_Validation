import CAPTCHAgenerator
from matplotlib import pyplot as plt
import cv2 as cv
import numpy as np
from os import listdir, getcwd
from os.path import isfile, join
import uuid


class Preprocess:
    def __init__(self, inputPath, dir=None):
        self.inputPath = inputPath
        self.files = [join(inputPath, f) for f in listdir(inputPath) if isfile(join(inputPath, f)) and ".png" in f]
        self.images = []
        self.CWD = getcwd()
        self.dir = dir if dir else "./ProcessedImage/"

    def binary(self, img):
        img = cv.cvtColor(img, cv.COLOR_BAYER_RG2GRAY)
        img = cv.medianBlur(img, 3)
        _, thresh = cv.threshold(img, 127, 255, cv.THRESH_OTSU)
        # img = cv.Canny(img, 100, 200)
        return thresh

    def slice_image(self, img, num, h, w):
        images = []
        w_increment = w / num
        for i in range(1, num + 1):
            images.append(img[0: int(h), int((i - 1) * w_increment):int(i * w_increment)])
        return images

    def find_contours(self, img):
        image, contours, hier = cv.findContours(img, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
        # color = (0, 255, 0)
        for c in contours:  # 遍历轮廓
            if cv.contourArea(c) > 50:
                x, y, w, h = cv.boundingRect(c)
            # cv.rectangle(img, (x, y), (x + w, y + h), color, 1)
            print(cv.boundingRect(c))
            # rect = cv.minAreaRect(c)  # 生成最小外接矩形
            # box_ = cv.boxPoints(rect)
            # h = abs(box_[3, 1] - box_[1, 1])
            # w = abs(box_[3, 0] - box_[1, 0])
            # print("宽，高", w, h)
            # # 只保留需要的轮廓
            # if (h > 3000 or w > 2200):
            #     continue
            # if (h < 2500 or w < 1500):
            #     continue
            # box = cv.boxPoints(rect)  # 计算最小面积矩形的坐标
            # box = np.int0(box)  # 将坐标规范化为整数
            return x, y, w, h
            # 绘制矩形
            # cv2.drawContours(img, [box], 0, (255, 0, 255), 3)
        print("轮廓数量", len(contours))

    def preprocess(self):
        print(self.files)
        for i in self.files:
            img = cv.imread(i, 0)
            img = self.binary(img)
            # self.find_contours(img)
            # TODO 这里切割次数 和 具体数据可能需要更改
            imgs = self.slice_image(img, 4, 60, 150)
            for i in imgs: self.images.append(i)
        return self.images

    def save_imgs(self):
        for i in self.images:
            uuid_str = uuid.uuid4().hex
            cv.imwrite((self.get_path() + uuid_str + ".png"),i)

    def get_path(self):
        return join(self.CWD, self.dir)
