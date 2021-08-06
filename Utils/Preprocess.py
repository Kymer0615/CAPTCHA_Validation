import logging
logging.getLogger().setLevel(logging.INFO)
import numpy as np
import cv2 as cv
import uuid
from os import listdir, getcwd
from os.path import isfile, join, isdir
from pathlib import Path
import json

from matplotlib import pyplot as plt


class Preprocess:
    def __init__(self, name):
        self.name = name
        self.images = []
        self.outputPath = getcwd() + "/DataGathering/PreprocessedImage/" + name + "/"
        self.jsonPath = getcwd() + "/Configs/" + name + ".json"
        Path(self.outputPath).mkdir(parents=True, exist_ok=True)

    @staticmethod
    def cvtColor(img, arg0):
        return cv.cvtColor(img, arg0)

    @staticmethod
    def medianBlur(img, arg0):
        return cv.medianBlur(img, arg0)

    @staticmethod
    def threshold(img, arg0, arg1, arg2):
        _, thresh = cv.threshold(img, arg0, arg1, arg2)
        return thresh

    @staticmethod
    def resize(imgs):
        return [cv.resize(i, (32, 32)) for i in imgs]

    @staticmethod
    def slice_image(img, num, h, w):
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

    def preprocess(self, files=None):
        if not files:
            inputPath = getcwd() + "/DataGathering/RawImage/" + self.name
            if isdir(inputPath):
                files = [join(inputPath, f) for f in listdir(inputPath) if isfile(join(inputPath, f)) and ".png" in f]
            else:
                logging.warning('Raw data have not been generated!')
                return
        for img in files:
            if isinstance(img, str):
                img = cv.imread(img, 0)
            elif isinstance(img, np.ndarray):
                img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            for i in self.execFromJson(img):
                self.images.append(i)
        return self.images

    def save_imgs(self):
        for i in self.images:
            uuid_str = uuid.uuid4().hex
            cv.imwrite((self.outputPath + uuid_str + ".png"), i)

    def execFromJson(self, img):
        with open(self.jsonPath) as f:
            configs = json.load(f)["Preprocessing"]
        for i in configs.keys():
            if len(configs[i]['args']) > 0:
                methodStr = "Preprocess.%s(img," % i
                paramStr = "".join([j + "," for j in configs[i]['args']])[:-1]
                methodStr = methodStr + paramStr + ')'
                logging.info("Current preprocessing method: %s" % methodStr)
                img = eval(methodStr)
            else:
                methodStr = "Preprocess.%s(img)" % i
                logging.info("Current preprocessing method: %s" % methodStr)
                img = eval(methodStr)
        return img
