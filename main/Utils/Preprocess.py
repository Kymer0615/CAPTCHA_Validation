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
        self.outputPath = getcwd() + "/resource/DataGathering/PreprocessedImage/" + name + "/"
        self.jsonPath = getcwd() + "/resource/Configs/" + name + ".json"
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
        print(thresh.shape)
        if len(thresh.shape) > 2: thresh = thresh[:, :, 1]
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

    @staticmethod
    def constant_padding(img):
        BLUE = [255, 0, 0]
        return cv.copyMakeBorder(img, 10, 10, 10, 10, cv.BORDER_CONSTANT, value=BLUE)

    @staticmethod
    def erosion(img, x, y, iterations):
        kernel = np.ones((x, y), np.uint8)
        return cv.erode(img, kernel, iterations=iterations)

    @staticmethod
    def contours_slice(img, digtNum):
        images = []
        image, contours, hier = cv.findContours(img, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
        res = []
        for cont in contours:
            x, y, w, h = cv.boundingRect(cont)
            if x != 0 and y != 0 and w * h >= 100:
                res.append([x, y, w, h])
        print(len(res))
        if len(res) < digtNum:
            res.sort(key=lambda a: a[2])
            width = {i[2]: i for i in res}
            width = [i / min(width) for i in width]
            if digtNum == 4:
                if len(res) == 3:
                    temp = res[-1]
                    res.remove(temp)
                    res.append((temp[0], temp[1], int(temp[2] / 2), temp[0]))
                    res.append((temp[0] + int(temp[2] / 2), temp[1], int(temp[2] / 2), temp[0]))
                if len(res) == 2:
                    pass

        print(len(res))
        for (x, y, w, h) in res:
            image = cv.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
        # elif counter > digtNum:
        # image = cv.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)

        plt.imshow(image)
        plt.show()
        # color = (0, 255, 0)

        # for c in contours:  # 遍历轮廓

        return images

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
