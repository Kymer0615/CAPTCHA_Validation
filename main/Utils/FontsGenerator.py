from os import getcwd, listdir
import random
import string
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from captcha.image import ImageCaptcha
import uuid
from main.Utils.Preprocess import Preprocess
import cv2 as cv


class FontsGenerator:
    def __init__(self, digitDum, name, num):
        self.num = num
        self.name = name
        self.inputPath = getcwd() + "/resource/Fonts/" + self.name
        self.outputPath = getcwd() + "/resource/Data/" + self.name
        Path(self.outputPath).mkdir(parents=True, exist_ok=True)
        self.image = ImageCaptcha(fonts=self.getFontPaths())
        self.preprocess = Preprocess(name)
        self.digitDum = digitDum

    def getFontPaths(self):
        fonts = []
        for i in listdir(self.inputPath):
            if '.ttf' in i:
                fonts.append(self.inputPath+'/'+i)
        return fonts

    def generateImg(self):
        captcha = ''.join(
            random.choices(string.ascii_lowercase + string.ascii_uppercase + string.digits, k=self.digitDum))
        img = self.image.generate(captcha).getvalue()
        nparr = np.frombuffer(img, np.uint8)
        img = cv.imdecode(nparr, cv.IMREAD_COLOR)
        return captcha, img

    def generateDataset(self):
        for i in range(self.num):
            captcha, img = self.generateImg()
            plt.imshow(img)
            plt.show()
            imgList = [i for i in self.preprocess.preprocess([img])]
            # for i in imgList:
            #     # a = plt.imread(i)
            #     print(i.shape)
            #     plt.imshow(i)
            #     plt.show()
            #     break
            # captcha = list(captcha)
            # for i in captcha:
            #     labelPath = self.outputPath + "/" + self.name + "/" + i
            #     Path(labelPath).mkdir(parents=True, exist_ok=True)
            #     uuid_str = uuid.uuid4().hex
            #     for img in imgList:
            #         file = open(labelPath + "/" +
            #                     uuid_str + ".png", "wb")
            #         file.write(img)
            #         file.close()
