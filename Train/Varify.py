import sys

import numpy as np
import torch
from torchvision.transforms import transforms
from os import getcwd, listdir
from os.path import join, isfile, exists, isdir
from .Lenet import LeNet5
from .gb688Dataset import gb688Dataset
from CAPTCHA_Validation.DataGathering.Preprocess import Preprocess
import cv2 as cv


# import utils
class Varify:
    def __init__(self, name):
        self.name = name
        self.checkpointDir = getcwd() + "/checkpoint/" + self.name + ".pth"
        self.dataset = gb688Dataset(root_dir=getcwd() + "/" + "Data")
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = LeNet5(self.dataset.getLabelNum())
        self.model.load_state_dict(torch.load(self.checkpointDir))
        self.model.eval()
        self.img = None

    def getResult(self, input):
        p = Preprocess()
        if isinstance(input, bytes):
            nparr = np.fromstring(input, np.uint8)
            self.img = cv.imdecode(nparr, cv.IMREAD_COLOR)
        labels = []
        results = ""
        for i in p.preprocess(files=[self.img]):
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914), (0.2023))
            ])
            i = transform(i).reshape(1, 1, 32, 32)
            i.to(self.device)
            _, y_prob = self.model(i)
            _, predicted_labels = torch.max(y_prob, 1)
            labels.append(predicted_labels)
            mapping = self.dataset.getMapping()
        for i in labels:
            results=results+(mapping[i.tolist()[0]])
        return results
