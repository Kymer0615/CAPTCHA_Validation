from torch.utils.data import Dataset, DataLoader
from os import getcwd, listdir
from os.path import join, isfile, exists, isdir
from pathlib import Path
import numpy as np
import copy
import cv2 as cv


class Dataset(Dataset):
    def __init__(self, rootDir, transform=None):
        self.rootDir = rootDir
        self.labelMapping = {}
        self.mapping = self.imgMapping()
        self.transform = transform

    def getLabelNum(self):
        return len({i[1] for i in self.mapping.values()})

    def getMapping(self):
        return {v: k for k, v in self.labelMapping.items()}

    def imgMapping(self):
        dic = {}
        index = 0
        for i in listdir(self.rootDir):
            if not isdir(join(self.rootDir, i)):
                continue
            for j in listdir(join(self.rootDir, i)):
                path = join(self.rootDir, i, j)
                if isfile(path) and ".png" in path:
                    label = str(Path(path).parent.absolute()).split('/')[-1]
                    if label not in self.labelMapping.keys():
                        self.labelMapping.update({label: len(self.labelMapping.values())})
                    dic.update({index: [path, self.labelMapping[label]]})
                    index += 1
                elif isfile(path) and ".png" not in path:
                    continue
        return copy.deepcopy(dic)

    def __len__(self):
        return len(self.mapping)

    def __getitem__(self, idx):
        image = cv.imread(self.mapping[idx][0])[:,:,1]
        image = np.array(image).reshape(32, 32, 1)
        label = self.mapping[idx][1]
        if self.transform:
            image = self.transform(image)

        return image, label
