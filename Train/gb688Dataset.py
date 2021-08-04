from torch.utils.data import Dataset, DataLoader
from os import getcwd, listdir
from os.path import join, isfile, exists, isdir
from pathlib import Path
import numpy as np
from torchvision.io import read_image
import copy
import cv2 as cv


class gb688Dataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.labelMapping = {}
        self.mapping = self.imgMapping()
        self.transform = transform

    def imgMapping(self):
        dic = {}
        index = 0
        for i in listdir(self.root_dir):
            if not isdir(join(self.root_dir, i)):
                continue
            for j in listdir(join(self.root_dir, i)):
                path = join(self.root_dir, i, j)
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
        image = read_image(self.mapping[idx][0])
        image = np.array(image).reshape(60,38,3)
        label = self.mapping[idx][1]
        if self.transform:
            image = self.transform(image)

        return image, label
