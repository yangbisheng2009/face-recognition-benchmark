import os
import cv2
import sys
import numpy as np
from PIL import Image
from collections import defaultdict

import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.utils import data
from torchvision import transforms as T
import torchvision

class Dataset(data.Dataset):
    def __init__(self, root, input_shape=(3, 112, 112)):
        self.input_shape = input_shape

        imgs = []
        self.name2id = defaultdict()
        self.id2name = defaultdict()
        for index, folder in enumerate(os.listdir(root)):
            self.name2id[folder] = index
            self.id2name[index] = folder
            for fname in os.listdir(os.path.join(root, folder)):
                imgs.append(os.path.join(root, folder, fname))
        print(self.name2id)
        self.imgs = np.random.permutation(imgs)

        self.transforms = T.Compose([
                T.Resize(self.input_shape[1:]),
                # T.RandomCrop(self.input_shape[1:]),
                T.ColorJitter(brightness=0.125, contrast=0.125, saturation=0.125),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    def __getitem__(self, index):
        img_path = self.imgs[index]

        label_name = img_path.split('/')[-2] # folder name
        #label = int(self.name2id[label_name]) # label need to be int
        label = int(label_name)

        data = Image.open(img_path).convert('RGB')
        data = self.transforms(data)
        return data, label, label_name

    def __len__(self):
        return len(self.imgs)


if __name__ == '__main__':
    dataset = Dataset(root='../train_data/new_training/train/',
                      phase='train',
                      input_shape=(1, 112, 112))

    trainloader = data.DataLoader(dataset, batch_size=10)
    for i, (data, label) in enumerate(trainloader):
        # imgs, labels = data
        # print imgs.numpy().shape
        # print data.cpu().numpy()
        # if i == 0:
        img = torchvision.utils.make_grid(data).numpy()
        # print img.shape
        # print label.shape
        # chw -> hwc
        img = np.transpose(img, (1, 2, 0))
        # img *= np.array([0.229, 0.224, 0.225])
        # img += np.array([0.485, 0.456, 0.406])
        img += np.array([1, 1, 1])
        img *= 127.5
        img = img.astype(np.uint8)
        img = img[:, :, [2, 1, 0]]

        plt.imshow(img)
        plt.show()

        # cv2.imshow('img', img)
        # cv2.waitKey()
        break
        # dst.decode_segmap(labels.numpy()[0], plot=True)
