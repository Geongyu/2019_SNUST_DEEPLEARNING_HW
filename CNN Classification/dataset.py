import os
import torch
import PIL
from PIL import Image
from torchvision import transforms, utils
from torch.utils.data import DataLoader
import numpy as np
from torch.utils.data import Dataset

class MNIST(Dataset):
    def __init__(self, data_dir, aug=False):
        self.data_dir = data_dir
        self.transforms_tensor = transforms.ToTensor()
        self.resize = transforms.Resize((32, 32))
        self.transfroms_normalize = transforms.Normalize((0.1307, ), (0.3081, ))
        self.transforms_aug = transforms.ColorJitter(brightness=(0.9, 1.1))
        file_list = os.listdir(data_dir)
        file_list.sort()
        self.aug = aug
        self.file_list = file_list

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img = Image.open(self.data_dir +  "/" + self.file_list[idx])
        img = self.resize(img)
        if self.aug == True :
            img = self.transforms_aug(img)
        img = self.transforms_tensor(img)
        img = self.transfroms_normalize(img)

        label = self.file_list[idx].split("_")[-1]
        label = label.split(".")[0]
        label = int(label)
        return img, label

if __name__ == '__main__':

    MNIST("../data/train_mnist/train")
