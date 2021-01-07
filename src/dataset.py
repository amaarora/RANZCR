import os

import cv2
import numpy as np
import torch
from PIL import Image


class Ranzcr_Dataset():
    """
    Dataset class that returns image as numpy array and labels as torch tensor.
    """

    def __init__(self, df, data_dir, aug=None):
        self.df = df
        self.labels = self.df.iloc[:, 1:12].values
        self.aug = aug

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.loc[index]
        img = cv2.imread(row.file_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.aug is not None:
            img = self.aug(image=img)['image']

        img = img.astype(np.float32)
        img = img.transpose(2, 0, 1)
        label = torch.tensor(self.labels[index]).float()
        return {'image': img, 'label': label}
