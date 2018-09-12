import os
import collections
import torch
import torchvision
import numpy as np
import scipy.misc as m
import matplotlib.pyplot as plt

from torch.utils import data
import cv2

from augmentations import *


augObj = {'hue': AdjustHue,
           'saturation': AdjustSaturation,
           'hflip': RandomHorizontallyFlip,
           'scale': Scale,
           'rotate': RandomRotate,
           'translate': RandomTranslate,
           }

augmentationsArray = []

for key, param in augObj.items():
        augmentationsArray.append(augObj[key](param))

class Dataloader(data.Dataset):
	def __init__(self, root, input_data, labels, is_transform=False, img_size=None, augmentations=None, img_norm=True):
		self.root = root
		self.data = input_data
		self.labels = labels
		self.img_size = [256, 256]
		self.is_transform = is_transform
		self.augmentations = augmentations
		self.img_norm = img_norm
		self.mean = np.array([104.00699, 116.66877, 122.67892])
		self.n_classes = 1
		self.files = collections.defaultdict(list)

		file_list = os.listdir(root + self.data)
		self.files = file_list


	def __len__(self):
		return len(self.files)

	def __getitem__(self, index):
		img_name = self.files[index]
		img_path = self.root + self.data + img_name
		lbl_path = self.root + self.labels + img_name

		img = cv2.imread(img_path)
		img = np.array(img, dtype=np.uint8)

		lbl = cv2.imread(lbl_path, 0)
		lbl = np.array(lbl, dtype=np.uint8)

		if self.augmentations is not None:
		    img, lbl = self.augmentations(img, lbl)

		if self.is_transform:
		    img, lbl = self.transform(img, lbl)

		return img, lbl
		


	def transform(self, img, lbl):
	    img = m.imresize(
	        img, (self.img_size[0], self.img_size[1])
	    )  # uint8 with RGB mode
	    img = img[:, :, ::-1]  # RGB -> BGR
	    img = img.astype(np.float64)
	    ret, lbl = cv2.threshold(lbl, 127, 1, cv2.THRESH_BINARY)
	    
	    img -= self.mean
	    if self.img_norm:
	        # Resize scales images from 0 to 255, thus we need
	        # to divide by 255.0
	        img = img.astype(float) / 255.0
	    # NHWC -> NCHW
	    img = img.reshape((1, self.img_size[0], self.img_size[1], 3))
	    img = img.transpose(0, 3, 1, 2)

	    img = torch.from_numpy(img).float()
	    lbl = torch.from_numpy(lbl).long()

	    return img, lbl


	def decode_segmap(self, temp, plot=False):
		temp = np.squeeze(temp)
		c = temp.copy()

		c[temp <= 0.5] = 0
		c[temp > 0.5] = 255

		cv2.imshow('image', c)
		cv2.waitKey(0)
		return c


class Compose(object):
    def __init__(self, augmentations):
        self.augmentations = augmentations
        self.PIL2Numpy = False

    def __call__(self, img, mask):
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img, mode="RGB")
            mask = Image.fromarray(mask, mode="L")
            self.PIL2Numpy = True

        assert img.size == mask.size
        for a in self.augmentations:
            img, mask = a(img, mask)
            assert img.size == mask.size

        if self.PIL2Numpy:
            img, mask = np.array(img), np.array(mask, dtype=np.uint8) 

        return img, mask
