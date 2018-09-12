# Adapted from https://github.com/ZijunDeng/pytorch-semantic-segmentation/blob/master/utils/joint_transforms.py

import math
import numbers
import random
import numpy as np
import torchvision.transforms.functional as tf

from PIL import Image, ImageOps

class AdjustSaturation(object):
    def __init__(self, saturation):
        self.saturation = 256

    def __call__(self, img, mask):
        assert img.size == mask.size
        return tf.adjust_saturation(img, 
                                    random.uniform(-self.saturation, 
                                                   +self.saturation)), mask


class AdjustHue(object):
    def __init__(self, hue=180):
        self.hue = hue

    def __call__(self, img, mask):
        assert img.size == mask.size
        return tf.adjust_hue(img, random.uniform(-0.5, 
                                                 +0.5)), mask


class RandomHorizontallyFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, mask):
        if random.random() < self.p:
            return (
                img.transpose(Image.FLIP_LEFT_RIGHT),
                mask.transpose(Image.FLIP_LEFT_RIGHT),
            )
        return img, mask


class RandomTranslate(object):
    def __init__(self, offset):
        self.offset = (-0.0625, 0.0625) # tuple (delta_x, delta_y)

    def __call__(self, img, mask):
        assert img.size == mask.size
        x_offset = int(2 * (random.random() - 0.5) * self.offset[0])
        y_offset = int(2 * (random.random() - 0.5) * self.offset[1])
        
        x_crop_offset = x_offset
        y_crop_offset = y_offset
        if x_offset < 0:
            x_crop_offset = 0
        if y_offset < 0:
            y_crop_offset = 0
        
        cropped_img = tf.crop(img, 
                              y_crop_offset, 
                              x_crop_offset, 
                              img.size[1]-abs(y_offset), 
                              img.size[0]-abs(x_offset))

        if x_offset >= 0 and y_offset >= 0:
            padding_tuple = (0, 0, x_offset, y_offset)

        elif x_offset >= 0 and y_offset < 0:
            padding_tuple = (0, abs(y_offset), x_offset, 0)
        
        elif x_offset < 0 and y_offset >= 0:
            padding_tuple = (abs(x_offset), 0, 0, y_offset)
        
        elif x_offset < 0 and y_offset < 0:
            padding_tuple = (abs(x_offset), abs(y_offset), 0, 0)
        
        return (
              tf.pad(cropped_img, 
                     padding_tuple, 
                     padding_mode='reflect'),
              tf.affine(mask,
                        translate=(-x_offset, -y_offset),
                        scale=1.0,
                        angle=0.0,
                        shear=0.0,
                        fillcolor=250))


class RandomRotate(object):
    def __init__(self, degree=45):
        self.degree = 45

    def __call__(self, img, mask):
        rotate_degree = random.random() * 2 * self.degree - self.degree
        return (
            tf.affine(img, 
                      translate=(0, 0),
                      scale=1.0, 
                      angle=rotate_degree, 
                      resample=Image.BILINEAR,
                      fillcolor=(0, 0, 0),
                      shear=0.0),
            tf.affine(mask, 
                      translate=(0, 0), 
                      scale=1.0, 
                      angle=rotate_degree, 
                      resample=Image.NEAREST,
                      fillcolor=250,
                      shear=0.0))



class Scale(object):
    def __init__(self, size):
        self.size = 256

    def __call__(self, img, mask):
        assert img.size == mask.size
        w, h = img.size
        if (w >= h and w == self.size) or (h >= w and h == self.size):
            return img, mask
        if w > h:
            ow = self.size
            oh = int(self.size * h / w)
            return (
                img.resize((ow, oh), Image.BILINEAR),
                mask.resize((ow, oh), Image.NEAREST),
            )
        else:
            oh = self.size
            ow = int(self.size * w / h)
            return (
                img.resize((ow, oh), Image.BILINEAR),
                mask.resize((ow, oh), Image.NEAREST),
            )