import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from torch.autograd import Variable

class conv2DBatchNormRelu(nn.Module):
	def __init__(
	    self,
	    in_channels,
	    n_filters,
	    k_size,
	    stride,
	    padding,
	    bias=True,
	    dilation=1,
	    with_bn=True,
	):
	    super(conv2DBatchNormRelu, self).__init__()

	    conv_mod = nn.Conv2d(
	    	int(in_channels),
	    	int(n_filters),
	    	kernel_size=k_size,
	    	padding=padding,
	    	stride=stride,
	    	bias=bias,
	    	dilation=1,
	    	)

	    self.cbr_unit = nn.Sequential(
	    	conv_mod, nn.BatchNorm2d(int(n_filters)), nn.ReLU(inplace=True)
	    	)

	def forward(self, inputs):
	    outputs = self.cbr_unit(inputs)
	    return outputs

class encoder1(nn.Module):
	def __init__(self, in_size, out_size):
		super(encoder1, self).__init__()
		self.conv1 = conv2DBatchNormRelu(in_size, out_size, 3, 1, 1)
		self.conv2 = conv2DBatchNormRelu(out_size, out_size, 3, 1, 1)
		self.maxpool2d = nn.MaxPool2d(2, 2, return_indices=True)

	def forward(self, x):
		x = self.conv1(x)
		x = self.conv2(x)
		unpooled_shape = x.size()
		x, ind = self.maxpool2d(x)
		return x, ind, unpooled_shape

class encoder2(nn.Module):
	def __init__(self, in_size, out_size):
		super(encoder2, self).__init__()
		self.conv1 = conv2DBatchNormRelu(in_size, out_size, 3, 1, 1)
		self.conv2 = conv2DBatchNormRelu(out_size, out_size, 3, 1, 1)
		self.conv3 = conv2DBatchNormRelu(out_size, out_size, 3, 1, 1)
		self.maxpool2d = nn.MaxPool2d(2, 2, return_indices=True)

	def forward(self, x):
		x = self.conv1(x)
		x = self.conv2(x)
		x = self.conv3(x)
		unpooled_shape = x.size()
		x, ind = self.maxpool2d(x)
		return x, ind, unpooled_shape

class decoder1(nn.Module):
	def __init__(self, in_size, out_size):
		super(decoder1, self).__init__()
		self.unpool = nn.MaxUnpool2d(2, 2)
		self.conv1 = conv2DBatchNormRelu(in_size, in_size, 3, 1, 1)
		self.conv2 = conv2DBatchNormRelu(in_size, out_size, 3, 1, 1)

	def forward(self, x, ind, out_shape):
		outputs = self.unpool(input=x, indices=ind, output_size=out_shape)
		outputs = self.conv1(outputs)
		outputs = self.conv2(outputs)
		return outputs			

class decoder2(nn.Module):
	def __init__(self, in_size, out_size):
		super(decoder2, self).__init__()
		self.unpool = nn.MaxUnpool2d(2, 2)
		self.conv1 = conv2DBatchNormRelu(in_size, in_size, 3, 1, 1)
		self.conv2 = conv2DBatchNormRelu(in_size, in_size, 3, 1, 1)
		self.conv3 = conv2DBatchNormRelu(in_size, out_size, 3, 1, 1)

	def forward(self, x, ind, out_shape):
		outputs = self.unpool(input=x, indices=ind, output_size=out_shape)
		outputs = self.conv1(outputs)
		outputs = self.conv2(outputs)
		outputs = self.conv3(outputs)
		return outputs