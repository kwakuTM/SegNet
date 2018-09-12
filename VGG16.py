import torch.nn as nn

def vgg16_init(vgg16):
	covNets = [down1, down2, down3, down4, down5]
	vgg16Features = list(vgg16.features.children())

	vgg16Layers = []
	for _layer in vgg16Features:
		if isinstance(_layer, nn.Conv2d):
			vgg16Layers.append(_layer)

	merged_layers = []
	for idx, conv in enumerate(covNets):
		if idx < 2:
			units = [conv.conv1, conv.conv2]
		else:
			units = [
				conv.conv1,
				conv.conv2,
				conv.conv3
			]
		for _unit in units:
			for _layer in unit:
				if isinstance(_layer, nn.Conv2d):
					merged_layers.append(_layer)

	assert len(vgg16Layers) == len(merged_layers)

	for l1, l2 in zip(vgg16Layers, merged_layers):
		if isinstance(l1, nn.Conv2d) and isinstance(l2, nn.Conv2d):
			assert l1.weight.size() == l2.weight.size()
			assert l1.bias.size() == l2.bias.size()
			l2.weight.data = l1.weight.data
			l2.weight.data = l1.bias.data