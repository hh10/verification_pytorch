""" Contains functions defining the various large, small, wide, residual models being used. """
import torch
import torch.nn as nn
import math
from torchvision.models.vgg import VGG
from convex_adversarial import DenseSequential as DS 
from convex_adversarial import Dense 
from torchvision import models as torch_models
from torchsummary import summary
import seaborn as sns

output = {}
f = 1 #0.125
SMALL = True

def get_model(m, d, Ni=None, Nc=None, Nic=None, pretrained=False): 
	""" Calls a specific attack method. The models currently in this file can be verified by wongs method using a 4GB RAM GPU in a span of 2days. TODO: improve method on both fronts: memory and speed. 
	
	#Arguments:
	-m: model to be trained/usd.
	-d: dataset on which m is to be trained/evaluated
	
	#Returns:
	-model (nn.Sequential type)
	"""
	print('model loading: ', m)
	if Ni==None or Nc==None or Nic==None:
		raise ValueError("Some parameters not told about the model, CHECK!!.")
	if not pretrained:
		if m == "2D":
			model = linear_2D(Ni, 100, Nc).cuda()
		elif m == "linear_mnist":
			model = linear_mnist_model().cuda()
		elif m == "conv_2layer":
			model = conv_2layer_model(Ni, Nic, Nc).cuda()
		elif m == "conv_2layer_GCM":
			model = conv_2layer_gradcam_model(Ni, Nic, Nc).cuda()
		elif m == "conv_4layer":
			model = conv_4layer_model(Ni, Nic, Nc).cuda()
		elif m == "conv_4layer_GCM":
			model = conv_4layer_gradcam_model(Ni, Nic, Nc).cuda()
		elif m == "conv_deep":
			model = conv_deep_model(Ni, Nic, Nc).cuda()
		elif m == "conv_wide":
			model = conv_wide_model(Ni, Nic, Nc).cuda() 
		elif m == "cifar_resnet":
			model = cifar_model_resnet().cuda()
		elif m == "imagenet_resnet":
			model = imagenet_model_resnet().cuda()
		elif m == "resnet18":
			model = resnet18(Ni, Nic, Nc).cuda()
		elif m == "resnet34":
			model = resnet34(Ni, Nic, Nc).cuda()
		elif m == "segm_imagenet_resnet":
			model = FCNs(pretrained_net=imagenet_model_resnet().cuda(), n_class=Nc, output=False).cuda()
		elif m == "segm_resnet18":
			model = FCNs(pretrained_net=ResNet(Ni=512, Nic=3, layers = [2,2,2,2,2], num_classes=Nc, segm=True).cuda(), n_class=Nc).cuda()
		elif m == "segm_small_resnet":
			model = FCN_SMALL(pretrained_net=ResNetSmall(Ni=64, Nic=3, layers = [2,2,2,2,2], num_classes=Nc, segm=True).cuda(), n_class=Nc).cuda()
		elif m == "segm_VGG":
			model = FCNs(pretrained_net=VGGNet(requires_grad=True).cuda(), n_class=Nc).cuda()
		else:
			raise ValueError("Untrained model not found: {}".format(m))
	else:
		if m == 'vgg19':
			modl = torch_models.vgg19(pretrained=pretrained).cuda()
			summary(modl, (Nic, Ni, Ni))
			layers_list = list(modl.features.children())
			layers_list.append(Flatten())
			layers_list.extend(modl.classifier.children())
			model = nn.Sequential(*layers_list)
		else:
			raise ValueError("Pretrained model not found: {}".format(m))
	return model
		
def get_target_layer(m, pretrained):
	if not pretrained:
		if m == "2D":
			return ["5"]
		elif m == "linear_mnist":
			return ["6"]
		elif m in ["conv_2layer", "conv_2layer_GCM"]:
			return ["3"]
		elif m in ["conv_4layer", "conv_4layer_GCM"]:
			return ["7"]
		elif m == "conv_deep":
			return ["7"]
		elif m == "conv_wide":
			return ["3"]
		elif m == "cifar_resnet":
			pass
		elif m == "imagenet_resnet":
			pass
		elif m == "resnet18":
			pass
		elif m == "resnet34":
			pass
		elif m == "segm_imagenet_resnet":
			pass
		elif m == "segm_resnet18":
			pass
		elif m == "segm_small_resnet":
			pass
		elif m == "segm_VGG":
			pass
	else:
		if m == 'vgg19':
			return ["35"]


#MODELS 
"""
For all/some models, the common arguments are:
	Ni: Input image size [Ni, Ni, Nic]
	Nic: # of channels in the input image
	Nc: # of output classes
	expansion: for wide/deep models, teh constant to increase number of filters/layers 
NOTE: for all models, Nf:= # of output features of first layer, and Ls := # of units in (size of) the Linear layer
"""

def linear_2D(Ni, Nh, Nc):
	model = nn.Sequential(
				  nn.Linear(Ni, Nh),
				  nn.ReLU(),
				  nn.Linear(Nh, Nh),
				  nn.ReLU(),
				  nn.Linear(Nh, Nh),
				  nn.ReLU(),
				  nn.Linear(Nh, Nc)
			)
	return model

def linear_mnist_model(): 
    bias = False
    model = nn.Sequential(
        Flatten(),
        nn.Linear(28*28, 500, bias=bias),
        nn.ReLU(),
        nn.Linear(500, 250, bias=bias),
        nn.ReLU(),
        nn.Linear(250, 100, bias=bias),
        nn.ReLU(),
        nn.Linear(100, 10, bias=bias)
    )
    return model

def conv_2layer_model(Ni, Nic, Nc): 
	print('conv_2layer_model: ', Ni, Nic, Nc)
	Nf, Ls = 16, 100
	bias = False
	model = nn.Sequential(
				nn.Conv2d(Nic, Nf, 4, stride=2, padding=1),
				nn.ReLU(),
				nn.Conv2d(Nf, 2*Nf, 4, stride=2, padding=1),
				nn.ReLU(),
				Flatten(),
				nn.Linear((2*Nf)*int(Ni/4)*int(Ni/4), Ls),
				nn.ReLU(),
				nn.Linear(Ls, Nc)
			)
	return model

def conv_2layer_gradcam_model(Ni, Nic, Nc): 
	print('conv_2layer_gradcam_model: ', Ni, Nic, Nc)
	Nf, Ls = 16, 100
	model = nn.Sequential(
				nn.Conv2d(Nic, Nf, 4, stride=2, padding=1),
				nn.ReLU(),
				nn.Conv2d(Nf, 2*Nf, 4, stride=2, padding=1),
				nn.ReLU(),
				Flatten(),
				nn.Linear((2*Nf)*int(Ni/4)*int(Ni/4), Nc)
			)
	return model

def conv_4layer_model(Ni, Nic, Nc): 
	print('conv_4layer_model: ', Ni, Nic, Nc)
	Nf, Ls = 32, 512
	model = nn.Sequential(
				nn.Conv2d(Nic, Nf, 3, stride=1, padding=1),
				nn.ReLU(),
				nn.Conv2d(Nf, Nf, 4, stride=2, padding=1),
				nn.ReLU(),
				nn.Conv2d(Nf, 2*Nf, 3, stride=1, padding=1),
				nn.ReLU(),
				nn.Conv2d(2*Nf, 2*Nf, 4, stride=2, padding=1),
				nn.ReLU(),
				Flatten(),
				nn.Linear((2*Nf)*int(Ni/4)*int(Ni/4), Ls),
				nn.ReLU(),
				nn.Linear(Ls, Ls),
				nn.ReLU(),
				nn.Linear(Ls, Nc)
		)
	#return model
	for m in model.modules():
		if isinstance(m, nn.Conv2d):
			n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
			m.weight.data.normal_(0, math.sqrt(2. / n))
			m.bias.data.zero_()
	return model

def conv_4layer_gradcam_model(Ni, Nic, Nc): 
	#params - 1.19M
	print('conv_4layer_gradcam_model: ', Ni, Nic, Nc)
	Nf, Ls = 32, 512
	model = nn.Sequential(
				nn.Conv2d(Nic, Nf, 3, stride=1, padding=1),
				nn.ReLU(),
				nn.Conv2d(Nf, Nf, 4, stride=2, padding=1),
				nn.ReLU(),
				nn.Conv2d(Nf, 2*Nf, 3, stride=1, padding=1),
				nn.ReLU(),
				nn.Conv2d(2*Nf, 2*Nf, 4, stride=2, padding=1),
				nn.ReLU(),
				Flatten(),
				nn.Linear((2*Nf)*int(Ni/4)*int(Ni/4), Nc)
		)
	#return model
	for m in model.modules():
		if isinstance(m, nn.Conv2d):
			n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
			m.weight.data.normal_(0, math.sqrt(2. / n))
			m.bias.data.zero_()
	return model

def conv_deep_model(Ni, Nic, Nc, expansion=1):
	def group(inf, outf, N): 
		if N == 1: 
			conv = [nn.Conv2d(inf, outf, 4, stride=2, padding=1), nn.ReLU()]
		else: 
			conv = [nn.Conv2d(inf, outf, 3, stride=1, padding=1), nn.ReLU()]
			for _ in range(1,N-1):
				conv.append(nn.Conv2d(outf, outf, 3, stride=1, padding=1))
				conv.append(nn.ReLU())
			conv.append(nn.Conv2d(outf, outf, 4, stride=2, padding=1))
			conv.append(nn.ReLU())
		return conv
	Nf, Ls = 8, 100
	conv1 = group(Nic, Nf, expansion)
	conv2 = group(Nf, 2*Nf, expansion)
	model = nn.Sequential(
						*conv1, *conv2, 
						Flatten(),
						nn.Linear((2*Nf)*int(Ni/4)*int(Ni/4), Ls),
						nn.ReLU(),
						nn.Linear(Ls, Nc)
				)
	return model
		
def conv_wide_model(Ni, Nic, Nc, expansion=1):
	Nf, Ls = 4, 128
	model = nn.Sequential(
				nn.Conv2d(Nic, expansion*Nf, 4, stride=2, padding=1),
				nn.ReLU(),
				nn.Conv2d(expansion*Nf, expansion*(2*Nf), 4, stride=2, padding=1),
				nn.ReLU(),
				Flatten(),
				nn.Linear(expansion*(2*Nf)*int(Ni/4)*int(Ni/4), expansion*Ls),
				nn.ReLU(),
				nn.Linear(expansion*Ls, Nc)
			)
	return model
   
#resnet
def conv3x3(in_planes, out_planes, stride=1):
	"""3x3 convolution with padding"""
	return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class BasicBlock(nn.Module):
	expansion = 1
	def __init__(self, inplanes, planes, stride=1, downsample=None):
		super(BasicBlock, self).__init__()
		self.conv1 = conv3x3(inplanes, planes, stride)
		self.bn1 = nn.BatchNorm2d(planes)
		self.relu = nn.ReLU(inplace=True)
		self.conv2 = conv3x3(planes, planes)
		self.bn2 = nn.BatchNorm2d(planes)
		self.downsample = downsample
		self.stride = stride

	def forward(self, x):
		residual = x
		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)
		out = self.conv2(out)
		out = self.bn2(out)
		if self.downsample is not None:
			residual = self.downsample(x)
		out += residual
		out = self.relu(out)
		return out

def cifar_model_resnet(N = 1, factor=1): 
    def  block(in_filters, out_filters, k, downsample): 
        if not downsample: 
            k_first = 3
            skip_stride = 1
            k_skip = 1
        else: 
            k_first = 4
            skip_stride = 2
            k_skip = 2
        return [
            Dense(nn.Conv2d(in_filters, out_filters, k_first, stride=skip_stride, padding=1)), 
            nn.ReLU(), 
            Dense(nn.Conv2d(in_filters, out_filters, k_skip, stride=skip_stride, padding=0), 
                  None, 
                  nn.Conv2d(out_filters, out_filters, k, stride=1, padding=1)), 
            nn.ReLU()
        ]
    conv1 = [nn.Conv2d(3,16,3,stride=1,padding=1), nn.ReLU()]
    conv2 = block(16,16*factor,3, False)
    for _ in range(N): 
        conv2.extend(block(16*factor,16*factor,3, False))
    conv3 = block(16*factor,32*factor,3, True)
    for _ in range(N-1): 
        conv3.extend(block(32*factor,32*factor,3, False))
    conv4 = block(32*factor,64*factor,3, True)
    for _ in range(N-1): 
        conv4.extend(block(64*factor,64*factor,3, False))
    layers = (
        conv1 + 
        conv2 + 
        conv3 + 
        conv4 +
        [Flatten(),
        nn.Linear(64*factor*8*8,1000), 
        nn.ReLU(), 
        nn.Linear(1000, 10)]
        )
    model = DS(*layers)
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
            if m.bias is not None: 
                m.bias.data.zero_()
    return model


def imagenet_model_resnet(Nic=3, layers=[0,0,0,0,0], num_classes=8, segm=False):
	def  block(in_filters, out_filters, k, downsample): 
		if not downsample: 
			k_first = 3
			skip_stride = 1
			k_skip = 1
		else: 
			k_first = 4
			skip_stride = 2
			k_skip = 2
		return [
		    Dense(nn.Conv2d(in_filters, out_filters, k_first, stride=skip_stride, padding=1)), 
		    nn.ReLU(), 
		    Dense(nn.Conv2d(in_filters, out_filters, k_skip, stride=skip_stride, padding=0), 
		          None, 
		          nn.Conv2d(out_filters, out_filters, k, stride=1, padding=1)), 
		    nn.ReLU()
		]
	conv0 = [nn.Conv2d(Nic, int(64*f), kernel_size=8, stride=2, padding=3, bias=False), nn.ReLU()]
	if not SMALL:
		layer1 = block(int(64*f), int(64*f), 3, False)
		for _ in range(layers[0]): 
			layer1.extend(block(int(64*f), int(64*f), 3, False))
	layer2 = block(int(64*f), int(128*f), 3, True)
	for _ in range(layers[1]): 
		layer2.extend(block(int(128*f), int(128*f), 3, False))
	layer3 = block(int(128*f), int(256*f), 3, True)
	for _ in range(layers[2]): 
		layer3.extend(block(int(256*f), int(256*f), 3, False))
	layer4 = block(int(256*f), int(512*f), 3, True)
	for _ in range(layers[3]): 
		layer4.extend(block(int(512*f), int(512*f), 3, False))
	if not SMALL:
		layers = (conv0 + layer1 + layer2 + layer3 + layer4 + 
				[Flatten(), nn.Linear(int(131072*4*f*f*f), 100), 
        		nn.ReLU(), 
        		nn.Linear(100, 10)]
			)
	else:
		layers = (conv0 + layer2 + layer3 + layer4 + 
				[Flatten(), nn.Linear(int(131072*4*f*f*f), 100), 
        		nn.ReLU(), 
        		nn.Linear(100, 10)]
			)
	model = DenseSequential(*layers)
	for m in model.modules():
		if isinstance(m, nn.Conv2d):
			n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
			m.weight.data.normal_(0, math.sqrt(2. / n))
			if m.bias is not None: 
				m.bias.data.zero_()
	return model
	
class DenseSequential(nn.Sequential): 
    def forward(self, x):
        global output
        #block = {5:'x1', 9:'x2', 13:'x3', 17:'x4', 21:'x5'}
        if SMALL:
	        block = {1:'x1', 5:'x2', 9:'x3', 13:'x4'}
        else:
            block = {5:'x1', 9:'x2', 13:'x3', 17:'x4'}

        xs = [x]
        print('MINE')
        for i, module in enumerate(self._modules.values()):
            if 'Dense' in type(module).__name__:
                xs_prop = module(*xs) 
                xs.append(xs_prop)
            else:
                xs_prop = module(xs[-1]) 
                xs.append(xs_prop)
            if i in block.keys():
                output[block[i]] = xs_prop
            print(i, xs_prop.shape)
        return xs[-1]


class ResNet(nn.Module):
	def __init__(self, Ni, Nic, layers, block=BasicBlock, num_classes=10, segm=False):
		print('ResNet_model: ', Ni, Nic, num_classes)
		self.segm = segm
		self.inplanes = 64
		super(ResNet, self).__init__()
		self.conv1 = nn.Conv2d(Nic, 64, kernel_size=7, stride=1, padding=3, bias=False)
		self.bn1 = nn.BatchNorm2d(64)
		self.relu = nn.ReLU(inplace=True)
		self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
		self.layer1 = self._make_layer(block, 64, layers[0])
		self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
		self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
		self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
		self.layer5 = self._make_layer(block, 512, layers[4], stride=2)
		self.avgpool = nn.AvgPool2d(7, stride=1)
		self.fc = nn.Linear(512 * 100 * block.expansion, num_classes)
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
				m.weight.data.normal_(0, math.sqrt(2. / n))
			elif isinstance(m, nn.BatchNorm2d):
				m.weight.data.fill_(1)
				m.bias.data.zero_()

	def _make_layer(self, block, planes, blocks, stride=1):
		downsample = None
		if stride != 1 or self.inplanes != planes * block.expansion:
			downsample = nn.Sequential(
					nn.Conv2d(self.inplanes, planes * block.expansion,kernel_size=1, stride=stride, bias=False),
					nn.BatchNorm2d(planes * block.expansion),
			)
		layers = []
		layers.append(block(self.inplanes, planes, stride, downsample))
		self.inplanes = planes * block.expansion
		for i in range(1, blocks):
			layers.append(block(self.inplanes, planes))
		return nn.Sequential(*layers)

	def forward(self, x):
		output = {}
		x = self.conv1(x)
		if self.segm:
			x = self.bn1(x)
		x = self.relu(x)
		if self.segm:
			x = self.maxpool(x)
		x = self.layer1(x)
		output['x1'] = x
		x = self.layer2(x)
		output['x2'] = x
		x = self.layer3(x)
		output['x3'] = x
		x = self.layer4(x)
		output['x4'] = x
		x = self.layer5(x)
		output['x5'] = x
		if self.segm:
			x = self.avgpool(x)
		x = x.view(x.size(0), -1)
		x = self.fc(x)
		if self.segm:
			return x, output
		else:
			return x
		
def resnet18(Ni, Nic, Nc, segm=False, pretrained=False):
	model = ResNet(Ni, Nic, layers = [2,2,2,2,2], num_classes = Nc, segm = segm) #[2,2,2,2]
	return model

def resnet34(Ni, Nic, Nc, segm=False, pretrained=False):
	model = ResNet(Ni, Nic, layers = [2,2,2,2,2], num_classes = Nc, segm = segm) #[3,4,6,3]
	return model
  
    
#model utils
class Flatten(nn.Module):
	def forward(self, x):
		return x.view(x.size(0), -1)
		
class SavePattern(nn.Module):
	def forward(self, x, index):
		xx = x.clone().detach().cpu()
		xx = torch.clamp(xx, min=0)
		xx = (xx>0)
		sns.heatmap(xx.numpy(), linewidth=0.5)                
		plt.title('layer shape: '+str(xx.shape) + ' ' + '   # of 0 spanning activations: '+str(torch.nonzero(xx).size(0)))
		plt.pause(0.5)
		plt.savefig('./clean_activations_'+str(index)+'.png', dpi=300, bbox_inches='tight')
		return x
		
##SEGMENTATION MODELS
#the encoder part
class VGGNet(VGG):
	def __init__(self, pretrained=False, model='vgg11', requires_grad=True, remove_fc=True, show_params=False):
		super().__init__(make_layers(cfg[model]))
		self.ranges = ranges_vgg[model]
		if pretrained:
			exec("self.load_state_dict(models.%s(pretrained=True).state_dict())" % model)
		if not requires_grad:
			for param in super().parameters():
				param.requires_grad = False
		if remove_fc:  # delete redundant fully-connected layer params, can save memory
			del self.classifier
		if show_params:
			for name, param in self.named_parameters():
				print(name, param.size())

	def forward(self, x):
		output = {}
		# get the output of each maxpooling layer (5 maxpool in VGG net)
		for idx in range(len(self.ranges)):
			for layer in range(self.ranges[idx][0], self.ranges[idx][1]):
				x = self.features[layer](x)
			output["x%d"%(idx+1)] = x
		return x, output

ranges_vgg = {
    'vgg11': ((0, 3), (3, 6),  (6, 11),  (11, 16), (16, 21)),
    'vgg13': ((0, 5), (5, 10), (10, 15), (15, 20), (20, 25)),
    'vgg16': ((0, 5), (5, 10), (10, 17), (17, 24), (24, 31)),
    'vgg19': ((0, 5), (5, 10), (10, 19), (19, 28), (28, 37))
}
# cropped version from https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
cfg = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

def make_layers(cfg, batch_norm=False):
	layers = []
	in_channels = 3
	for v in cfg:
		if v == 'M':
			layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
		else:
			conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
			if batch_norm:
				layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
			else:
				layers += [conv2d, nn.ReLU(inplace=True)]
			in_channels = v
	return nn.Sequential(*layers)

#the decoder part of the segmentation network
class FCNs(nn.Module):
	def __init__(self, pretrained_net, n_class, output=True):
		super().__init__()
		self.n_class = n_class
		self.output_present = output
		self.pretrained_net = pretrained_net
		self.relu    = nn.ReLU(inplace=True)
		self.deconv1 = nn.ConvTranspose2d(int(512*f), int(512*f), kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
		self.bn1     = nn.BatchNorm2d(int(512*f))
		self.deconv2 = nn.ConvTranspose2d(int(512*f), int(256*f), kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
		self.bn2     = nn.BatchNorm2d(int(256*f))
		self.deconv3 = nn.ConvTranspose2d(int(256*f), int(128*f), kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
		self.bn3     = nn.BatchNorm2d(int(128*f))
		self.deconv4 = nn.ConvTranspose2d(int(128*f), int(64*f), kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
		self.bn4     = nn.BatchNorm2d(int(64*f))
		self.deconv5 = nn.ConvTranspose2d(int(64*f), int(32*f), kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
		self.bn5     = nn.BatchNorm2d(int(32*f))
		self.classifier = nn.Conv2d(int(32*f), n_class, kernel_size=1)

	def forward(self, x):
		global output
		if self.output_present:
			_, output = self.pretrained_net(x)
		else:
			_ = self.pretrained_net(x)
		x5 = output['x5']  # size=(N, 512, x.H/32, x.W/32) 16
		x4 = output['x4']  # size=(N, 512, x.H/16, x.W/16) 32
		x3 = output['x3']  # size=(N, 256, x.H/8,  x.W/8) 64
		x2 = output['x2']  # size=(N, 128, x.H/4,  x.W/4) 128
		x1 = output['x1']  # size=(N, 64, x.H/2,  x.W/2) 256
		#print(x4.shape, x3.shape, x2.shape, x1.shape)

		score = self.bn1(self.relu(self.deconv1(x5)))     # size=(N, 512, x.H/16, x.W/16)
		score = score + x4                                # element-wise add, size=(N, 512, x.H/16, x.W/16)
		score = self.bn2(self.relu(self.deconv2(score)))  # size=(N, 256, x.H/8, x.W/8)
		score = score + x3                                # element-wise add, size=(N, 256, x.H/8, x.W/8)
		score = self.bn3(self.relu(self.deconv3(score)))  # size=(N, 128, x.H/4, x.W/4)
		score = score + x2                                # element-wise add, size=(N, 128, x.H/4, x.W/4)
		score = self.bn4(self.relu(self.deconv4(score)))  # size=(N, 64, x.H/2, x.W/2)
		score = score + x1                                # element-wise add, size=(N, 64, x.H/2, x.W/2)
		score = self.bn5(self.relu(self.deconv5(score)))  # size=(N, 32, x.H, x.W)
		score = self.classifier(score)                    # size=(N, n_class, x.H/1, x.W/1)
		return score  # size=(N, n_class, x.H/1, x.W/1)



#SEGMATION experiments
class ResNetSmall(nn.Module):
	def __init__(self, Ni, Nic, layers, block=BasicBlock, num_classes=10, segm=False):
		print('num_classes and Image size: ', num_classes, Ni) #64
		self.segm = segm
		self.inplanes = 8
		super(ResNetSmall, self).__init__()
		self.conv1 = nn.Conv2d(Nic, self.inplanes, kernel_size=7, stride=1, padding=3, bias=False)
		self.bn1 = nn.BatchNorm2d(self.inplanes)
		self.relu = nn.ReLU(inplace=True)
		self.maxpool = nn.MaxPool2d(kernel_size=3, stride=4, padding=1) #Ni/4,Ni/4,inplanes
		self.layer1 = self._make_layer(block, self.inplanes, layers[0]) ##Ni/4,Ni/4,inplanes
		self.layer2 = self._make_layer(block, 2*self.inplanes, layers[1], stride=2) #Ni/8,Ni/8,2*inplanes
		self.layer3 = self._make_layer(block, 4*self.inplanes, layers[2], stride=2) #Ni/16,Ni/16,4*inplanes
		self.fc = nn.Linear(4*self.inplanes * 4 * block.expansion, num_classes)
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
				m.weight.data.normal_(0, math.sqrt(2. / n))
			elif isinstance(m, nn.BatchNorm2d):
				m.weight.data.fill_(1)
				m.bias.data.zero_()

	def _make_layer(self, block, planes, blocks, stride=1):
		downsample = None
		if stride != 1 or self.inplanes != planes * block.expansion:
			downsample = nn.Sequential(
					nn.Conv2d(self.inplanes, planes * block.expansion,kernel_size=1, stride=stride, bias=False),
					nn.BatchNorm2d(planes * block.expansion),
			)
		layers = []
		layers.append(block(self.inplanes, planes, stride, downsample))
		self.inplanes = planes * block.expansion
		for i in range(1, blocks):
			layers.append(block(self.inplanes, planes))
		return nn.Sequential(*layers)

	def forward(self, x):
		output = {}
		x = self.conv1(x)
		if self.segm:
			x = self.bn1(x)
		x = self.relu(x)
		if self.segm:
			x = self.maxpool(x)
		x = self.layer1(x)
		output['x1'] = x
		x = self.layer2(x)
		output['x2'] = x
		x = self.layer3(x)
		output['x3'] = x
		x = x.view(x.size(0), -1)
		x = self.fc(x)
		if self.segm:
			return x, output
		else:
			return x
		
def resnet18_segm(Ni, Nic, Nc, segm=False, pretrained=False):
	model = ResNetSmall(Ni, Nic, layers = [2,2,2,2,2], num_classes = Nc, segm = segm) #[2,2,2,2]
	return model


class FCN_SMALL(nn.Module):
	def __init__(self, pretrained_net, n_class, output=True):
		super().__init__()
		self.n_class = n_class
		self.output_present = output
		self.pretrained_net = pretrained_net
		self.relu    = nn.ReLU(inplace=True)
		self.deconv1 = nn.ConvTranspose2d(4, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
		self.bn1     = nn.BatchNorm2d(32)
		self.deconv2 = nn.ConvTranspose2d(8, 16, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
		self.bn2     = nn.BatchNorm2d(16)
		self.deconv3 = nn.ConvTranspose2d(16, 8, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
		self.bn3     = nn.BatchNorm2d(8)
		self.classifier = nn.Conv2d(8, n_class, kernel_size=1)

	def forward(self, x):
		global output
		if self.output_present:
			_, output = self.pretrained_net(x)
		else:
			_ = self.pretrained_net(x)
		x3 = output['x3']  # size=(N, 32, x.H/16,  x.W/16) 
		x2 = output['x2']  # size=(N, 16, x.H/8,  x.W/8) 
		x1 = output['x1']  # size=(N, 8, x.H/4,  x.W/4) 
		print(x3.shape, x2.shape, x1.shape)

		score = self.bn1(self.relu(self.deconv1(x3)))     # size=(N, 512, x.H/16, x.W/16)
		score = score + x2                                # element-wise add, size=(N, 512, x.H/16, x.W/16)
		score = self.bn2(self.relu(self.deconv2(score)))  # size=(N, 256, x.H/8, x.W/8)
		score = score + x1                                # element-wise add, size=(N, 256, x.H/8, x.W/8)
		score = self.bn3(self.relu(self.deconv3(score)))  # size=(N, 128, x.H/4, x.W/4)
		score = self.classifier(score)                    # size=(N, n_class, x.H/1, x.W/1)
		return score  # size=(N, n_class, x.H/1, x.W/1)