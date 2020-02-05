""" Contains functions for loading (+augmenting) various datasets being used. """
import os
import datetime
import numpy as np
import sys
import pickle
import glob
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm

#pytorch imports
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms

def load_dataset(d = None, batch_size = 50, noise = 0.001, dim = 2, num_channels = 1, train_pts = 128, test_pts = 16, classes = 2, segm=False, dataset_file=None, grad_cam=False, test_batch_size=None):
	""" Calls a specific dataset loading method.
	
	#Arguments:
	-d: dataset on which m is to be trained/evaluated (it can be name in case of standard datasets or the name of the folder to load images from)
	-batch_size: (for both train and test loading)
	
	#Returns:
	-train_loader, test_loader
	"""
	if d == 'mnist':
		train_loader, test_loader = mnist_loaders(batch_size, noise, test_batch_size=test_batch_size)
	elif d == 'fmnist':
		train_loader, test_loader = fashion_mnist_loaders(batch_size)	
	elif d == 'cifar':
		train_loader, test_loader = cifar_loaders(batch_size, noise, test_batch_size=test_batch_size)
	elif d == 'imagenet':
		train_loader, test_loader = imagenet_loaders(batch_size, dim)
	elif (d == 'syn' or os.path.isfile(d) or dataset_file!=None):
		if dataset_file!=None:
			train_loader, test_loader, _ = syn_data(batch_size, dim, train_pts, test_pts, classes, dataset_file)
		else:
			train_loader, test_loader, _ = syn_data(batch_size, dim, train_pts, test_pts, classes, d)
	elif os.path.isdir(d):
		train_loader, test_loader = custom_loader(batch_size, d, dim, num_channels, segm, grad_cam, test_batch_size=test_batch_size)
	return train_loader, test_loader

#LOADERS
def mnist_loaders(batch_size, noise, shuffle_test=True, test_batch_size=None):
	if test_batch_size==None:
		test_batch_size = batch_size 
	print("test_batch_size : ", test_batch_size)
	mnist_train = datasets.MNIST("./data/mnist", train=True, download=True, 
		transform=transforms.Compose([
			#transforms.RandomAffine(2, translate=(0.02,0.02), scale=(0.97,1.03)),
			transforms.ToTensor(),
			#transforms.Lambda(lambda x : x + noise*torch.randn(*x.size())),
			]))
	mnist_test = datasets.MNIST("./data/mnist", train=False, download=True, transform=transforms.ToTensor())
	train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=False, pin_memory=True)
	test_loader = torch.utils.data.DataLoader(mnist_test, batch_size=test_batch_size, shuffle=shuffle_test, pin_memory=True)
	return train_loader, test_loader

def fashion_mnist_loaders(batch_size): 
	mnist_train = datasets.MNIST("./data/fashion_mnist", train=True, download=True, transform=transforms.ToTensor())
	mnist_test = datasets.MNIST("./data/fashion_mnist", train=False, download=True, transform=transforms.ToTensor())
	train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, pin_memory=True)
	test_loader = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False, pin_memory=True)
	return train_loader, test_loader

def imagenet_loaders(batch_size, N): 
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    imagenet_train = datasets.ImageFolder("/home/hh/masters_thesis_notebooks/master_repo/data/imagenet/train",
        			transform = transforms.Compose([
			            transforms.Resize(N),
			            transforms.CenterCrop(N),
			            transforms.ToTensor(),
			            #normalize,
			        ]))
    imagenet_test = datasets.ImageFolder("/home/hh/masters_thesis_notebooks/master_repo/data/imagenet/test",
    				transform = transforms.Compose([
			            transforms.Resize(N),
			            transforms.CenterCrop(N),
			            transforms.ToTensor(),
			            #normalize,
			        ]))
    train_loader = torch.utils.data.DataLoader(imagenet_train, batch_size=batch_size, shuffle=True, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(imagenet_test, batch_size=batch_size, shuffle=False, pin_memory=True)
    return train_loader, test_loader

def cifar_loaders(batch_size, noise, shuffle_test=True, test_batch_size=None): 
	if test_batch_size==None:
		test_batch_size = batch_size 
	normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.225, 0.225, 0.225])
	train = datasets.CIFAR10("./data/cifar", train=True, download=True, 
        transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            #transforms.RandomAffine(10, scale=(1,1.05)),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            #transforms.Lambda(lambda x : x + noise*torch.randn(*x.size())),
            normalize,
        ]))
	test = datasets.CIFAR10("./data/cifar", train=False, transform=transforms.Compose([transforms.ToTensor(), normalize]))
	train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=False, pin_memory=True)
	test_loader = torch.utils.data.DataLoader(test, batch_size=test_batch_size,shuffle=shuffle_test, pin_memory=True)
	return train_loader, test_loader
    
class ImageFolderWithPaths(datasets.ImageFolder):
	#Custom dataset (extends torchvision.datasets.ImageFolder) that includes image file paths.
	def __getitem__(self, index):
		original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
		path = self.imgs[index][0]
		tuple_with_path = (original_tuple + (path,))
		return tuple_with_path

def custom_loader(batch_size, data_dir, dim, num_channels, segm=False, grad_cam=False, test_batch_size=None):
	if test_batch_size==None:
		test_batch_size = batch_size 
	#data_dir
	# --train
	# --test	
	if data_dir is None:
		raise ValueError('the output dir need to be specified.')
	if not os.path.exists(os.path.dirname(data_dir)):
		os.makedirs(os.path.dirname(data_dir))

	normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
     	                                std=[0.229, 0.224, 0.225])
	if (not segm) and (not grad_cam):
		data_transforms = {
			'train': transforms.Compose([
								transforms.Resize(dim),
								transforms.CenterCrop(dim),
								#transforms.RandomHorizontalFlip(),
								transforms.ToTensor(),
								normalize
								]),
			'test': transforms.Compose([
								transforms.Resize(dim),
								transforms.CenterCrop(dim),
								transforms.ToTensor(),
								normalize
								]),
		}
		if num_channels==1 and True:
			image_datasets = {x: ImageFolderWithPaths(os.path.join(data_dir,x), transforms.Compose([transforms.Grayscale(num_output_channels=1), transforms.ToTensor()])) for x in ['train', 'test']}
		else:
			image_datasets = {x: ImageFolderWithPaths(os.path.join(data_dir,x), transform=data_transforms[x]) for x in ['train', 'test']} #transforms.Compose([transforms.ToTensor(), normalize])
		dataloaders = {}
		dataloaders['train'] = torch.utils.data.DataLoader(image_datasets['train'], batch_size=batch_size, shuffle=True, num_workers=4)
		dataloaders['test'] = torch.utils.data.DataLoader(image_datasets['test'], batch_size=test_batch_size, shuffle=True, num_workers=4)
		return dataloaders['train'], dataloaders['test']
	elif segm:
		image_transforms = {
			'train': transforms.Compose([
								transforms.Resize(dim),
								transforms.CenterCrop(dim),
								transforms.ToTensor(),
								]),
			'test': transforms.Compose([
								transforms.Resize(dim),
								transforms.CenterCrop(dim),
								transforms.ToTensor(),
								]),
		}
		label_transforms = {
			'train': transforms.Compose([
								transforms.Resize(dim, interpolation=Image.NEAREST),
								transforms.CenterCrop(dim),
								transforms.ToTensor(),
								]),
			'test': transforms.Compose([
								transforms.Resize(dim, interpolation=Image.NEAREST),
								transforms.CenterCrop(dim),
								transforms.ToTensor(),
								]),
		}
		train_image_path = os.path.join(data_dir,'train','image')
		train_label_path = os.path.join(data_dir,'train','label')
		test_image_path = os.path.join(data_dir,'test','image')
		test_label_path = os.path.join(data_dir,'test','label')
		train_images = [image_transforms['train'](Image.open(f).convert("RGB")) for f in sorted(glob.glob(train_image_path + "/*.png"))]
		train_labels = [label_transforms['train'](Image.open(f).convert("RGB")) for f in sorted(glob.glob(train_label_path + "/*.png"))]
		test_images = [image_transforms['test'](Image.open(f).convert("RGB")) for f in sorted(glob.glob(test_image_path + "/*.png"))]
		test_labels = [label_transforms['test'](Image.open(f).convert("RGB")) for f in sorted(glob.glob(test_label_path + "/*.png"))]	
		return [train_images, train_labels], [test_images, test_labels]
	elif grad_cam:
		train_image_path = os.path.join(data_dir,'train','image')
		train_label_path = os.path.join(data_dir,'train','label')
		test_image_path = os.path.join(data_dir,'test','image')
		test_label_path = os.path.join(data_dir,'test','label')
		
		image_transforms = transforms.Compose([
								transforms.Resize(dim),
								transforms.CenterCrop(dim),
								transforms.ToTensor(),
								normalize,
								])
		label_transforms = transforms.Compose([
								transforms.Resize(dim, interpolation=Image.NEAREST),
								transforms.CenterCrop(dim),
								transforms.Grayscale(num_output_channels=1),
								transforms.ToTensor(),
								])
		label_datasets, image_datasets = {}, {}
		label_datasets['train'] = ImageFolderWithPaths(train_label_path, label_transforms)
		label_datasets['test'] = ImageFolderWithPaths(test_label_path, label_transforms)
		image_datasets['train'] = ImageFolderWithPaths(train_image_path, image_transforms)
		image_datasets['test'] = ImageFolderWithPaths(test_image_path, image_transforms)

		imageloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4) for x in ['train', 'test']}
		labelloaders = {x: torch.utils.data.DataLoader(label_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4) for x in ['train', 'test']}
		return [imageloaders['train'], labelloaders['train']], [imageloaders['test'], labelloaders['test']] 

def syn_data(batch_size, dim, train_pts, test_pts, classes, data_file):
	if os.path.exists(data_file):
		with open(data_file, 'rb') as f:
			x = pickle.load(f)
		train_dataset = torch.utils.data.TensorDataset(x['train_data'], x['train_label'])
		test_dataset = torch.utils.data.TensorDataset(x['test_data'], x['test_label'])
		return torch.utils.data.DataLoader(train_dataset, batch_size = batch_size), torch.utils.data.DataLoader(test_dataset, batch_size = batch_size), x['base_points']	
	else:
		np.random.seed(0)
		data_file = os.getcwd() + '/outputs/data/' + datetime.datetime.now().strftime("%m_%d_%H%M")
	
	# construct base points
	#if random dataset is needed use this
	base_pts = [np.random.uniform(low = -1., high = 1., size = [dim,]) for _ in range(classes)]
	#if you want correlated dataset, change logic below and uncomment
	#base_pts = [[0.35,-0.35,],[-0.35,0.35]] #[[0.83138438, -0.48],[-0.48, 0.83138438]] #
	base_pts = np.array(base_pts)
	print('Base points constructed!')
	
	# construct data points
	data_set = []
	label_set = []
	pts = train_pts + test_pts
	i=0
	for idx in range(pts):
		sys.stdout.write('%d / %d loaded\r' % (idx + 1, pts))
		#if random dataset is needed, use this
		data_pt = np.random.uniform(low = -1., high = 1., size = [dim,])
		#if you want correlated dataset, change logic below and uncomment
		'''
		data_pt = [0,0]
		data_pt[0] = np.random.uniform(low = base_pts[i][0]-0.3, high = base_pts[i][0]+0.3)
		data_pt[1] = np.random.uniform(low = base_pts[i][1]-0.3, high = base_pts[i][1]+0.3)
		'''
		dist_list = [(idx, np.linalg.norm(base_pt - data_pt) ** 2) for idx, base_pt in enumerate(base_pts)]
		dist_list = sorted(dist_list, key = lambda x: x[1])		

		data_set.append(data_pt)
		label_set.append(dist_list[0][0])
		if idx>pts/2:
			i=1
	train_data_set = torch.from_numpy(np.array(data_set[:train_pts])).float()
	train_label_set = torch.from_numpy(np.array(label_set[:train_pts], dtype = int))
	test_data_set = torch.from_numpy(np.array(data_set[train_pts:])).float()
	test_label_set = torch.from_numpy(np.array(label_set[train_pts:], dtype = int))
	print('Data points constructed!')
	
	colors = cm.rainbow(np.linspace(0, 1, classes))
	label_colors = [colors[label] for label in label_set]
	#plot and save the dataset in image so you can see what you created
	plt.scatter([x[0] for x in data_set[:train_pts]], [x[1] for x in data_set[:train_pts]], s = 3, color = label_colors[:train_pts])
	plt.scatter([x[0] for x in data_set[train_pts:]], [x[1] for x in data_set[train_pts:]], s = 10, facecolors='none', edgecolors=label_colors[train_pts:])
	plt.xticks([-1., 0., 1.])
	plt.yticks([-1., 0., 1.])
	plt.xlim(-1., 1.)
	plt.ylim(-1., 1.)
	plt.savefig(data_file+'.png', bbox = 'tight', dpi = 500)
    
	pickle.dump({'train_data': train_data_set,
							 'train_label': train_label_set,
							 'test_data': test_data_set,
							 'test_label': test_label_set,
							 'base_points': base_pts,
							 'classes': classes}, open(data_file, 'wb'))		
	print('Information dumpped in file %s' % data_file)
	train_dataset = torch.utils.data.TensorDataset(train_data_set, train_label_set)
	test_dataset = torch.utils.data.TensorDataset(test_data_set, test_label_set)
	return torch.utils.data.DataLoader(train_dataset, batch_size = batch_size), torch.utils.data.DataLoader(test_dataset, batch_size = batch_size), base_pts


#done