""" Contains functions for graphs, visualizations and saving images. """
import numpy as np
import matplotlib.pyplot as plt
import datetime
import os
from scipy.misc import imsave
from scipy.misc import imresize
from MulticoreTSNE import MulticoreTSNE as TSNE

import torch

def make_2Dplot(t, x, Y, Legends, Labels, savepath):
	#for attacks and verifier
	#x = [float(xx) for xx in x]
	plt.figure(figsize = (5,5))
	icons = ["*-", "^-", "+-", "--", ".-", "*-", "^-", "+-", "--", ".-"]
	for j in range(len(Y)):
		plt.plot(x, Y[j], icons[j], label=Legends[j])
	plt.legend()
	#plt.yticks(np.arange(0, max(max(y for y in Y)), step=np.add(max(max(y for y in Y)),0.1)/10.))
	plt.xticks(np.arange(0, max(x), step=max(x)/10.0))
	plt.title(t)
	plt.xlabel(Labels[0])
	plt.ylabel(Labels[1])
	if savepath is not None:
		print('Saving fig to {}'.format(savepath)) 
		plt.savefig(savepath + t + '.png')
	plt.show()
	
def save_image(path, image):
	plt.imsave(path, image, cmap='gray')
	
def show_results(epsilons, examples, savepath):
	cnt = 0
	plt.figure(figsize=(8,10))
	for i in range(1,len(epsilons)):
		for j in range(len(examples[i])):
			cnt += 1
			plt.subplot(len(epsilons),len(examples[1]),cnt)
			plt.xticks([], [])
			plt.yticks([], [])
			if j == 0:
				plt.ylabel("Eps: {}".format(epsilons[i]), fontsize=14)
			act,orig,adv,advn,ex = examples[i][j]
			plt.title("{}->{}->{}->{}".format(act, orig, adv, advn))
			#if ex.dim()==3:
			ex = np.transpose(ex, (1,2,0))
			plt.imshow(ex, cmap="gray")
	plt.tight_layout()
	if savepath is not None:
		print('Saving fig to {}'.format(savepath)) 
		plt.savefig(savepath + 'results_sample.png')
	plt.show()

def plot_mean_and_CI(mean, std, color_mean=None, color_shading=None, axis=None):
	lb = [m - s for m,s in zip(mean, std)]
	ub = [m + s for m,s in zip(mean, std)]
	# plot the shaded range of the confidence intervals
	axis.fill_between(range(len(mean)), ub, lb, color=color_shading, alpha=.5)
	# plot the mean on top
	axis.plot(mean, color_mean)

def convert_cifar10(img, nup=False, t=True):
    #img : [Nc, H, W]
    if nup==False:
        im = img.clone().numpy()
    else:
        im = img
    # approximate unnormalization 
    im[0,:,:] = im[0,:,:]*0.229 + 0.485
    im[1,:,:] = im[1,:,:]*0.224 + 0.456
    im[2,:,:] = im[2,:,:]*0.225 + 0.406
    if t:
        im_new = np.transpose(im,(1,2,0))
    else:
        im_new = torch.tensor(im, dtype = torch.float)
    return im_new

def denormalize_cifar10(img):
    #img : [1, Nc, H, W]
    # approximate unnormalization 
    im = img
    im[:,0,:,:] = im[:,0,:,:]*0.229 + 0.485
    im[:,1,:,:] = im[:,1,:,:]*0.224 + 0.456
    im[:,2,:,:] = im[:,2,:,:]*0.225 + 0.406
    return im

def normalize_cifar10(img):
    #img := [1, Nc, N, N]
    # approximate unnormalization 
    im = img
    im[:,0,:,:] = (im[:,0,:,:] - 0.485)/0.229
    im[:,1,:,:] = (im[:,1,:,:] - 0.456)/0.224
    im[:,2,:,:] = (im[:,2,:,:] - 0.406)/0.225
    return im

#eps_n_bound
def plot_update(plt, X, epsilon, A, color):
	#for eps_and_boundaries
	plt.xticks([-1., 0., 1.])
	plt.yticks([-1., 0., 1.])
	plt.xlim(-1., 1.)
	plt.ylim(-1., 1.)	
	plt.axes().set_aspect('equal')		
	data = {}
	data[0] = X[0][0]
	data[1] = X[0][1]
	epsilon_l = epsilon[0]
	epsilon_u = epsilon[1]
	e = np.array([[-epsilon_l[0][0],-epsilon_l[0][0]], [-epsilon_l[0][1],epsilon_u[0][1]]])*np.transpose(A)
	plt.plot([data[0]+e[0,0],data[0]+e[0,1]] , [data[1]+e[1,0],data[1]+e[1,1]], color = 'orange')
	e = np.array([[epsilon_u[0][0],epsilon_u[0][0]], [-epsilon_l[0][1],epsilon_u[0][1]]])*np.transpose(A)
	plt.plot([data[0]+e[0,0],data[0]+e[0,1]] , [data[1]+e[1,0],data[1]+e[1,1]], color = 'orange')
	e = np.array([[-epsilon_l[0][0],epsilon_u[0][0]], [-epsilon_l[0][1],-epsilon_l[0][1]]])*np.transpose(A)
	plt.plot([data[0]+e[0,0],data[0]+e[0,1]] , [data[1]+e[1,0],data[1]+e[1,1]], color = 'orange')
	e = np.array([[-epsilon_l[0][0],epsilon_u[0][0]], [epsilon_u[0][1],epsilon_u[0][1]]])*np.transpose(A)
	plt.plot([data[0]+e[0,0],data[0]+e[0,1]] , [data[1]+e[1,0],data[1]+e[1,1]], color = 'orange')		
	plt.draw()
	
def plot_clear_update(loaders):
	#for eps_and_boundaries
	plt.clf()
	plt.xticks([-1., 0., 1.])
	plt.yticks([-1., 0., 1.])
	plt.xlim(-1., 1.)
	plt.ylim(-1., 1.)
	plt.axes().set_aspect('equal')		
	eps=0.05
	colors = {0:'red', 1:'blue', 2:'pink', 3:'brown'}
	for X, y in loaders[0]:
		plt.scatter(X[0][0].numpy(), X[0][1].numpy(), s = 10, facecolors='none', edgecolors = colors[y[0].item()])
		plt.text(x = X[0][0].numpy() + 1e-2, y = X[0][1].numpy() - 6e-2, s = str(y[0].numpy()), color = 'g', fontsize = 6)
		plt.draw()
	for X, y in loaders[1]:
		plt.scatter(X[0][0].numpy(), X[0][1].numpy(), s = 3, color = colors[y[0].item()])
		plt.text(x = X[0][0].numpy() + 1e-2, y = X[0][1].numpy() - 6e-2, s = str(y[0].numpy()), color = 'g', fontsize = 6)
		plt.plot([X[0][0]-eps,X[0][0]-eps] , [X[0][1]-eps,X[0][1]+eps], color = 'silver')
		plt.plot([X[0][0]+eps,X[0][0]+eps] , [X[0][1]-eps,X[0][1]+eps], color = 'silver')
		plt.plot([X[0][0]-eps,X[0][0]+eps] , [X[0][1]-eps,X[0][1]-eps], color = 'silver')
		plt.plot([X[0][0]-eps,X[0][0]+eps] , [X[0][1]+eps,X[0][1]+eps], color = 'silver')
		plt.draw()
	plt.pause(0.01)

#train_main
def plot_bounds(listb, path, name):
	#for trainer
	plt.figure()
	if len(listb) > 1: 
		fig, axes = plt.subplots(1, len(listb), figsize=(100,40))
		for i,l in enumerate(listb): #5
			plot_mean_and_CI([ll[0] for ll in l], [ll[1] for ll in l], 'g', 'g', axes[i])
			plot_mean_and_CI([ll[2] for ll in l], [ll[3] for ll in l], 'r', 'r', axes[i])
			axes[i].plot([ll[4] for ll in l], 'k--')
			axes[i].plot([ll[5] for ll in l], 'b--')
	else:
		plot_mean_and_CI([ll[0] for ll in listb[0]], [ll[1] for ll in listb[0]], 'g', 'g', plt)
		plot_mean_and_CI([ll[2] for ll in listb[0]], [ll[3] for ll in listb[0]], 'r', 'r', plt)
		plt.plot([ll[4] for ll in listb[0]], 'k--')
		plt.plot([ll[5] for ll in listb[0]], 'b--')
	plt.savefig(path + name + '_bounds.png')

def plot_params(methods, params_epoch, params, t, log_dir_path, name, plot_ind=8):
	#for trainer
	for i in range(len(methods)):
		if methods[i] == 'actual':
			params_epoch['actual'].append([np.mean(params['corr_lbound_baseline']), np.std(params['corr_lbound_baseline']),np.mean(params['wrong_ubound_baseline']), np.std(params['wrong_ubound_baseline']), np.mean(params['corr_lbound_baseline']), np.mean(params['wrong_ubound_baseline'])])
		elif methods[i] == 'wong':
			params_epoch['wong'].append([np.mean(params['corr_lbound_wong']), np.std(params['corr_lbound_wong']), np.mean(params['wrong_ubound_wong']), np.std(params['wrong_ubound_wong']), np.mean(params['corr_wong']), np.mean(params['wrong_wong'])])
		elif methods[i] == 'wongIBP':
			params_epoch['wibp'].append([np.mean(params['corr_lbound_wibp']), np.std(params['corr_lbound_wibp']), np.mean(params['wrong_ubound_wibp']), np.std(params['wrong_ubound_wibp']), np.mean(params['corr_wibp']), np.mean(params['wrong_wibp'])])
			params_epoch['wibp_q'].append([np.mean(params['corr_lbound_wibp_q']), np.std(params['corr_lbound_wibp_q']), np.mean(params['wrong_ubound_wibp_q']), np.std(params['wrong_ubound_wibp_q']), np.mean(params['corr_wibp']), np.mean(params['wrong_wibp'])])
			params_epoch['wibp_s'].append([np.mean(params['corr_lbound_wibp_s']), np.std(params['corr_lbound_wibp_s']), np.mean(params['wrong_ubound_wibp_s']), np.std(params['wrong_ubound_wibp_s']), np.mean(params['corr_wibp']), np.mean(params['wrong_wibp'])])
		elif methods[i] == 'ibp':
			params_epoch['ibp'].append([np.mean(params['corr_lbound_ibp']), np.std(params['corr_lbound_ibp']), np.mean(params['wrong_ubound_ibp']), np.std(params['wrong_ubound_ibp']), np.mean(params['corr_ibp']), np.mean(params['wrong_ibp'])])		
		if (t+1) % plot_ind == 0:
			plot_list = []
			for k,v in params_epoch.items():
				plot_list.append(v)
			plot_bounds(plot_list, log_dir_path, name)

#gene_adv_exs
def show_data(fig, X, y, adv_y):
	columns, rows = 3, 1
	fig.add_subplot(rows, columns, 1)
	plt.imshow(transforms.ToPILImage()(X))
	fig.add_subplot(rows, columns, 2)
	plt.imshow(transforms.ToPILImage()(y))
	fig.add_subplot(rows, columns, 3)
	plt.imshow(transforms.ToPILImage()(adv_y))
	plt.draw()
	plt.pause(0.1)


#VAE
class PlotReproducePerformance():
	def __init__(self, savepath, Ni=20, Iw=28, Ih=28, Nic=1, rf=1.0):
		self.dir = savepath
		assert Ni>0
		self.Ni, self.Nic = Ni, Nic
		self.Iw, self.Ih = Iw, Ih
		self.rf = rf

	def save_images(self, images, name='result.jpg'):
		#image vector to [Iw, Ih]
		images = images.reshape(-1, self.Ih, self.Iw, self.Nic) #.cpu().numpy()
		imsave(self.dir + "/" + name, self._merge(images, [self.Ni, self.Ni]))

	def _merge(self, images, size):
		h, w, c = images.shape[1], images.shape[2], images.shape[3]
		h_ = int(h * self.rf)
		w_ = int(w * self.rf)
		if c==1:
			images = images[:,:,:,0]
			img = np.zeros((h_ * size[0], w_ * size[1]))
		else:
			img = np.zeros((h_ * size[0], w_ * size[1], c))
		for idx, image in enumerate(images):
			i = int(idx % size[1])
			j = int(idx / size[1])
			image_ = imresize(image, size=(w_, h_), interp='bicubic')
			if c==1:
				img[j * h_:j * h_ + h_, i * w_:i * w_ + w_] = image_
			else:
				img[j * h_:j * h_ + h_, i * w_:i * w_ + w_, :] = image_
		return img

	def plot(self, x_PRR, n, add_noise):
		# Plot for reproduce performance
		if x_PRR.shape[3]==1:
			x_PRR = x_PRR.squeeze(3)
		x_PRR = x_PRR.clone().cpu().numpy()
		self.save_images(x_PRR, name=n+'.jpg')
		print('saved:', n+'.jpg')
		if add_noise:
			x_PRR = x_PRR * np.random.randint(2, size=x_PRR.shape)
			x_PRR += np.random.randint(2, size=x_PRR.shape)
			self.save_images(x_PRR, name=n+'_noise.jpg')
			print('saved:', n+'_noise.jpg')

class PlotManifoldLearningResult():
	def __init__(self, savepath, Ni=20, Iw=28, Ih=28, Nic=1, rf=1.0, z_range=4, dim_z=2):
		self.dir = savepath
		assert (Ni>0 and z_range>0)
		self.Ni, self.Nic = Ni, Nic
		self.Iw, self.Ih = Iw, Ih
		self.rf = rf
		self.z_range = z_range
		self.dim_z = dim_z
		self.set_latent_vectors()

	def set_latent_vectors(self):
		step = self.Ni**(2/self.dim_z)
		if step%int(step) != 0:
			print(self.Ni, self.dim_z, step)
			raise ValueError("step not int, change dim_z or Ni accordinly")
		grid= np.mgrid[tuple(slice(self.z_range, -self.z_range, complex(0, step)) for _ in range(self.dim_z))]
		#z = np.rollaxis(np.mgrid[self.z_range:-self.z_range:self.Ni*1j, self.z_range:-self.z_range:self.Ni* 1j], 0, 3)
		z = np.rollaxis(grid, 0, self.dim_z+1)
		self.z = z.reshape([-1, self.dim_z])		

	def save_images(self, images, name='result.jpg'):
		#image vector to [Iw, Ih]
		images = images.cpu().numpy().reshape(-1, self.Ih, self.Iw, self.Nic)
		imsave(self.dir + "/" + name, self._merge(images, [self.Ni, self.Ni, self.Nic]))

	def _merge(self, images, size, epoch=0):
		h, w, c = images.shape[1], images.shape[2], images.shape[3]
		h_ = int(h * self.rf)
		w_ = int(w * self.rf)
		if c==1:
			images = images[:,:,:,0]
			img = np.zeros((h_ * size[0], w_ * size[1]))
		else:
			img = np.zeros((h_ * size[0], w_ * size[1], c))
		for idx, image in enumerate(images):
			if epoch==149:
				plt.imsave('pics/'+str(idx)+'.png', image, cmap='gray', vmin=0., vmax=1.)
				print('saving images at ', 'pics/'+str(idx)+'.png')
			i = int(idx % size[1])
			j = int(idx / size[1])
			image_ = imresize(image, size=(w_, h_), interp='bicubic')
			if c==1:
				img[j * h_:j * h_ + h_, i * w_:i * w_ + w_] = image_
			else:
				img[j * h_:j * h_ + h_, i * w_:i * w_ + w_, :] = image_				
		return img

	def plot_latent_vectors_image(self, z, id, name='latent_vectors_image.jpg', N=10):
		z = z.cpu().numpy()
		id = id.cpu().numpy() #np.argmax(id, 1)    .reshape((id.shape[0],-1))
		plt.figure(figsize=(8, 6))
		plt.scatter(z[:, 0], z[:, 1], c=id, marker='o', edgecolor='none', cmap=discrete_cmap(N, 'jet'))
		plt.colorbar(ticks=range(N))
		axes = plt.gca()
		axes.set_xlim([-self.z_range - 2, self.z_range + 2])
		axes.set_ylim([-self.z_range - 2, self.z_range + 2])
		plt.grid(True)
		plt.savefig(self.dir + "/" + name)

	def plot_tSNE_clusters(self, z, id, name='tSNE_image.jpg', N=10):
		z = z.cpu().numpy()
		embd = TSNE(n_jobs=4).fit_transform(z)
		id = id.cpu().numpy() #np.argmax(id, 1)    .reshape((id.shape[0],-1))
		plt.figure(figsize=(8, 6))
		plt.scatter(embd[:, 0], embd[:, 1], c=id, marker='o', edgecolor='none', cmap=discrete_cmap(N, 'jet'))
		plt.colorbar(ticks=range(N))
		axes = plt.gca()
		plt.grid(True)
		plt.savefig(self.dir + "/" + name)

	def plot(self, X, y, Nsamples, add_noise):  
	  # Plot for manifold learning result
		x_PMLR = X[0:Nsamples, :]
		id_PMLR = y[0:Nsamples]
		if add_noise:
			x_PMLR = x_PMLR * np.random.randint(2, size=x_PMLR.shape)
			x_PMLR += np.random.randint(2, size=x_PMLR.shape)
		return x_PMLR



def discrete_cmap(N, base_cmap=None):
	"""Create an N-bin discrete colormap from the specified input map"""

	# Note that if base_cmap is a string or None, you can simply do
	#    return plt.cm.get_cmap(base_cmap, N)
	# The following works for string, None, or a colormap instance:

	base = plt.cm.get_cmap(base_cmap)
	color_list = base(np.linspace(0, 1, N))
	cmap_name = base.name + str(N)
	return base.from_list(cmap_name, color_list, N)