""" Contains functions defining attacks and adversarial example generation. """
import matplotlib.pyplot as plt #TODO remove later
import numpy as np
import math

#pytorch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import transforms
import torch.optim as optim

from visualize import convert_cifar10, denormalize_cifar10, normalize_cifar10
from visualize import save_image


VERBOSE = False
MAX_FLOAT = 1e38	# 1.7976931348623157e+308
columns, rows = 3, 1
		
img_cnt = 0
label2class = {
    (0, 0, 0, 255): 0,  # Sky
    (255, 255, 255, 255): 1,  # Plane
}

def get_attack(a, model, X, y, eps, loss='CE', stats=None, adv_exs=None, save_path=None, noise = [0.001], adv_y=None, segm=False, ethresh=0.9, fig = None, norm='linf', ball='linf', do_err_test=True, dataset='mnist', **kwargs):
	""" Calls a specific attack method.
	
	#Arguments (common for all attacks below):
	- a: one of FGSM, PGD to indicate which attack to make.
	- model: model to be fooled
	- data: [X, y]
	- eps: the maximum l-inf perturbation to be made in the images.
	- loss: one of NLL (Negative Log Likelihood), CE (Cross Entropy)
	- stats:
	- adv_exs: the list of adversarial images and  predictions for visualization
	- save_path: the path of teh folder of the dataset (if None, images are not saved) 
	- adv_y: FGSM is targetted if adv_y is a number, which is then taken as the desired output class label, so keep the number < # of classes.
	
	#Returns:
	-	err = 1 if image misclassified before attack, 0 otherwise.
	- err_atk: 1 if image misclassified after attack, 0 oterwise.
	- err_noise_atk: a list of shape [1, #noise_magnitudes]. The value corressponding to a given noise magnitude is 1 if the trained model misclassifies the adversarial images after the adversarial images have been perturbed with noise of that magnitude, 0 otherwise.
	"""
	if segm == False:
		if a == 'FGSM':
			err, err_atk, err_noise_atk, adv_X = fgsm_attack(model, X, y, adv_y, eps, loss, stats, adv_exs, save_path, noise,  dataset=dataset)
		elif a == 'PGD':
			err, err_atk, err_noise_atk, _, adv_X = pgd_attack(model, X, y, adv_y, eps, stats, adv_exs, save_path, noise, norm=norm, ball=ball, fig = fig, do_err_test=do_err_test, dataset=dataset, **kwargs)
	else:
		err, err_atk, err_noise_atk, adv_X = segm_attack(model, X, y, adv_y, eps, save_path, ethresh, noise=noise)		
	return err, err_atk, err_noise_atk, adv_X


def fgsm_attack(model, X, y, adv_y, eps, loss, stats, adv_exs, save_path, noise, dataset):
	""" Prepares the Fast Gradient Sign Method attack against a model for given data [X, y].""" 
	opt = optim.Adam([X], lr=1e-3)
    
	y_hat = model(X)
	baseline_pred = y_hat.data.max(1, keepdim=True)[1]
	err = (baseline_pred.item() != y.data).float().sum().item()
	
	if baseline_pred.item() != y.data[0]:
		if VERBOSE:
			print('pred:{} while target:{} so model not well trained for this dataset. Try reducing test error first..!'.format(baseline_pred.item(), y.data[0]))
		#return err, 0, [0]*len(noise)
		
	if eps>0:
		if VERBOSE:
			print('pred:{} and target:{}, so attacking..!'.format(baseline_pred.item(), y.data[0]))
		if adv_y is None:
			if loss == 'NLL':
				loss = F.nll_loss(F.log_softmax(y_hat, dim=1), y)
			elif loss == 'CE':
				loss = F.cross_entropy(y_hat, y)
		else:
			if loss == 'NLL':
				loss = F.nll_loss(F.log_softmax(y_hat, dim=1), torch.tensor([adv_y]).cuda()) 
			elif loss == 'CE':
				loss = F.cross_entropy(y_hat, torch.tensor([adv_y]).cuda())		
		opt.zero_grad()
		loss.backward()
		if adv_y is None:
			adv_data = X + eps*X.grad.data.sign()
		else:
			adv_data = X - eps*X.grad.data.sign()
		adv_pred = model(adv_data).data.max(1)[1]
		err_fgsm = (adv_pred.item() != y.data).float().sum()/X.size(0)
	else:
		adv_data = X
		adv_pred = baseline_pred
		err_fgsm = err
	err_noise_fgsm = adv_perturbation_error([adv_data, adv_pred], X, y, baseline_pred, model, stats, adv_exs, eps, save_path, noise, dataset)
	return err, err_fgsm, err_noise_fgsm, adv_data.detach()


def pgd_attack(model, X, y, adv_y, eps, stats, adv_exs, save_path, noise, niters=100, alpha=0.001, norm='linf', sinkhorn_maxiters=50, regularization=2000, ball='linf', epsilon_iters=20, epsilon_factor=1.5, fig=None, do_err_test=True, dataset='mnist'):
	""" Prepares the Fast Gradient Sign Method attack against a model for given data [X, y].
	#Arguments (additional to the common ones):
	-niters: # of iterations for the PGD attack
	-alpha: the step size of the image update in the ascent/descent.
	""" 
	alpha = eps*0.1
	p, kernel_size = 2, 7
	
	y_hat = model(X)
	if fig is not None:
		fig.add_subplot(rows, columns, 1)
		if dataset=='mnist':	
			plt.imshow(X.squeeze().clone().detach().cpu().numpy(), cmap='gray', vmin=0, vmax=1)
		elif dataset=='cifar':
			plt.imshow(convert_cifar10(X.clone().detach().squeeze().cpu()))
		else:
			plt.imshow(np.transpose(X.clone().detach().squeeze(0).cpu().numpy(), (1,2,0)))			
	baseline_pred = y_hat.data.max(1, keepdim=True)[1]
	err_baseline = (baseline_pred != y).float().sum().item()/X.size(0)
	
	if fig is not None:
		if err_baseline==0:
			plt.text(35, -15, 'Baseline prediction : ' + str(baseline_pred.item()), style='italic', bbox={'facecolor': 'green', 'alpha': 0.5, 'pad': 10})
		else:
			plt.text(35, -15, 'Baseline prediction : ' + str(baseline_pred.item()), style='italic', bbox={'facecolor': 'red', 'alpha': 0.5, 'pad': 10})

	epsilon_best = eps
	if err_baseline!=0:
		if VERBOSE:
			print('pred:{} while target:{} so model not well trained for this dataset. Try reducing test error first..!'.format(baseline_pred.item(), y.data[0]))
	#	#return err_baseline, 0, [0]*len(noise), epsilon_best, X.detach()
	
	if eps>0:
		if norm=='wasserstein':
			C = wasserstein_cost(X, p=p, kernel_size=kernel_size)		
		#X = torch.clamp(X, min=0., max=1.)
		X_pgd = Variable(X.data, requires_grad=True)
		err_best = err_baseline
		epsilon_best = eps
		opt = optim.Adam([X_pgd], lr=0.1)
		for i in range(niters): 
			if adv_y is None:
				loss = F.cross_entropy(model(X_pgd), y)
			else:
				loss = F.cross_entropy(model(X_pgd), torch.tensor([adv_y]).cuda())
			opt.zero_grad()
			loss.backward()
			sign = 1 if adv_y is None else -1
			with torch.no_grad():         
				if norm=='linf':
					X_int = X_pgd.clone() + sign*alpha*X_pgd.grad.sign()
				elif norm=='l2':
					X_int = X_pgd.clone() + alpha*X_pgd.grad/(X_pgd.grad.view(X.size(0),-1).norm(dim=1).view(X.size(0),1,1,1))
				elif norm=='wasserstein':
					sd_normalization = X_pgd.view(X.size(0),-1).sum(-1).view(X.size(0),1,1,1)
					X_int = (conjugate_sinkhorn(X_pgd.clone()/sd_normalization, 
	                                               X_pgd.grad, C, alpha, regularization, 
	                                               verbose=VERBOSE, maxiters=sinkhorn_maxiters, fig=fig
	                                               )*sd_normalization)
				else:
					raise ValueError("Unknown norm for PGD attack")
				if fig is not None:
					fig.add_subplot(rows, columns, 2)		
					if dataset == 'mnist':
						plt.imshow(X_int.squeeze().clone().detach().cpu().numpy(), cmap='gray', vmin=0, vmax=1)
					elif dataset == 'cifar':
						plt.imshow(convert_cifar10(X_int.clone().squeeze().detach().cpu()))
					else:
						plt.imshow(np.transpose(X_int.clone().squeeze().detach().cpu(), (1,2,0)))						
					plt.xlabel('x_adv after update in lp norm')
					plt.draw()
					plt.pause(0.1)

				#ball projection
				if ball == 'linf':
					# adjust to be within [-epsilon, epsilon]
					eta = torch.clamp(X_int.data - X.data, -eps, eps)
					X_int = X.data + eta
				elif ball == 'wasserstein':
					normalization = X.view(X.size(0),-1).sum(-1).view(X.size(0),1,1,1)
					X_int = (projected_sinkhorn(X.clone()/normalization, 
	                                          X_int.detach()/normalization, 
	                                          C, eps, regularization, verbose=VERBOSE, 
	                                          maxiters=sinkhorn_maxiters, fig=fig)*normalization)
				else:
					raise ValueError("Unknown ball")
					
				if fig is not None:
					fig.add_subplot(rows, columns, 3)		
					if dataset=='mnist':
						plt.imshow(X_int.squeeze().clone().detach().cpu().numpy(), cmap='gray', vmin=0, vmax=1)
					elif dataset=='cifar':
						plt.imshow(convert_cifar10(X_int.clone().squeeze().detach().cpu()))
					else:
						plt.imshow(np.transpose(X_int.clone().squeeze().detach().cpu().numpy(), (1,2,0)))
					plt.xlabel('x_adv after projection to ball')
					plt.draw()
					plt.pause(0.1)

				adv_pred = model(X_int).max(1, keepdim=True)[1]
				err = (adv_pred != y).float().sum().item()/X.size(0)
				if fig is not None:
					if err == 0:
						plt.text(-34, -7.5, 'Adversarial prediction   ' + "  ", bbox={'facecolor': 'green', 'alpha': 0.5, 'pad': 10})
					else:
						plt.text(-34, -7.5, 'Adversarial prediction : ' + str(adv_pred.item()), bbox={'facecolor': 'red', 'alpha': 0.5, 'pad': 10})
						if ball=='wasserstein':
							plt.text(-50, 90, 'X and X_adv mass (normalized as Joint Probability Matrix) : ' + str((torch.sum(X/normalization).item())) + "    " + str((torch.sum(X_int/normalization).item())), fontsize=10)
							plt.text(-50, 93, 'X and X_adv mass after reverting normalization : ' + str(round(torch.sum(X).item())) + "    " + str(round(torch.sum(X_int).item())), fontsize=10)
					plt.draw()
					plt.pause(0.2)
				if err >= err_best:
					X_pgd = Variable(X_int, requires_grad=True)
					err_best = err
					epsilon_best = eps
				if err_best == 1:
					break
				if i > 0 and i % epsilon_iters == 0 and False: ##CHECK
					eps *= epsilon_factor
		if fig is not None:
			if err == 0:
				plt.text(-50, 97, 'Could NOT find an adversarial example till epsilon : ' + str(eps), fontsize=10) #47
			else:
				plt.text(-50, 97, 'Found an attack within epsilon : ' + str(epsilon_best), fontsize=10)
			plt.draw()
			plt.pause(2)
			#vvv = input()
			plt.clf()
        #endFOR (inner PGD iterations)
		adv_data = X_pgd
		adv_pred = model(X_pgd).data.max(1)[1]
		err_pgd = (adv_pred != y).float().sum().item()/X.size(0)
		if VERBOSE:
			print('pred:{} and target:{} and adv_pred:{} and err_best:{}..!'.format(baseline_pred.item(), y.data[0], adv_pred.item(), err_best))
	else:
		adv_data = X
		adv_pred = baseline_pred
		err_pgd = err_baseline
	if do_err_test:
		err_noise_pgd = adv_perturbation_error([adv_data, adv_pred], X, y, baseline_pred, model, stats, adv_exs, eps, save_path, noise, dataset)
	else:
		# TODO, change it to NaN later and not plot it or stat it.
		err_noise_pgd = 0
	return err_baseline, err_pgd, err_noise_pgd, epsilon_best, adv_data.detach()

#extra
def adv_perturbation_error(adv_data, X, y, baseline_pred, model, stats, adv_exs, eps, save_path, noise, dataset):
	""" does random affine transforms to the perturbed image to see if they lose their attack on such transforms. 
			It is not unlikely because, if the NNs are not rotation invariant so a attack aiming at fooling the NN should not have rotation invariance and should lose its effect. see https://arxiv.org/pdf/1707.03501.pdf. 
			
	#Arguments:
	- adv_data: [X_perturbed, prediction of the model to this X_perturbed]
	- act_data: original X and its label
	- baseline_pred: the prediction of the baseline model to the untampered/original input
	- model, stats, adv_exs, eps, save_path: explained in the doctring of calling function
	- noise: list of noise to be added and tested for affect on adversarial examples
	
	#Returns:
	a list of shape [1, #noise_magnitudes]. The value corressponding to a given noise magnitude is 1 if adversarial images is misclassified after perturbation of that magnitude, 0 otherwise 
	NOTE: When eps = 0, the return value indicates the rotation (noise) invariance of the baseline network; should be 0 for a well-trained network 
	"""
	err_noise_pgd = []
	for n in range(len(noise)):
		transform = transforms.Compose([transforms.ToPILImage(),
										#transforms.RandomAffine((-5,5), scale=(0.95,1.05)),
										transforms.ToTensor(),
										transforms.Lambda(lambda x : x + n*torch.randn(*x.size())),])
		#perturbing clean data with the same noise to see that the model resists noise affect without adversary;
		#else adversary data will intuitively not be getting corrected towards the model clean class. 
		if dataset == 'mnist':
			pert_act_data = transform(X.data.clone().cpu().detach().squeeze(0))
		elif dataset == 'cifar':
			pert_act_data = transform(torch.tensor(convert_cifar10(X.clone().cpu().detach().squeeze(0), t=False), dtype=torch.float))			
		else:
			pert_act_data = transform(X.data.clone().cpu().detach().squeeze(0))			
		pert_act_hat = model(pert_act_data.unsqueeze(0).cuda())
		pert_act_pred = pert_act_hat.max(1, keepdim=True)[1]
		if pert_act_pred.item() == y.item():
			stats['baseline_noise_acc'][n] += 1 #shd be high, means model is robust and well-trained for noise.
		
		if dataset=='mnist':
			pert_adv_data = transform(adv_data[0].cpu().detach().squeeze().unsqueeze(0))
		elif dataset == 'cifar':
			pert_adv_data = transform(torch.tensor(convert_cifar10(adv_data[0].cpu().detach().squeeze(0), t=False), dtype=torch.float))			
		else:
			pert_adv_data = transform(adv_data[0].cpu().detach().squeeze(0))			
		pert_adv_hat = model(pert_adv_data.unsqueeze(0).cuda())
		pert_adv_pred = pert_adv_hat.max(1, keepdim=True)[1]
		
		if dataset=='mnist':
			make_adv_dataset_visualization(adv_data[0], [y.item(), baseline_pred.item(), adv_data[1].item(), pert_adv_pred.item()], eps, adv_exs, save_path)
		elif dataset == 'cifar':
			make_adv_dataset_visualization(torch.tensor(convert_cifar10(adv_data[0].cpu().detach().squeeze(0), t=False), dtype=torch.float).unsqueeze(0), [y.item(), baseline_pred.item(), adv_data[1].item(), pert_adv_pred.item()], eps, adv_exs, save_path)	
		else:
			make_adv_dataset_visualization(adv_data[0], [y.item(), baseline_pred.item(), adv_data[1].item(), pert_adv_pred.item()], eps, adv_exs, save_path)
		
		if baseline_pred.item() == y.item():
			if pert_adv_pred.item() == y.item():
				if adv_data[1].item() == y.item():
					stats['baseline_adv_noise_corr'][n] += 1 #shd be high, means either model is robust and well-trained or adv attack is not good enough, strengthen it..!
				elif adv_data[1].item() != y.item():
					stats['baseline_noise_corr_adv_aff'][n] += 1 #shd be high, noise nullified the effect of adversary, so supports this as a defense against adversarial attack.
			else:
				if adv_data[1].item() == y.item():
					stats['baseline_adv_corr_noise_aff'][n] += 1 #shd be small, if large shows that while training augmentations must be less, because just noise and not attack has affected the model
				elif adv_data[1].item() == pert_adv_pred.item():
					stats['baseline_corr_adv_aff_noise_steady'][n] += 1 #shd be small, if large shows that noise doesn't help in defense (as wrong pred was stable)
				elif adv_data[1].item() != y.item() and adv_data[1].item() != pert_adv_pred.item():
					stats['baseline_corr_adv_aff_noise_help'][n] += 1 #shd be large, shows that even if model is not robust to attacks and noise can help in defense (as adversarial images shift classes) --> similar to stats['baseline_noise_corr_adv_aff'] where noise was even more helpful.		
		err_noise_pgd.append((pert_adv_pred.item() != y.data).float().sum().item())
	return err_noise_pgd
	

def make_adv_dataset_visualization(image, preds=None, eps=None, adv_exs=None, save_path=None, segm=False, label=None, label_pred=None, label_adv=None, target_mask=None):
	""" saves the adversial example in a dataset at save_path."""
	adv_ex = image.squeeze().detach().cpu().numpy()
	global img_cnt
	if not segm:
		ae = [*preds, adv_ex]
		#for visualization keeping 5 images
		if preds[2] == preds[0]:
			if (eps == 0) and (len(adv_exs) < 5):
				adv_exs.append(ae)
		else:
			if len(adv_exs) < 5:
				adv_exs.append(ae)
		#HH TODO change to save only if adversarial
			if save_path is not None:
				img_cnt += 1
				#save_image(save_path+str(preds[0])+'_'+str(preds[1])+'_'+str(preds[2])+'_'+str(preds[3])+'_'+str(img_cnt)+'.png', adv_ex)
	else:
		if save_path is not None:
			img_cnt += 1
			save_image(save_path+str(img_cnt)+'_x.png', np.transpose(adv_ex, (1, 2, 0)))
			save_image(save_path+str(img_cnt)+'_y.png', np.transpose(label, (1, 2, 0)))
			save_image(save_path+str(img_cnt)+'_ypred.png', np.transpose(label_pred, (1, 2, 0)))
			save_image(save_path+str(img_cnt)+'_yadv.png', np.transpose(label_adv, (1, 2, 0)))
			#save_image(save_path+str(img_cnt)+'_target.png', np.transpose(target_mask, (1, 2, 0)))


#####SEGMENTATION ATTACKS
def segm_attack(model, X, Y, adv_Y, eps, save_path, ethresh, niters=400, alpha=0.6, noise=[0]):
	#baseline
	out = (model(X))
	assert out.shape==Y.shape
	main_labels = np.argmax(out.data.cpu().numpy(), 1)
	Y_labels = np.zeros((len(label2class), out.shape[2], out.shape[3]))
	for label in range(len(label2class)):
		mask = (main_labels[:,:,:]==label)
		print('mask shape ori: ', mask.shape)
		Y_labels[label,:,:] = mask*1.
	
	target_mask = torch.from_numpy((adv_Y.data.cpu().numpy()!=Y.data.cpu().numpy())*1.).cuda().float()
	
	err = np.absolute(Y_labels-Y.data.cpu().numpy()).sum()/(Y.size(0)*Y.size(2)*Y.size(3))
	if err > ethresh:
		print('err:{} so model not well trained for this dataset. Try reducing test error first..!'.format(err.item()))
		return err, 0, [0]*len(noise), X
	
	if eps>0:
		print('err:{}, so attacking..!'.format(err.item()))
		X_segm = Variable(X.data, requires_grad=True)
		
		target_mask_X = torch.sum(target_mask, dim=1).squeeze(0)/len(label2class)
		print('unique elements in target_mask_X: ', torch.unique(target_mask_X))
		TargetMask_X = torch.zeros(X_segm.size()).cuda().float()
		for i in range(3):
			TargetMask_X[0,i,:,:] = target_mask_X
		
		for i in range(niters): 
			opt = optim.SGD([X_segm], lr=0.7, momentum=0.9, weight_decay=5e-4)				
			#loss_correct = nn.BCELoss(reduce=True)(torch.sigmoid(model(X_segm)), Y)
			loss_incorrect = nn.BCELoss(reduce=True)(torch.sigmoid(model(X_segm))*target_mask_X, adv_Y*target_mask_X)
			loss = loss_incorrect
			opt.zero_grad()
			loss.backward()
			opt.step()
			print('here')			
			
			X_segm = Variable(X_segm.data + alpha*X_segm.grad.data.sign()*TargetMask_X, requires_grad=True)		#*TargetMask_X   /torch.norm(X_segm.grad.data.sign()*TargetMask_X, p=float("inf"))
			# adjust to be within [-epsilon, epsilon]
			eta = torch.clamp(X_segm.data - X.data, -eps, eps)
			X_segm = Variable(X.data + eta, requires_grad=True)		
		
			adv_data = X_segm
			out_adv = (model(adv_data))
			#print(torch.max(out[0,0,:,:]),torch.max(out_adv[0,0,:,:]))
			assert out_adv.shape==Y.shape
			main_labels_adv = np.argmax(out_adv.data.cpu().numpy(), 1)
			Y_labels_adv = np.zeros((len(label2class), out_adv.shape[2], out_adv.shape[3]))
			for label in range(len(label2class)):
				mask_adv = (main_labels_adv[:,:,:]==label)
				print('mask_adv shape ori: ', mask_adv.shape)
				Y_labels_adv[label,:,:] = mask_adv*1.
			adv_pred = Y_labels_adv
			err_adv = np.absolute(adv_pred-Y.data.cpu().numpy()).sum()/(Y.size(0)*Y.size(2)*Y.size(3))
			print('err_adv : ', err_adv)
			if err_adv > 0.51: 
				break
		if err_adv > 0*ethresh: 
			make_adv_dataset_visualization(adv_data[0].unsqueeze(0), eps=eps, save_path=save_path, segm=True, label=convert_to_label(adv_Y), label_pred=convert_to_label(torch.from_numpy(Y_labels).unsqueeze(0), out), label_adv=convert_to_label(torch.from_numpy(Y_labels_adv).unsqueeze(0), out_adv)) #,target_mask=convert_to_label(target_mask))	
	else:
		adv_data = X
		adv_pred = Y_labels
	err_segm = np.absolute(adv_pred-Y.data.cpu().numpy()).sum()/(Y.size(0)*Y.size(2)*Y.size(3))
	print('err segm : ', err_segm)
	#err_noise_pgd = adv_perturbation_error([adv_data, adv_pred], X, y, baseline_pred, model, stats, adv_exs, eps, save_path, noise)
	return err, err_segm, [0]*len(noise), adv_data


## taken from https://github.com/locuslab/projected_sinkhorn
#utils for wasserstein
def any_nan(X): 
    return (X != X).any().item()
def any_inf(X): 
    return (X == float('inf')).any().item()

def _expand(X, shape): 
    return X.view(*X.size()[:-1], *shape)
def _expand_filter(X, nfilters): 
    sizes = list(-1 for _ in range(X.dim()))
    sizes[-3] = nfilters
    return X.expand(*sizes)

def _unfold(x, kernel_size, padding=None): 
    # this is necessary because unfold isn't implemented for multidimensional batches
    size = x.size()
    if len(size) > 4: 
        x = x.contiguous().view(-1, *size[-3:])
    out = F.unfold(x, kernel_size, padding=kernel_size//2)
    if len(size) > 4: 
        out = out.view(*size[:-3], *out.size()[1:])
    return out

def _mm(A,x, shape): 
    kernel_size = A.size(-1)
    nfilters = shape[1]
    unfolded = _unfold(x, kernel_size, padding=kernel_size//2).transpose(-1,-2)
    unfolded = _expand(unfolded, (A.size(-3),A.size(-2)*A.size(-1))).transpose(-2,-3)
    out = torch.matmul(unfolded, collapse2(A.contiguous()).unsqueeze(-1)).squeeze(-1)
    return unflatten2(out)

def unsqueeze3(X):
    return X.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

# batch dot product
def _bdot(X,Y): 
    return torch.matmul(X.unsqueeze(-2), Y.unsqueeze(-1)).squeeze(-1).squeeze(-1)
def bdot(x,y): 
    return _bdot(collapse3(x), collapse3(y))

def collapse2(X): 
    return X.view(*X.size()[:-2], -1)
def collapse3(X): 
    return X.contiguous().view(*X.size()[:-3], -1)

def unflatten2(X): 
    # print('unflatten2', X.size())
    n = X.size(-1)
    k = int(math.sqrt(n))
    return _expand(X,(k,k))

def conjugate_sinkhorn(*args, **kwargs): 
    return log_sinkhorn(*args, objective='conjugate', **kwargs)
def projected_sinkhorn(*args, **kwargs): 
    return log_sinkhorn(*args, objective='2norm', **kwargs)

def log_sinkhorn(X, Y, C, epsilon, lam, verbose=False, plan=False,
   objective='2norm', maxiters=50, return_objective=False, fig=None):  
    """ 
    if objective == '2norm': 
        minimize_Z ||Y-Z||_2 subject to Z in Wasserstein ball around X 
        we return Z

    if objective == 'conjugate': 
        minimize_Z -Y^TZ subject to Z in Wasserstein ball around X
        however instead of Z we return the dual variables u,psi for the 
        equivalent objective: 
        minimize_{u,psi} 
    Inputs: 
        X : batch size x nfilters x image width x image height
        Y : batch size x noutputs x total input dimension
    """				
    batch_sizes = X.size()[:-3]
    nfilters = X.size(-3)

    # size check
    for xd,yd in (zip(reversed(X.size()),reversed(Y.size()))): 
        assert xd == yd or xd == 1 or yd == 1

    # helper functions
    expand_filter = lambda x: _expand_filter(x, X.size(-3))
    mm = lambda A,x: _mm(expand_filter(A),x,X.size())
    norm = lambda x: torch.norm(collapse3(x), dim=-1)
    # like numpy
    allclose = lambda x,y: (x-y).abs() <= 1e-4 + 1e-4*y.abs()
    
    # assert valid distributions
    assert (X>=0).all()
    assert ((collapse3(X).sum(-1) - 1).abs() <= 1e-4).all()
    assert X.dim() == Y.dim()

    size = tuple(max(sx,sy) for sx,sy in zip(X.size(), Y.size()))
    m = collapse3(X).size(-1)
    
    if objective == 'conjugate': 
        alpha = torch.log(X.new_ones(*size)/m) + 0.5
        exp_alpha = torch.exp(-alpha)
        beta = -lam*Y.expand_as(alpha).contiguous()
        exp_beta = torch.exp(-beta)

        # check for overflow
        if (exp_beta == float('inf')).any(): 
            print(beta.min())
            raise ValueError('Overflow error: in logP_sinkhorn for e^beta')

        # EARLY TERMINATION CRITERIA: if the nu_1 and the 
        # center of the ball have no pixels with overlapping filters, 
        # thenthe wasserstein ball has no effect on the objective. 
        # Consequently, we should just return the objective 
        # on the center of the ball. Notably, if the filters don't overlap, 
        # then the pixels themselves don't either, so we can conclude that 
        # the objective is 0. 

        # We can detect overlapping filters by applying the cost 
        # filter and seeing if the sum is 0 (e.g. X*C*Y)
        C_tmp = C.clone() + 1
        while C_tmp.dim() < Y.dim(): 
            C_tmp = C_tmp.unsqueeze(0)
        I_nonzero = bdot(X,mm(C_tmp,Y)) != 0
        I_nonzero_ = unsqueeze3(I_nonzero).expand_as(alpha)

        def eval_obj(alpha, exp_alpha, psi, K): 
            return (-psi*epsilon - bdot(torch.clamp(alpha,max=MAX_FLOAT),X) - bdot(exp_alpha, mm(K, exp_beta)))/lam
        def eval_z(alpha, exp_alpha, psi, K): 
            return (-psi*epsilon - mm(torch.clamp(alpha,max=MAX_FLOAT),X) - mm(exp_alpha, mm(K, exp_beta)))/lam

        psi = X.new_ones(*size[:-3])
        K = torch.exp(-unsqueeze3(psi)*C - 1)

        old_obj = -float('inf')
        i = 0

        with torch.no_grad(): 
            while True: 
                alpha[I_nonzero_] = (torch.log(mm(K,exp_beta)) - torch.log(X))[I_nonzero_]
                exp_alpha = torch.exp(-alpha)

                dpsi = -epsilon + bdot(exp_alpha,mm(C*K,exp_beta))
                ddpsi = -bdot(exp_alpha,mm(C*C*K,exp_beta))
                delta = dpsi/ddpsi

                psi0 = psi
                t = X.new_ones(*delta.size())
                neg = (psi - t*delta < 0)
                while neg.any() and t.min().item() > 1e-2:
                    t[neg] /= 2
                    neg = psi - t*delta < 0
                psi[I_nonzero] = torch.clamp(psi - t*delta, min=0)[I_nonzero]

                K = torch.exp(-unsqueeze3(psi)*C - 1)

                # check for convergence
                obj = eval_obj(alpha, exp_alpha, psi, K)
                if verbose: 
                    print('obj', obj)
                i += 1
                if i > maxiters or allclose(old_obj,obj).all(): 
                    if verbose: 
                        print('terminate at iteration {}'.format(i))
                    break
                z = eval_z(alpha, exp_alpha, psi, K)
                z[~I_nonzero] = 0
                old_obj = obj

        if return_objective: 
            obj = -bdot(X,Y)
            obj[I_nonzero] = eval_obj(alpha, exp_alpha, psi, K)[I_nonzero]
            return obj
        else: 
            z = eval_z(alpha, exp_alpha, psi, K)
            z[~I_nonzero] = 0
            if fig is not None:
                fig.add_subplot(rows, columns, 2)		
                plt.imshow(z.squeeze().detach().cpu().numpy(), cmap='gray', vmin=0, vmax=1)
                plt.draw()
                plt.pause(0.1)
            return z

    elif objective == '2norm': 
        alpha, beta = torch.log(X.new_ones(*size)/m), torch.log(X.new_ones(*size)/m)
        u, v = torch.exp(-alpha), torch.exp(-beta)
        # check for overflow
        if (v == float('inf')).any() or (u == float('inf')).any(): 
            print(alpha.min(), beta.min())
            raise ValueError('Overflow error: in logP_sinkhorn for e^alpha, e^beta')

        psi = X.new_ones(*size[:-3])
        K = torch.exp(-unsqueeze3(psi)*C - 1)
        
        def eval_obj(alpha, beta, u, v, psi, K): 
            return (-0.5/lam*bdot(beta,beta) - psi*epsilon 
                    - bdot(torch.clamp(alpha,max=1e10),X) 
                    - bdot(torch.clamp(beta,max=1e10),Y)
                    - bdot(u, mm(K, v)))

        old_obj = -float('inf')
        i = 0
        if verbose:
            start_time = time.time()
        with torch.no_grad(): 
            while True: 
                alphat = norm(alpha)
                betat = norm(beta)

                alpha = (torch.log(mm(K,v)) - torch.log(X))
                u = torch.exp(-alpha)
                
                beta = lambertw(lam*torch.exp(lam*Y)*mm(K,u)) - lam*Y
                v = torch.exp(-beta)

                dpsi = -epsilon + bdot(u,mm(C*K,v))
                ddpsi = -bdot(u,mm(C*C*K,v))
                delta = dpsi/ddpsi
                
                psi0 = psi
                t = X.new_ones(*delta.size())
                neg = (psi - t*delta < 0)
                while neg.any() and t.min().item() > 1e-2:
                    t[neg] /= 2
                    neg = psi - t*delta < 0
                psi = torch.clamp(psi - t*delta, min=0)
                K = torch.exp(-unsqueeze3(psi)*C - 1)

                # check for convergence
                obj = eval_obj(alpha, u, beta, v, psi, K) 
                i += 1
                if i > maxiters or allclose(old_obj,obj).all(): 
                    if verbose: 
                        print('terminate at iteration {}'.format(i), maxiters)
                        if i > maxiters: 
                            print('warning: took more than {} iters'.format(maxiters))
                    break
                old_obj = obj
        return beta/lam + Y


def wasserstein_cost(X, p=2, kernel_size=None):
    if kernel_size is None:
        kernel_size = X.shape[-1] if (X.shape[-1]%2)==1 else (X.shape[-1]-1) 
    if kernel_size % 2 != 1: 
        raise ValueError("Need odd kernel size")        
    center = kernel_size // 2
    C = X.new_zeros(kernel_size,kernel_size)
    for i in range(kernel_size): 
        for j in range(kernel_size): 
            C[i,j] = (abs(i-center)**2 + abs(j-center)**2)**(p/2)
    return C

def convert_to_label(Y, preds=None):
	# [1, 10, N, N]
	y = np.zeros((3, Y.size()[2], Y.size()[3]), dtype="uint8")
	for k,v in label2class.items():
		c = (Y[0,v,:,:]==1.).cpu().numpy()
		if preds is None:
			y[0,:,:] = np.add(y[0,:,:], k[0]*c)
			y[1,:,:] = np.add(y[1,:,:], k[1]*c)
			y[2,:,:] = np.add(y[2,:,:], k[2]*c)
		else:
			y[0,:,:] = np.add(y[0,:,:], k[0]*c*preds[0,v,:,:].data.cpu().numpy())
			y[1,:,:] = np.add(y[1,:,:], k[1]*c*preds[0,v,:,:].data.cpu().numpy())
			y[2,:,:] = np.add(y[2,:,:], k[2]*c*preds[0,v,:,:].data.cpu().numpy())			
	return y


OMEGA = 0.56714329040978387299997  # W(1, 0)
EXPN1 = 0.36787944117144232159553  # exp(-1)

def evalpoly(coeff, degree, z): 
    powers = torch.arange(degree,-1,-1).float().to(z.device)
    return ((z.unsqueeze(-1)**powers)*coeff).sum(-1)

def lambertw(z0, tol=1e-5): 
    # this is a direct port of the scipy version for the 
    # k=0 branch for *positive* z0 (z0 >= 0)

    # skip handling of nans
    if torch.isnan(z0).any(): 
        raise NotImplementedError

    w0 = z0.new(*z0.size())
    # under the assumption that z0 >= 0, then I_branchpt 
    # is never used. 
    I_branchpt = torch.abs(z0 + EXPN1) < 0.3
    I_pade0 = (-1.0 < z0)*(z0 < 1.5)
    I_asy = ~(I_branchpt | I_pade0)
    if I_pade0.any(): 
        z = z0[I_pade0]
        num = torch.Tensor([
            12.85106382978723404255,
            12.34042553191489361902,
            1.0
        ]).to(z.device)
        denom = torch.Tensor([
            32.53191489361702127660,
            14.34042553191489361702,
            1.0
        ]).to(z.device)
        w0[I_pade0] = z*evalpoly(num,2,z)/evalpoly(denom,2,z)

    if I_asy.any(): 
        z = z0[I_asy]
        w = torch.log(z)
        w0[I_asy] = w - torch.log(w)

    # split on positive and negative, 
    # and ignore the divergent series case (z=1)
    w0[z0 == 1] = OMEGA
    I_pos = (w0 >= 0)*(z0 != 1)
    I_neg = (w0 < 0)*(z0 != 1)
    if I_pos.any(): 
        w = w0[I_pos]
        z = z0[I_pos]
        for i in range(100): 
            # positive case
            ew = torch.exp(-w)
            wewz = w - z*ew
            wn = w - wewz/(w + 1 - (w + 2)*wewz/(2*w + 2))

            if (torch.abs(wn - w) < tol*torch.abs(wn)).all():
                break
            else:
                w = wn
        w0[I_pos] = w

    if I_neg.any(): 
        w = w0[I_neg]
        z = z0[I_neg]
        for i in range(100):
            ew = torch.exp(w)
            wew = w*ew
            wewz = wew - z
            wn = w - wewz/(wew + ew - (w + 2)*wewz/(2*w + 2))
            if (torch.abs(wn - w) < tol*torch.abs(wn)).all():
                break
            else:
                w = wn
        w0[I_neg] = wn
    return w0