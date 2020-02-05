#python bounds_images.py --datasets mnist --models conv_2layer --model_files ~/masters_thesis_notebooks/master_repo/outputs/trainings/mnist_small_0_3.pth --batch_size 1 --max_iter 50 --lr 0.03 --mode semi_nonuniform
import os
import datetime
import setproctitle
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import autograd

import configs 
import models
import datasets
from convex_adversarial import get_RobustBounds as robust_bounds_wong
from convex_adversarial import robust_loss as robust_loss_wong


def get_eps(args, X):
	if args.mode == 'uniform':
		eps = torch.zeros([args.batch_size, 1], device = "cuda", requires_grad = True)
		eps.data.fill_(np.sqrt(args.init_margin))
		epsilon = [eps,]
	elif args.mode in ['semi_nonuniform', 'covariance']:
		eps = torch.zeros([*X.shape], device = "cuda", requires_grad = True)
		eps.data.fill_(np.sqrt(args.init_margin))
		epsilon = [eps,]
	elif args.mode == 'nonuniform':
		eps = torch.zeros([*X.shape], device = "cuda")
		eps.data.fill_(np.sqrt(args.init_margin))
		epsilon = [Variable(eps.clone(), requires_grad = True), Variable(eps.clone(), requires_grad = True),]
	else:
		raise ValueError('Unrecognized mode: %s' % args.mode)
	return epsilon

def get_optimiser(args, epsilon, An):
	optims = []
	if args.mode in ['uniform', 'covariance', 'semi_nonuniform']:
		optims.append(torch.optim.Adam(epsilon, lr = args.lr))
		if args.mode=='covariance':
			optims.append(torch.optim.Adam([An,], lr = args.lr_w))
	elif args.mode=='nonuniform':
		optims.append(torch.optim.Adam([epsilon[0],], lr = args.lr))
		optims.append(torch.optim.Adam([epsilon[1],], lr = args.lr))
	else:
		raise ValueError('Unrecognized mode: %s' % args.mode)
	return optims

def get_bounds(model, epsilon, X, y, norm, mode, An):
	if mode=='uniform':
		lb, ub, _, _ = robust_bounds_wong(model, epsilon[0], X, y, norm, non_uniform=False)		
	elif mode=='covariance':
		epsilon_A = torch.mm(epsilon[0], An.t())
		lb, ub, _, _ = robust_bounds_wong(model, [epsilon_A, epsilon_A], X, y, norm, non_uniform=True)	
	elif mode=='semi_nonuniform':
		lb, ub, _, _ = robust_bounds_wong(model, [epsilon[0], epsilon[0]], X, y, norm, non_uniform=True)	
	elif mode=='nonuniform':
		lb, ub, _, _ = robust_bounds_wong(model, epsilon, X, y, norm, non_uniform=True)	
	else:
		raise ValueError('Unrecognized mode: %s' % mode)
	return lb, ub

def get_f(model, epsilon, X, y, norm, mode, An):
	if mode=='uniform':
		_, _, _, _, _, _, _, f = robust_loss_wong(model, epsilon[0], X, y, get_bounds=False, non_uniform=False)
	elif mode=='covariance':
		epsilon_A = torch.mm(epsilon[0], An.t())
		_, _, _, _, _, _, _, f = robust_loss_wong(model, [epsilon_A, epsilon_A], X, y, get_bounds=False, non_uniform=True)
	elif mode=='semi_nonuniform':
		_, _, _, _, _, _, _, f = robust_loss_wong(model, [epsilon[0], epsilon[0]], X, y, get_bounds=False, non_uniform=True)
	elif mode=='nonuniform':
		_, _, _, _, _, _, _, f = robust_loss_wong(model, epsilon, X, y, get_bounds=False, non_uniform=True)
	else:
		raise ValueError('Unrecognized mode: %s' % mode)
	return -f

def get_objective(epsilon, An, mode):
	if mode=='uniform':
		return - (torch.sum(torch.log(epsilon[0]*epsilon[0]), dim = 1))			
	elif mode=='covariance':
		epsilon_A = torch.mm(epsilon[0], An.t())
		return - (torch.sum(torch.log(epsilon_A*epsilon_A), dim = 1))			
	elif mode=='semi_nonuniform':
		return - (torch.sum(torch.log(epsilon[0]*epsilon[0]), dim = 1))			
	elif mode=='nonuniform':
		#print('nonuniform objective')
		return - (torch.sum(torch.log(epsilon[0]*epsilon[0]), dim = 1)) - (torch.sum(torch.log(epsilon[1]*epsilon[1]), dim = 1))			

def update_step(optims, mode, index, A=False):
	if mode in ['uniform', 'semi_nonuniform']:
		for optim in optims:
			optim.step()			
	elif mode=='nonuniform':
		optims[index].step()
	elif mode=='covariance':
		if A:
			optims[1].step()
		else:
			optims[0].step()

def compute_eps(dataset, dataloader, model, args, base_pts, log, train_loader, in_dim, out_dim, log_dir_path):
	data_batch, label_batch, result_mask_batch, eps_lower_batch, eps_upper_batch, A_batch, Volume_batch, avg_lower_batch, avg_upper_batch = [], [], [], [], [], [], [], [], []	

	for i, (X,y) in enumerate(dataloader):
		if i==2:
			break
		X,y = X.cuda(), y.cuda().long()
		if y.dim() == 2: 
			y = y.squeeze(1)
		y_hat = model(Variable(X))
		corr_pred_mask = (y_hat.max(1)[1] == y).float()
		if corr_pred_mask==0:
			print('wrong pred', y.item(), y_hat.max(1)[1].item())
			continue
		else:
			print('right pred', y.item(), y_hat.max(1)[1].item())
		label_mask = torch.ones([args.batch_size, out_dim], device="cuda")
		label_mask = label_mask.scatter_(dim = 1, index = y.view(args.batch_size, 1), value = 0)
		
		A = torch.eye(in_dim, requires_grad = False, device="cuda")
		An = Variable(A/torch.norm(A), requires_grad=True)
		epsilon = get_eps(args, X)
		optims = get_optimiser(args, epsilon, An)

		#dual and penalty vars
		rho = args.beta
		grad_clip = args.grad_clip
		lam = torch.zeros([args.batch_size, out_dim], requires_grad=False, device = "cuda")
		if args.mode=='nonuniform':
			index = 2
		else:
			index = 1			
		
		#for iter_idx in range(args.max_iter):
		for ind in range(index):
			for iter_idx in range(args.max_iter):
				lb, ub = get_bounds(model, epsilon, X, y, 'l1', args.mode, An)
				low_c = lb.gather(1, y.view(-1, 1))
				v = low_c - ub - args.delta
				'''
				f = get_f(model, epsilon, X, y, 'l1', args.mode, An)
				v = f	
				'''
				err = torch.min(v, - lam / rho) * label_mask
				eps_loss = get_objective(epsilon, An, args.mode)	
				err_loss = torch.sum(lam * err, dim = 1) + rho / 2. * torch.norm(err, dim = 1) ** 2
				loss = torch.sum((eps_loss + err_loss) * corr_pred_mask)/torch.sum(corr_pred_mask)

				if args.mode=='covariance' and torch.any(v<0):
					lb, ub = get_bounds(model, epsilon, X, y, 'l1', args.mode, An)
					low_c = lb.gather(1, y.view(-1, 1))
					v = low_c - ub - args.delta
					'''
					f = get_f(model, epsilon, X, y, 'l1', args.mode, An)
					v = f
					'''
					err = torch.min(v, - lam / rho) * label_mask
					eps_loss = get_objective(epsilon, An, args.mode)	
					err_loss = torch.sum(lam * err, dim = 1) + rho / 2. * torch.norm(err, dim = 1) ** 2
					loss = torch.sum((eps_loss + err_loss) * corr_pred_mask)/torch.sum(corr_pred_mask)
					for optim in optims:
						optim.zero_grad()
					loss.backward(retain_graph=True)
					update_step(optims, args.mode, ind, A=True)

				for optim in optims:
					optim.zero_grad()
				loss.backward(retain_graph=True)
				update_step(optims, args.mode, ind, A=False)

				#updating lagrangian multiplier
				if (iter_idx + 1) % args.update_dual_freq == 0:
					lam.data = lam.data + rho * err
				#updating penalty parameters
				if iter_idx + 1 > args.inc_min and (iter_idx + 1 - args.inc_min) % args.inc_freq == 0:
					rho *= args.inc_rate
					if args.grad_clip is not None:
						grad_clip /= np.sqrt(args.inc_rate)
				if iter_idx==args.max_iter:
					print('reached max_iter', iter_idx)
		shrink_times = 0
		while shrink_times < 1000:
			for ind in range(index):			
				lb, ub = get_bounds(model, epsilon, X, y, 'l1', args.mode, An)
				low_c = lb.gather(1, y.view(-1, 1))
				v = low_c - ub - args.delta
				'''
				f = get_f(model, epsilon, X, y, 'l1', args.mode, An)
				v = f
				'''
				err_min, _ = torch.min(v * label_mask + 1e-10, dim = 1, keepdim = True)
				err_min = err_min * corr_pred_mask.view(-1, 1) + 1e-10
				shrink_times += 1
				coeff = ((1. - args.final_decay) / 2. * torch.sign(err_min) + (1. + args.final_decay) / 2.)
				if args.mode=='nonuniform':
					epsilon[ind].data = epsilon[ind].data * coeff
				elif args.mode in ['uniform', 'semi_nonuniform']:
					epsilon[0].data = epsilon[0].data * coeff
		print('Shrink time = %d' % shrink_times)
		data_batch.extend(X.data.cpu().numpy())
		label_batch.extend(y.data.cpu().numpy())
		result_mask_batch.extend(corr_pred_mask.data.cpu().numpy())
		eps_lower_batch.extend(epsilon[0].clamp(min=0).clamp(max=1.).detach().cpu().numpy())
		lavg = np.average(np.average(epsilon[0].clamp(min=0).clamp(max=1.).detach().cpu().numpy(), axis=0))
		avg_lower_batch.append(lavg)		
		
		if args.show_images:
			fig2=plt.figure(figsize=(8, 8))
			fig2.add_subplot(1, 3, 1)
			plt.imshow((X[0][0].cpu()-epsilon[0][0][0].clamp(min=0).clamp(max=1.).detach().cpu()), cmap='gray', vmin=0, vmax=1)
			fig2.add_subplot(1, 3, 2)
			plt.imshow(X[0][0].cpu(), cmap='gray', vmin=0, vmax=1)
			fig2.add_subplot(1, 3, 3)

		if args.mode=='nonuniform':
			eps_upper_batch.extend(epsilon[1].clamp(min=0).clamp(max=1.).detach().cpu().numpy())
			uavg = np.average(np.average(epsilon[1].clamp(min=0).clamp(max=1.).detach().cpu().numpy(), axis=0))
			if args.show_images:
				plt.imshow((X[0][0].cpu()+epsilon[1][0][0].clamp(min=0).clamp(max=1.).detach().cpu()), cmap='gray', vmin=0, vmax=1)
		else:
			eps_upper_batch.extend(epsilon[0].clamp(min=0).clamp(max=1.).detach().cpu().numpy())
			uavg = np.average(np.average(epsilon[0].clamp(min=0).clamp(max=1.).detach().cpu().numpy(), axis=0))
			if args.show_images:
				plt.imshow((X[0][0].cpu()+epsilon[0][0][0].clamp(min=0).clamp(max=1.).detach().cpu()), cmap='gray', vmin=0, vmax=1)
		avg_upper_batch.append(uavg)
		if args.show_images:
			fig2.suptitle('average robust epsilons: ['+ str(lavg) + ', ' + str(uavg) + ']')
			plt.savefig(log_dir_path + str(i) + '_robust_eps.png')
	fig1=plt.figure(figsize=(8, 8))
	fig1.suptitle('average robust eps: ['+ str(np.average(avg_lower_batch)) + ', ' + str(np.average(avg_upper_batch)) + ']')
	fig1.add_subplot(1, 2, 1)
	if args.mode=='uniform':
		plt.imshow(np.average(eps_lower_batch)*np.ones((X[0][0].shape)), cmap='gray', vmin=0, vmax=1)
	else:
		plt.imshow((np.sum(np.array(eps_lower_batch), axis=0)/len(eps_lower_batch))[0], cmap='gray', vmin=0, vmax=1)
	fig1.add_subplot(1, 2, 2)
	if args.mode=='uniform':
		plt.imshow(np.average(eps_upper_batch)*np.ones((X[0][0].shape)), cmap='gray', vmin=0, vmax=1)
	else:
		plt.imshow((np.sum(np.array(eps_upper_batch), axis=0)/len(eps_upper_batch))[0], cmap='gray', vmin=0, vmax=1)
	plt.savefig(log_dir_path + 'average_robust_eps.png')
	plt.close()
	#hist plot
	mean_lower = ((np.sum(np.array(eps_lower_batch), axis=0)/len(eps_lower_batch))[0]).reshape(-1)
	print('check mean logic :', mean_lower==np.mean(np.array(eps_lower_batch), axis=0))
	sns_lower = sns.distplot(mean_lower)
	sns_lower.get_figure().savefig(log_dir_path + "lower_hist.png")
	mean_upper = ((np.sum(np.array(eps_upper_batch), axis=0)/len(eps_upper_batch))[0]).reshape(-1)
	print('check mean logic :', mean_upper==np.mean(np.array(eps_upper_batch), axis=0))
	sns_upper = sns.distplot(mean_upper)
	sns_upper.get_figure().savefig(log_dir_path + 'upper_hist.png')
	

def compute_boundary_and_eps(exp_run_time, args, dataset, model, model_file):
	#logging
	prefix = model + '_' + dataset
	now = datetime.datetime.now()
	log_dir_path = os.getcwd() + '/outputs/eps_boundary_results/' + prefix + exp_run_time + '/'
	try:  
		os.makedirs(log_dir_path, exist_ok=True)
	except OSError:  
		print ("Creation of the directory %s failed" % log_dir_path)
	else:  
		print ("Created train log directory %s " % log_dir_path)
	print("saving file to {}".format(log_dir_path + prefix))
	setproctitle.setproctitle(prefix)
	eps_boundary_log = open(log_dir_path + prefix.split('/')[-1] + "_eps_boundary.log", "w")
	eps_boundary_log.write(prefix)
	eps_boundary_log.write(model_file)
	eps_boundary_log.write(str(args.max_iter))
	
	#dataset
	if dataset=='mnist':
		in_dim, out_dim = 28, 10
		_, test_loader = datasets.load_dataset(dataset, args.batch_size, 0., dim=in_dim)
	elif os.path.isdir(dataset):
		in_dim, out_dim = 28, 10
		_, test_loader = datasets.load_dataset(dataset, args.batch_size, 0., dim=in_dim)
	#model
	model = models.get_model(model, dataset, Ni=in_dim, Nc=out_dim, Nic=1)
	model.load_state_dict(torch.load(model_file, map_location='cpu')['state_dict'][0])
	model = model.cuda()
	model.eval()
	#dataset, dataloader, model, args, base_pts, log, train_loader, out_dim, log_dir_path
	_,_ = compute_eps(dataset, test_loader, model, args, None, eps_boundary_log, None, in_dim, out_dim, log_dir_path)
	return #boundaries

if __name__=="__main__":
	exp_start_time = datetime.datetime.now().strftime("_%m_%d_%H%M")
	args = configs.argparser_eps_boundaries()

	dataset, model, model_file = args.datasets[0], args.models[0], args.model_files[0]
	epsilons, boundaries = [], []
	boundary = compute_boundary_and_eps(exp_start_time, args, dataset, model, model_file)