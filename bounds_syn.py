#file done finding robut bounds in synthetic data 2D #modes = ['uniform', 'covariance', semi_nonuniform', nonuniform']
import os
import datetime
import setproctitle
import argparse
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import autograd

import configs 
import models
import datasets
#from visualize import plot_update, plot_clear_update
from convex_adversarial import get_RobustBounds as robust_bounds_wong
from convex_adversarial import robust_loss as robust_loss_wong

def plot_update(plt, X, epsilon, A, color, useA=False):
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
	if useA:
		e = np.array([[-epsilon_l[0][0],-epsilon_l[0][0]], [-epsilon_l[0][1],epsilon_u[0][1]]])*np.transpose(A)
		plt.plot([data[0]+e[0,0],data[0]+e[0,1]] , [data[1]+e[1,0],data[1]+e[1,1]], color = color)
		e = np.array([[epsilon_u[0][0],epsilon_u[0][0]], [-epsilon_l[0][1],epsilon_u[0][1]]])*np.transpose(A)
		plt.plot([data[0]+e[0,0],data[0]+e[0,1]] , [data[1]+e[1,0],data[1]+e[1,1]], color = color)
		e = np.array([[-epsilon_l[0][0],epsilon_u[0][0]], [-epsilon_l[0][1],-epsilon_l[0][1]]])*np.transpose(A)
		plt.plot([data[0]+e[0,0],data[0]+e[0,1]] , [data[1]+e[1,0],data[1]+e[1,1]], color = color)
		e = np.array([[epsilon_u[0][0],-epsilon_l[0][0]], [epsilon_u[0][1],-epsilon_l[0][1]]])*np.transpose(A)
		plt.plot([data[0]+e[0,0],data[0]+e[0,1]] , [data[1]+e[1,0],data[1]+e[1,1]], color = color)		
	else:
		e = np.array([[-epsilon_l[0][0],-epsilon_l[0][0]], [-epsilon_l[0][1],epsilon_u[0][1]]])
		plt.plot([data[0]+e[0,0],data[0]+e[0,1]] , [data[1]+e[1,0],data[1]+e[1,1]], color = color)
		e = np.array([[epsilon_u[0][0],epsilon_u[0][0]], [-epsilon_l[0][1],epsilon_u[0][1]]])
		plt.plot([data[0]+e[0,0],data[0]+e[0,1]] , [data[1]+e[1,0],data[1]+e[1,1]], color = color)
		e = np.array([[-epsilon_l[0][0],epsilon_u[0][0]], [-epsilon_l[0][1],-epsilon_l[0][1]]])
		plt.plot([data[0]+e[0,0],data[0]+e[0,1]] , [data[1]+e[1,0],data[1]+e[1,1]], color = color)
		e = np.array([[-epsilon_l[0][0],epsilon_u[0][0]], [epsilon_u[0][1],epsilon_u[0][1]]])
		plt.plot([data[0]+e[0,0],data[0]+e[0,1]] , [data[1]+e[1,0],data[1]+e[1,1]], color = color)		
	plt.draw()
	pass

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
	pass

def get_eps(args):
	if args.mode == 'uniform':
		eps = torch.zeros([args.batch_size, 1], device = "cuda", requires_grad = True)
		eps.data.fill_(np.sqrt(args.init_margin))
		epsilon = [eps,]
	elif args.mode in ['semi_nonuniform', 'covariance']:
		eps = torch.zeros([args.batch_size, args.in_dim], device = "cuda", requires_grad = True)
		eps.data.fill_(np.sqrt(args.init_margin))
		epsilon = [eps,]
	elif args.mode == 'nonuniform':
		#print('nonuniform eps')
		eps = torch.zeros([args.batch_size, args.in_dim], device = "cuda")
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
		#print('nonuniform optim')
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

def compute_eps(dataset, dataloader, model, args, base_pts, log, train_loader, log_dir_path):
	data_batch, label_batch, result_mask_batch, eps_lower_batch, eps_upper_batch, A_batch, Volume_batch, avg_lower_batch, avg_upper_batch = [], [], [], [], [], [], [], [], []	
	plot_clear_update([dataloader, train_loader])

	for i, (X,y) in enumerate(dataloader):
		X,y = X.cuda(), y.cuda().long()
		if y.dim() == 2: 
			y = y.squeeze(1)
		plt.scatter(X[0][0].cpu(), X[0][1].cpu(), s = 3, color = 'k')
		plt.text(x = X[0][0].cpu() + 1e-2, y = X[0][1].cpu() - 6e-2, s = str(y[0].cpu().item()), color = 'g', fontsize = 6)
		plt.draw()
		plt.pause(0.01)
		print('press a key to start')
		if i ==0:
			while input()==0:
				pass
		y_hat = model(Variable(X))
		corr_pred_mask = (y_hat.max(1)[1] == y).float()
		#one-hot label mask
		label_mask = torch.ones([args.batch_size, args.out_dim], device="cuda")
		label_mask = label_mask.scatter_(dim = 1, index = y.view(args.batch_size, 1), value = 0)
		
		A = torch.eye(args.in_dim, requires_grad = False, device="cuda")
		An = Variable(A/torch.norm(A), requires_grad=True)
		epsilon = get_eps(args)
		optims = get_optimiser(args, epsilon, An)

		#dual and penalty vars
		rho = args.beta
		grad_clip = args.grad_clip
		lam = torch.zeros([args.batch_size, args.out_dim], requires_grad=False, device = "cuda")
		
		if args.mode=='nonuniform':
			index = 2
		else:
			index = 1			
		#for ind in range(index):
		#	for iter_idx in range(args.max_iter):
		for iter_idx in range(args.max_iter):
			for ind in range(index):
				lb, ub = get_bounds(model, epsilon, X, y, 'l1', args.mode, An)
				low_c = lb.gather(1, y.view(-1, 1))
				v = low_c - ub - args.delta
				err = torch.min(v, - lam / rho) * label_mask
				eps_loss = get_objective(epsilon, An, args.mode)	
				err_loss = torch.sum(lam * err, dim = 1) + rho / 2. * torch.norm(err, dim = 1) ** 2
				loss = torch.sum((eps_loss + err_loss) * corr_pred_mask)/torch.sum(corr_pred_mask)

				if args.mode == 'covariance' and torch.any(v < 0):
					lb, ub = get_bounds(model, epsilon, X, y, 'l1', args.mode, An)
					low_c = lb.gather(1, y.view(-1, 1))
					v = low_c - ub - args.delta
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
				if args.mode=='nonuniform':
					plot_update(plt, X, [epsilon[0], epsilon[1]], An.data.cpu().numpy(), 'orange')		
				elif args.mode=='uniform':
					a = torch.ones([args.batch_size, args.in_dim]).cuda()
					plot_update(plt, X, [epsilon[0]*a, epsilon[0]*a], An.data.cpu().numpy(), 'pink')							
					del a
				elif args.mode=='semi_nonuniform':
					plot_update(plt, X, [epsilon[0], epsilon[0]], An.data.cpu().numpy(), 'green')					
				elif args.mode=='covariance':
					a = torch.ones([args.batch_size, args.in_dim]).cuda()
					plot_update(plt, X, [epsilon[0], epsilon[0]], An.data.cpu().numpy(), 'purple', True)					
					del a
				plt.pause(0.01)
				if iter_idx%10 == 0:
					plot_clear_update([dataloader, train_loader])
				#eps_iter = eps_iter+1
				if iter_idx==args.max_iter:
					print('reached max_iter', iter_idx)
		shrink_times = 0
		while shrink_times < 100:
			for ind in range(index):			
				lb, ub = get_bounds(model, epsilon, X, y, 'l1', args.mode, An)
				low_c = lb.gather(1, y.view(-1, 1))
				v = low_c - ub - args.delta
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
		avg_lower_batch.append(np.average(np.average(epsilon[0].clamp(min=0).clamp(max=1.).detach().cpu().numpy(), axis=0)))
		if args.mode=='nonuniform':
			eps_upper_batch.extend(epsilon[1].clamp(min=0).clamp(max=1.).detach().cpu().numpy())
			avg_upper_batch.append(np.average(np.average(epsilon[1].clamp(min=0).clamp(max=1.).detach().cpu().numpy(), axis=0)))
		else:
			eps_upper_batch.extend(epsilon[0].clamp(min=0).clamp(max=1.).detach().cpu().numpy())
			avg_upper_batch.append(np.average(np.average(epsilon[0].clamp(min=0).clamp(max=1.).detach().cpu().numpy(), axis=0)))

	plot_clear_update([dataloader, train_loader])
	counts = [0 for _ in range(args.out_dim)]
	eps_avg, eps_lone1, eps_uone1, eps_lone2, eps_uone2 = [], [], [], [], []
	for p_idx, (data, label, result, eps_l, eps_u, avg_eps_l, avg_eps_u) in enumerate(zip(data_batch, label_batch, result_mask_batch, eps_lower_batch, eps_upper_batch, avg_lower_batch, avg_upper_batch)):
		if result == 0:
			print("result")
			continue
		dis_list = [np.linalg.norm(data - base_pt) for base_pt in base_pts]
		predict = np.argmin(dis_list)
		counts[predict] += 1
		plt.scatter([data[0],], [data[1],], s = 3, color = 'k')
		plt.text(x = data[0] + 1e-2, y = data[1] + 6e-2, s = str(np.sum(counts)), color = 'b', fontsize = 5)
		plt.text(x = data[0] + 1e-2, y = data[1] - 6e-2, s = str(label), color = 'g', fontsize = 6)
		colors = {0:'red', 1:'blue', 2:'pink', 3:'brown'}
		if args.mode=='uniform':
			a = np.ones([args.batch_size, args.in_dim])
			plot_update(plt, [data], [eps_l*a, eps_u*a], A, 'pink')
			del a	
		elif args.mode=='covariance':
			a = np.ones([args.batch_size, args.in_dim])
			plot_update(plt, [data], [eps_l*a, eps_u*a], A, 'purple', True)
			del a	
		else:
			plot_update(plt, [data], [[eps_l], [eps_u]], A, 'orange')#colors[label])	
		eps_lone1.append(eps_l[0])
		eps_uone1.append(eps_u[0])
		eps_lone2.append(eps_l[1])
		eps_uone2.append(eps_u[1])
		eps_avg.append(eps_l+eps_u)
	plt.pause(0.5)		
	print('counts', counts, np.sum(eps_avg), np.average(eps_avg), np.sum(eps_lone1), np.average(eps_lone1), np.sum(eps_uone1), np.average(eps_uone1), np.sum(eps_lone2), np.average(eps_lone2), np.sum(eps_uone2), np.average(eps_uone2))
	plt.savefig(log_dir_path + 'boundaries_and_eps', bbox = 'tight', dpi = 500)
	return		


def compute_boundary_and_eps(exp_run_time, args, dataset, model, model_file):
	prefix = model+ '_' + dataset
	log_dir_path = os.getcwd() + '/outputs/eps_boundary_syn_results/' + prefix + exp_run_time + '/'
	try:
		os.makedirs(log_dir_path, exist_ok=True)
	except OSError:
		print ("Creation of the directory %s failed" % log_dir_path)
	else:  
		print ("Created train log directory %s " % log_dir_path)
	setproctitle.setproctitle(prefix)
	eps_boundary_log = open(log_dir_path + prefix.split('/')[-1] + "_eps_boundary.log", "w")
	eps_boundary_log.write(prefix)
	eps_boundary_log.write(model_file)
	eps_boundary_log.write(str(args.max_iter))
	
	#dataset
	train_loader, test_loader, base_pts = datasets.syn_data(args.batch_size, dim = args.in_dim, train_pts = args.batch_size*25, test_pts = args.batch_size*4, classes = args.out_dim, data_file = args.dataset_file)

	#model
	model = models.get_model(model, dataset, Ni=args.in_dim, Nc=args.out_dim, Nic=1)
	model.load_state_dict(torch.load(model_file, map_location='cpu')['state_dict'][0])
	model = model.cuda()
	model.eval()

	plt.xticks([-1., 0., 1.])
	plt.yticks([-1., 0., 1.])
	plt.xlim(-1., 1.)
	plt.ylim(-1., 1.)
	plt.axes().set_aspect('equal')
	compute_eps(dataset, test_loader, model, args, base_pts, eps_boundary_log, train_loader, log_dir_path)
	#plotting
	plt.show()
	return



if __name__=="__main__":
	exp_start_time = datetime.datetime.now().strftime("_%m_%d_%H%M")
	args = configs.argparser_eps_boundaries()

	dataset, model, model_file = args.datasets[0], args.models[0], args.model_files[0]
	epsilons, boundaries = [], []
	boundary = compute_boundary_and_eps(exp_start_time, args, dataset, model, model_file)	


'''
Example runs:
python bounds_syn.py --datasets syn --in_dim 2 --out_dim 2 --dataset_file outputs/data/07_15_1214 --models 2D --model_files ~/masters_thesis_notebooks/master_repo/outputs/trainings/08_20_2046/_2D_syn/v449_best.pth --batch_size 1 --max_iter 20 --lr 0.03
'''