import argparse
import setproctitle
import math
import numpy as np
import time
import datetime
import os
import itertools

#pytorch imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.cuda
import torchvision.transforms as transforms
from torch.autograd import Variable
from PIL import Image

#our module imports
import configs 
import models
import datasets
from trainer import *
from visualize import plot_params
from common_utils import get_standard_params, make_params, prepare_logs_checkpoints

DEBUG = True
stop_batch = 1000

def train_session(dataset, model_name, model_file, args, exp_run_time):
	""" Trains a model by the given model for a given dataset. """
	dataset, model, dim, num_class, num_channels = get_standard_params(dataset, model_name, args)
	#logging
	prefix = args.prefix + '_' + model_name + '_' + dataset
	log_dir_path = os.getcwd() + '/outputs/trainings/' + exp_run_time + '/' + prefix + '/' 
	try:  
		os.makedirs(log_dir_path, exist_ok=True)
	except OSError:  
		print ("Creation of the directory %s failed" % log_dir_path)
	else:  
		print ("Created train log directory %s " % log_dir_path)
	setproctitle.setproctitle(prefix)
	#logging and checkpoints saving part
	train_log_handles, test_log_handles, model_paths, checkpoint_paths = prepare_logs_checkpoints(args.methods, prefix, log_dir_path, args.log_data)

	#dataset
	train_loader, test_loader = datasets.load_dataset(dataset, args.batch_size, args.training_noise, dim = dim, num_channels = num_channels,
													 train_pts = 20*num_class, test_pts = 4*num_class, classes = num_class, segm=args.segm,
													 dataset_file=args.dataset_file)

	#models (if specifying too many datasets and models, it may become difficult to load all models into memory)
	model_list = [models.get_model(model_name, dataset, Ni=dim, Nc=num_class, Nic=num_channels)]*len(args.methods)
	#loading from previous model if given
	if model_file != []:
		for model in model_list:
			model.load_state_dict(torch.load(model_file, map_location='cpu')['state_dict'][0])

	#when experimenting with correlated datasets, you may want to use A \in [dim, dim] to correlate all image pixels
	if args.useA:
		A = Variable(torch.eye(dim).cuda(), requires_grad=True)
		A.retain_grad()
	else:
		A = None

	##optimizer
	opt_list = []
	for model in model_list:
		if args.useA:
			optimiser_params = itertools.chain(*[model.parameters(), [A]])
		else:
			optimiser_params = model.parameters()
		if args.opt == 'sgd':
			opt = optim.SGD(optimiser_params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
		else:
			opt = optim.Adam(optimiser_params, args.lr) #preferred
		opt_list.append(opt)
	
	##lr_rate
	lr_scheduler = optim.lr_scheduler.StepLR(opt, step_size=5, gamma=0.975)
	##eps_rate scheduling
	## TODO : instead of fixed args.epsilon, keep increasinf epsilon till possible 
	eps_schedule = [0]*args.warm_start
	eps_grow = np.linspace(args.starting_epsilon, args.epsilon, args.schedule_length)
	for e in eps_grow:
		eps_schedule.extend([e]*args.plateau_length)
	##K scheduling that decides whether to penalize more the nominal err part or robust err part
	K_schedule = [1]*args.warm_start
	K_grow = np.linspace(0.98, 0.5, args.schedule_length)
	for k in K_grow:
		K_schedule.extend([k]*args.plateau_length)
	
	#loss
	criterion = nn.CrossEntropyLoss()
	
	#paramters to track the training and convergence of training (esp. robust training)
	params_epoch_train, params_epoch_eval = make_params(args.methods)

	#if robustifying against wasserstein norm attack, then prepare the cost matrix 
	if args.norm_train == 'wasserstein':
		M = wasserstein_cost2(dim, p=2)
	else:
		M = None

	#TRAINING PART BEGINS
	fig=plt.figure(figsize=(8, 8))
	best_err, best_robust_err = [1] * len(args.methods), [1] * len(args.methods)
	plt_acc, plt_acc_avg, plt_loss, plt_rloss, plt_loss_avg, plt_rloss_avg, plt_sparsity = [[]]*len(args.methods), [[]]*len(args.methods), [[]]*len(args.methods), [[]]*len(args.methods), [[]]*len(args.methods), [[]]*len(args.methods), [[]]*len(args.methods)
	fig = plt.figure()
	for t in range(args.epochs):
		tot_train = stop_batch if (DEBUG and t>=args.warm_start) else len(train_loader)
		tot_test = int(stop_batch*0.125) if (DEBUG and t>=args.warm_start) else len(test_loader)

		lr_scheduler.step(epoch=max(t-len(eps_schedule), 0))
		if t < len(eps_schedule) and args.starting_epsilon is not None:
			eps_epoch = float(eps_schedule[t])
		else:
			eps_epoch = args.epsilon
		if t < len(K_schedule):
			K_epoch = float(K_schedule[t])
		else:
			K_epoch = 0.5
		
		#if working with non-uniform epsilons, preapre them now
		if args.nonuniform:
			if dataset in ['syn']:
				eps = torch.zeros([args.batch_size, dim], requires_grad=True, device="cuda")
			else:
				eps = torch.zeros([args.batch_size, 1, dim, dim], requires_grad=True, device="cuda")
			epsilon	= [eps.data.fill_(eps_epoch), eps.data.fill_(eps_epoch)]
		else:
			epsilon = eps_epoch	

		if not args.segm:
			params_train = train(train_loader, args.methods, model_list, opt_list, criterion, t, epsilon, train_log_handles, norm_type=args.norm_train, bounded_input=False,
							 num_class=num_class, non_uniform=args.nonuniform, A=A, pca_var=args.pca_var, proj=args.proj, fig=fig, K=K_epoch, M=M, warm_start=args.warm_start, tot=tot_train, DEBUG=DEBUG, stop_batch=stop_batch, save_path=log_dir_path, 
							 plt_acc=plt_acc, plt_acc_avg=plt_acc_avg, plt_loss=plt_loss, plt_rloss=plt_rloss, plt_loss_avg=plt_loss_avg, plt_rloss_avg=plt_rloss_avg, plt_sparsity=plt_sparsity, intermediate=args.intermediate)
			if t%4==0 or t==0 or t==args.epochs-1: 
				params_eval, errors, robust_errors, _, _ = evaluate(test_loader, args.methods, model_list, criterion, t, epsilon, test_log_handles, 
								num_class=num_class, non_uniform=args.nonuniform, norm_type=args.norm_train, A=A, pca_var=args.pca_var, proj=args.proj, tot=tot_test, DEBUG=DEBUG, stop_batch=int(stop_batch*0.125), save_path=log_dir_path)
		else:
			segm('train', train_loader, args.methods, model_list, opt_list, t, epsilon, train_log_handles, args.verbose, norm_type=args.norm_train, bounded_input=False, fig=fig)
			errors = segm('test', test_loader, args.methods, model_list, opt_list, t, epsilon, test_log_handles, args.verbose, norm_type=args.norm_train, fig=fig)
		
		# plot params
		if args.params and args.segm==False and DEBUG:
			plot_params(args.methods, params_epoch_train, params_train, t, log_dir_path, 'train')
			plot_params(args.methods, params_epoch_eval, params_eval, t, log_dir_path, 'test')
	
		for i in range(len(args.methods)):
			if errors[i].avg <= best_err[i] or robust_errors[i].avg <= best_robust_err[i] or t%8==0 or t==args.epochs-1:
				model_paths[i] = log_dir_path + args.methods[i] + str(t) + "_best.pth"
				if errors[i].avg <= best_err[i]:
					best_err[i] = errors[i].avg
					train_log_handles[i].write(model_paths[i])
				if robust_errors[i].avg <= best_robust_err[i]:
					best_robust_err[i] = robust_errors[i].avg
					test_log_handles[i].write(model_paths[i])
				torch.save({
		        'state_dict' : [model_list[i].state_dict()], 
		        'err' : best_err[i],
		        'robust_err' : best_robust_err[i],
		        'epoch' : t
		        }, model_paths[i])
				checkpoint_paths[i] = log_dir_path + args.methods[i] + "_checkpoint.pth"
				train_log_handles[i].write(checkpoint_paths[i])
				test_log_handles[i].write(checkpoint_paths[i])
				torch.save({ 
			    'state_dict': [model_list[i].state_dict()],
			    'err' : errors[i].avg,
			    'robust_err' : best_robust_err[i],
			    'epoch' : t
			    }, checkpoint_paths[i])
			if t==args.epochs-1 or t%5==0:
				fig = plt.gcf()
				plt.clf()
				fig.add_subplot(1, 3, 1)
				plt.title('accuracy')
				plt.plot(np.arange(len(plt_acc[i])), plt_acc[i], 'g', alpha=0.5)
				plt.plot(np.arange(len(plt_acc[i])), plt_acc_avg[i], 'g')
				fig.add_subplot(1, 3, 2)
				plt.title('loss')
				plt.plot(np.arange(len(plt_loss[i])), plt_loss[i], 'r', alpha=0.5)
				plt.plot(np.arange(len(plt_loss[i])), plt_loss_avg[i], 'r')
				plt.plot(np.arange(len(plt_rloss[i])), plt_rloss[i], 'b', alpha=0.5)
				plt.plot(np.arange(len(plt_rloss_avg[i])), plt_rloss_avg[i], 'b')
				fig.add_subplot(1, 3, 3)
				plt.title('sparsity')
				plt.plot(np.arange(len(plt_sparsity[i])), plt_sparsity[i], 'k')
				plt.suptitle('results after epoch '+ str(t))
				plt.gcf().savefig(log_dir_path + '/results_' + str(t) + '.png')
				plt.pause(0.1)
				plt.close()

	return model_paths[0]
	
if __name__ == "__main__":
	#logging
	exp_run_time = datetime.datetime.now().strftime("%m_%d_%H%M")
	args = configs.argparser_train()
  
	dataset_list, model_list = args.datasets, args.models
	model_file_dict = {}
	if args.model_files != []:
		# if providing a list of files, then (#_of_datasets x #_of_models) model_files are epected in the order as explained below:
		# --datasets d1 d2 --models m1 m2 m3 --model_files file_d1m1 file_d1m2 file_d1m3 file_d2m1 file_d2m2 file_d2m3 
		i = 0
		for di in range(len(dataset_list)):
			for mi in range(len(model_list)):
				model_file_dict[dataset_list[di] + '_' + model_list[mi]] = args.model_files[i]
				i = i + 1
	
	combinations = [[d,m] for d in dataset_list for m in model_list]
	for combo in combinations:
		if args.model_files != []:
			_ = train_session(*combo, model_file_dict[combo[0] + '_' + combo[1]], args, exp_run_time)
		else:
			_ = train_session(*combo, [], args, exp_run_time)