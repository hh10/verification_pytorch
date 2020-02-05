import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

import datetime
import time
import math
import os
import setproctitle
import json
import matplotlib.pyplot as plt

import configs 
import models
import datasets

from attacks import get_attack
from trainer import AverageMeter
from train_main import get_standard_params

DEBUG = False

def do_adv_training(dataset, modeln, model_file, args, exp_run_time):
	dataset, _, dim, num_class, num_channels = get_standard_params(dataset, modeln, args)
	#logging (based on attack, model, dataset)
	prefix = args.prefix + '_' + dataset + '_' + modeln
	log_dir_path = os.getcwd() + '/outputs/adv_trainings/' + exp_run_time + '/' + prefix.split('/')[-1].split('.')[0] + '/'
	try:  
		os.makedirs(log_dir_path, exist_ok=True)
	except OSError:  
		print ("Creation of the main log directory %s failed, already exists" % log_dir_path)
	else:  
		print ("Created the main log directory %s " % log_dir_path)
	at_log = open(log_dir_path + "_adv_training.log", "w")
	at_log.write(args.log_data)
	at_log.write(prefix)
	if model_file != ' ':
		at_log.write(model_file)
	setproctitle.setproctitle(prefix)

	#dataset
	train_loader, test_loader = datasets.load_dataset(dataset, args.batch_size, dim=dim)
	dataset_path = log_dir_path+'dataset'
	#model
	model = models.get_model(modeln, dataset, Ni=dim, Nc=num_class, Nic=num_channels)
	if model_file != ' ':
		model.load_state_dict(torch.load(model_file, map_location='cpu')['state_dict'][0])
	
	if args.opt == 'adam':
		opt = optim.Adam(model.parameters(), args.lr)
	elif args.opt == 'sgd':
		opt = optim.SGD(model.parameters(), lr=args.lr, 
							momentum=args.momentum, 
							weight_decay=args.weight_decay)				
	##lr_rate
	lr_scheduler = optim.lr_scheduler.StepLR(opt, step_size=5, gamma=0.975)
	##eps_rate
	eps_schedule = [0]*args.warm_start
	eps_grow = np.linspace(args.starting_epsilon, args.epsilon, args.schedule_length)
	for e in eps_grow:
		eps_schedule.extend([e]*args.plateau_length)
	
	#training	
	adv_exs = []
	if DEBUG:
		stop_iter = 500
	else:
		stop_iter = len(test_loader) 
	N = stop_iter
	eps = args.epsilon
	stats = {'same_noise':[[0]*3,[0]*3], 
			'same_transform': [[0]*6, [0]*6], 
			'same_attack':[[0]*(2 if dataset=='mnist' else 1), [0]*(2 if dataset=='mnist' else 1)]}
	best_err = 1
	fig = plt.figure(figsize=(15, 15))
	plt_acc, plt_acc_avg, plt_loss, plt_loss_avg, plt_sparsity = [], [], [], [], []
	for t in range(args.epochs):
		lr_scheduler.step(epoch=max(t-len(eps_schedule), 0))
		if t < len(eps_schedule) and args.starting_epsilon is not None: 
			epsil = float(eps_schedule[t])
		else:
			epsil = args.epsilon
		batch_time = AverageMeter()
		data_time = AverageMeter()
		vlosses = AverageMeter()
		verrors = AverageMeter()
		
		model.train()

		end = time.time()
		for i,(X,y) in enumerate(train_loader):
			X,y = X.cuda(), y.cuda().long()
			if y.dim() == 2: 
				y = y.squeeze(1)
			data_time.update(time.time() - end)
			
			_, _, _, out_attack = get_attack('PGD', model, X, y, eps, 'CE', stats, adv_exs, dataset_path, args.attack_noise, norm='linf', ball='linf', fig=None, do_err_test=False, dataset=dataset)
			X_vars = [X]
			X_vars.append(out_attack)
			for X_var in X_vars:
				xout = model(Variable(X_var.data.cuda()))
				xce = nn.CrossEntropyLoss()(xout, y)
				xerr = (xout.data.max(1)[1] != y).float().sum()/X.size(0)
				
				opt.zero_grad()
				xce.backward()
				opt.step()
				
				batch_time.update(time.time()-end)
				end = time.time()
				vlosses.update(xce.detach().item(), X.size(0))
				verrors.update(xerr, X.size(0))
			print('\n', t, i, xce.item(), xerr.item())
			
			if args.verbose and i % args.verbose == 0: 
				print('Epoch: [{0}][{1}/{2}]\t'
					'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
					'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
					'VAR Loss {vloss.val:.4f} ({vloss.avg:.4f})\t'
					'VAR Error {verrors.val:.3f} ({verrors.avg:.3f})\t'
					.format(t, i, len(train_loader), batch_time=batch_time,
					data_time=data_time, vloss=vlosses, verrors=verrors), file=at_log)
			at_log.flush()
			if DEBUG and i==(stop_iter-1):
				break
		plt_acc.append(1-verrors.val)
		plt_acc_avg.append(1-verrors.avg)
		plt_loss.append(vlosses.val)
		plt_loss_avg.append(vlosses.avg)
		nn_weights = []
		for l in model:
			try:
				nn_weights.extend(l.weight.view(-1).data.cpu().numpy())
				print(l)
			except:
				if l != model[-1]:
					print(l, ' has no weight, continuing')
					continue
		up = np.array(nn_weights)<0.01
		be = np.array(nn_weights)>-0.01
		sparsity = np.sum((up & be)*1.)/len(nn_weights)
		plt_sparsity.append(sparsity)
		
		if verrors.avg <= best_err or t==args.epochs-1 or t%5==0:
			best_err = verrors.avg
			model_paths = log_dir_path + args.method + str(t) + "_best.pth"
			at_log.write(model_paths)
			torch.save({
				'state_dict' : [model.state_dict()], 
				'err' : best_err,
				'epoch' : t
				}, model_paths)
			checkpoint_paths = log_dir_path + args.method + "_checkpoint.pth"
			torch.save({ 
				'state_dict': [model.state_dict()],
				'err' : verrors.avg,
				'epoch' : t
				}, checkpoint_paths)
			fig = plt.gcf()
			fig.add_subplot(1, 4, 1)
			plt.title('accuracy')
			plt.plot(np.arange(len(plt_acc)), plt_acc, 'g', alpha=0.5)
			plt.plot(np.arange(len(plt_acc)), plt_acc_avg, 'g')
			fig.add_subplot(1, 4, 2)
			plt.title('loss')
			plt.plot(np.arange(len(plt_loss)), plt_loss, 'r', alpha=0.5)
			plt.plot(np.arange(len(plt_loss)), plt_loss_avg, 'r')
			fig.add_subplot(1, 4, 3)
			plt.title('sparsity')
			plt.plot(np.arange(len(plt_sparsity)), plt_sparsity, 'k')
			fig.add_subplot(1, 4, 4)
			plt.title('epsilon')
			plt.plot(np.arange(len(eps_schedule)), eps_schedule, 'k')
			plt.suptitle('results after epoch '+ str(t))
			plt.gcf().savefig(log_dir_path + '/results_' + str(t) + '.png')
			plt.pause(0.1)
			plt.close()
	return stats, adv_exs, log_dir_path, at_log


if __name__ == '__main__':
	#logging
	exp_run_time = datetime.datetime.now().strftime("%m_%d_%H%M")
	args = configs.argparser_adv_training()
	
	stats, adv_exs, savepath, at_log = do_adv_training(args.dataset, args.model, args.model_file, args, exp_run_time)
	print(stats)