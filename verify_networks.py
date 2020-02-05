import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
import torchvision.transforms as transforms

from sklearn.decomposition import PCA
import random   
import setproctitle
import datetime

import visualize
import configs
import models
import datasets
import attacks
from trainer import *
from common_utils import get_standard_params

import math
import os
import numpy as np
import json

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
random.seed(0)
np.random.seed(0)
DEBUG = True

def try_to_fetch_attack(attack, dataset, model, test_loader, epsilon, attack_noise):
	print("here try_to_get_attack")
	stop_iter = 100 if DEBUG else len(test_loader)
	N = stop_iter*args.batch_size
	adv_exs = []
	stats = {'baseline_acc':N, 'adv_acc':N,
						'adv_noise_acc':[N]*len(args.attack_noise), 'baseline_noise_acc':[0]*len(args.attack_noise),
						'baseline_adv_noise_corr':[0]*len(args.attack_noise),
						'baseline_noise_corr_adv_aff':[0]*len(args.attack_noise),
						'baseline_adv_corr_noise_aff':[0]*len(args.attack_noise),
						'baseline_corr_adv_aff_noise_help':[0]*len(args.attack_noise), 
						'baseline_corr_adv_aff_noise_steady':[0]*len(args.attack_noise) }
	for ii, (X,y) in enumerate(test_loader):
		X,y = X.cuda(), y.cuda().long()
		if y.dim() == 2: 
			y = y.squeeze(1)
		X.requires_grad = True
		fig = None  #plt.figure()
		err, err_atk, err_noise_atk, _ = attacks.get_attack(attack, model, X, y, epsilon, 'CE', stats, adv_exs, noise = attack_noise, dataset=dataset, fig=fig)
		stats['baseline_acc'] = stats['baseline_acc'] - err
		stats['adv_acc'] = stats['adv_acc'] - err_atk
		for i in range(len(args.attack_noise)):
			stats['adv_noise_acc'][i] = stats['adv_noise_acc'][i] - err_noise_atk[i]
		if ii==(stop_iter-1):
			break
	stats['baseline_acc'] = stats['baseline_acc']/float(N)
	stats['adv_acc'] = stats['adv_acc']/float(N)
	stats['adv_noise_acc'] = [l/float(N) for l in stats['adv_noise_acc'] ]
	stats['baseline_noise_acc'] = [l/ii for l in stats['baseline_noise_acc'] ]	
	return stats, adv_exs
		

def evaluate_network(exp_run_time, args, dataset, modeln, method, model_file, epsilon):
	dataset, _, dim, num_class, num_channels = get_standard_params(dataset, modeln, args, args.pretrained)
	#logging
	prefix = modeln + '_' + dataset + '_' + method
	now = datetime.datetime.now()
	log_dir_path = os.getcwd() + '/outputs/ver_results/' + exp_run_time + '/' + prefix + '/'
	try:  
		os.makedirs(log_dir_path, exist_ok=True)
	except OSError:  
		print ("Creation of the directory %s failed" % log_dir_path)
	else:  
		print ("Created train log directory %s " % log_dir_path)
	print("saving file to {}".format(log_dir_path + prefix))
	setproctitle.setproctitle(prefix)
	ver_log = open(log_dir_path + "verification.log", "w")
	ver_log.write(args.log_data)
	ver_log.write(prefix)
	ver_log.write(str(epsilon))
	if model_file is not None:
		ver_log.write(model_file)
	#dataset
	train_loader, test_loader = datasets.load_dataset(dataset, batch_size=args.train_batch_size, dim=dim, test_batch_size=args.batch_size, num_channels=num_channels)
	#model
	model = models.get_model(modeln, dataset, Ni=dim, Nc=num_class, Nic=num_channels, pretrained=args.pretrained)
	if model_file is not None:
		model.load_state_dict(torch.load(model_file, map_location='cpu')['state_dict'][0]) #[0]
	model.eval()
	
	best_err = 1
	criterion = nn.CrossEntropyLoss()
	torch.set_grad_enabled(False)
	fig = plt.figure(figsize=(15,15))
	
	# verifier output (error upper bound)
	err, robust_err, robust_err_pca, lb, ub, lb_pca, ub_pca = evaluate_verifier(test_loader, [method], [model], criterion, 0, float(epsilon), [ver_log], num_class=num_class,
     		  norm_type=args.norm, bounded_input=False, proj=args.proj, pca_var=args.pca_var, DEBUG=DEBUG, save_path=log_dir_path, pretrained=args.pretrained, fig=fig, pca=None, embed_train=None, X_train=None, y_train=None, phase=args.phase, intermediate=args.intermediate)

	# emprirical check (error lower bound)
	stats, adv_exs = [], []
	for attack in args.attacks:
		s, a = try_to_fetch_attack(attack, dataset, model, test_loader, float(epsilon), args.attack_noise)
	stats.append(s)
	adv_exs.append(a)
	return err, robust_err, robust_err_pca, log_dir_path, stats, adv_exs, ver_log#, np.average(np.sum(ub-lb))


if __name__ == "__main__": 
	exp_run_time = datetime.datetime.now().strftime("%m_%d_%H%M")
	args = configs.argparser_evaluate()
  
	dataset_list, model_list, model_file_list, epsilon_list, method_list = args.datasets, args.models, args.model_files, args.epsilons, args.methods
	model_file_dict = {}
	# for a list of files provided, (#_of_datasets x #_of_models) model_files are epected in the order as explained below:
	# --datasets d1 d2 --models m1 m2 m3 --model_files file_d1m1 file_d1m2 file_d1m3 file_d2m1 file_d2m2 file_d2m3 
	i = 0
	for di in range(len(dataset_list)):
		for mi in range(len(model_list)):
			model_file_dict[dataset_list[di] + '_' + model_list[mi]] = args.model_files[i] if args.model_files != [] else None
			i = i + 1

	combinations = [[d,m,md] for d in dataset_list for m in model_list for md in method_list]
	for combo in combinations:
		errs, robust_errs, robust_errs_pca, stats_eps, adv_exs_eps = [], [], [], [], []
		for e in epsilon_list:
			err, robust_err, robust_err_pca, savepath, stats, adv_exs, ver_log = evaluate_network(exp_run_time, args, *combo, model_file_dict[combo[0] + '_' + combo[1]], e)
			errs.append([e.avg for e in err])
			robust_errs.append([re.avg for re in robust_err])
			if args.pca_var<1.:
				robust_errs_pca.append([repca.avg for repca in robust_err_pca])
			ver_log.write(json.dumps(stats))
			stats_eps.append(stats)
			adv_exs_eps.append(adv_exs)
		print(errs, file=ver_log)
		print(robust_errs, file=ver_log)
		print(robust_errs_pca, file=ver_log)
		
		#plotting and visualization part
		if args.pca_var<1.:
			Y = [ [e for e in errs], [re for re in robust_errs], [repca for repca in robust_errs_pca]]
			legends = ["baseline_err", "robust_err", "robust_err_pca"]
		else:
			Y = [ [e for e in errs], [re for re in robust_errs] ]
			legends = ["baseline_err", "robust_err"]
		print(Y, legends)
		visualize.make_2Dplot("Robust_error vs Epsilon", epsilon_list, Y, legends, ["Epsilon","Errors"], savepath) #[errs]
		for i in range(len(args.attacks)):
			baseline_noise_acc = []
			for j in range(len(args.attack_noise)):
				noise=[]
				for l in stats_eps:
					noise.append(l[i]['baseline_noise_acc'][j])
				baseline_noise_acc.append(noise)
			print(baseline_noise_acc)
			adv_noise_acc = []
			for j in range(len(args.attack_noise)):
				noise=[]
				for l in stats_eps:
					noise.append(l[i]['adv_noise_acc'][j])
				adv_noise_acc.append(noise)
			print(adv_noise_acc)

			Y = [ [l[i]['baseline_acc'] for l in stats_eps], [l[i]['adv_acc'] for l in stats_eps], *[l for l in adv_noise_acc], *[l for l in baseline_noise_acc] ]
			legends = ["baseline_acc", "adv_acc", *['adv_noise_{}_acc'.format(n) for n in args.attack_noise], *['baseline_noise_{}_accs'.format(n) for n in args.attack_noise] ]
			visualize.make_2Dplot("Accuracies vs Epsilon{}".format(args.attack_noise[i]), epsilon_list, Y, legends, ["Epsilon","Accuracies"], savepath)

			baseline_corr_adv_aff_noise_help = []
			for j in range(len(args.attack_noise)):
				noise=[]
				for l in stats_eps:
					noise.append(l[i]['baseline_corr_adv_aff_noise_help'][j])
				baseline_corr_adv_aff_noise_help.append(noise)
			baseline_corr_adv_aff_noise_steady = []
			for j in range(len(args.attack_noise)):
				noise=[]
				for l in stats_eps:
					noise.append(l[i]['baseline_corr_adv_aff_noise_steady'][j])
				baseline_corr_adv_aff_noise_steady.append(noise)
			
			Y = [ *[l for l in baseline_corr_adv_aff_noise_help], *[l for l in baseline_corr_adv_aff_noise_steady] ]
			legends = [*['BNA_{}'.format(n) for n in args.attack_noise], *['BNS_{}'.format(n) for n in args.attack_noise] ]
			print(Y, legends)
			visualize.make_2Dplot("Misclassifications vs Epsilon{}".format(args.attack_noise[i]), epsilon_list, Y, legends, ["Epsilon","Misclassifications"], savepath)
			visualize.show_results(epsilon_list, adv_exs_eps[i], savepath)