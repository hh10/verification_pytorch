""" Run 'python generate_adv_examples.py to generate adversarial examples
		against a list of models (default [conv_2layer] in models.py)
		by a list of methods (--methods, default [FGSM])
		for a list of datasets (--datasets, default [MNIST]) 
		and a list of epsilons (--epsilons, default [0, 0.5])
If using pretrained models, provide a list of model_files (--model_files) in the order as explained on Line 120, else the specified models (--models) are trained from scratch for the datasets (--datasets).
"""
import datetime
import os
import setproctitle
import json
import numpy as np
import matplotlib.pyplot as plt

#pytorch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

#our imports
import visualize
from attacks import get_attack
import configs 
import models
import datasets
from train_main import get_standard_params

DEBUG = True
'''
label2class = {
    (255, 255, 255, 255): 0,  # Street
    (0, 0, 255, 255): 1,  # Building
    (0, 255, 255, 255): 2,  # Low veg
    (0, 255, 0, 255): 3,  # Trees
    (255, 255, 0, 255): 4,  # Cars
    (127, 127, 127, 255): 5,  # Unknown
    (255, 0, 0, 255): 6,  # Water => Unknown
    (255, 20, 147, 255): 7,  # Humans => Unknown
    (81, 42, 0, 255): 8,  # Masts => Unknown
    (151, 0, 206, 255): 9  # Wires => Unknown
}
'''
label2class = {
    (255, 255, 255): 1,  # Plane
    (0, 0, 0): 0,  # Sky
}

def convert_to_one_hot(y):
	#converts labeled image to one hot
	Y = np.zeros((len(label2class), y.size()[1], y.size()[2]))
	for k,v in label2class.items():
		r = (np.floor((y[0,:,:])*100)/100 == np.floor((k[0]/255.0)*100)/100)
		g = (np.floor((y[1,:,:])*100)/100 == np.floor((k[1]/255.0)*100)/100)
		b = (np.floor((y[2,:,:])*100)/100 == np.floor((k[2]/255.0)*100)/100)
		c = (r*g*b)
		Y[v,:,:] = c
	return Y

def prepare_attack(dataset, modeln, attack, eps, model_file, args, exp_run_time, fig):
	""" Generates adversarial attacks against a trained model for a given dataset and saves the dataset (optional).
	
	#Arguments:
	- dataset: the dataset of images which have to be perturbed.
	- model: the model which has to be fooled by the attack.
	- model_file: the .pth file of the trained model.
	- eps: the max l-inf bound on the perturbation to a image.
	- args: other configs that are common for all attacks (see argparse_attack in configs.py)
	- exp_run_time: the time when this set of experiment started (only for logging purposes) 
	
	#Returns:
	- stats: {
				#NOISE independent stats
				'baseline_acc': the accuracy of the trained model on correct images (baseline), i.e., the # of images correctly classified
				'adv_acc': the accuracy of the trained model on adversarial images, i.e., the # of adversarial images correctly classified
				'adv_noise_acc': the accuracy after random noise is added to the adversarial dataset 
				'baseline_noise_acc': the accuracy after random noise is added to the clean dataset
				
				#NOISE dependent stats
				'baseline_adv_noise_corr': the # of images that were correctly classified before and after attack and addition of noise,
				'baseline_noise_corr_adv_aff': the # of adversarial images that were misclassified but were correctly classified afetr the addition of noise, 
				'baseline_adv_corr_noise_aff': the # of adversarial images that were first correctly classfified but were misclassified afetr the addition of noise,
				'baseline_corr_adv_aff_noise_help': the # of adversarial images that were misclassified, but on addition of noise, there were misclassified to some other wrong class, 
				'baseline_corr_adv_aff_noise_steady': the # of adversarial images that consistently misclassified to the same incorrect classes
			}
	"""
	dataset, _, dim, num_class, num_channels = get_standard_params(dataset, modeln, args)
	#logging (based on attack, model, dataset)
	prefix = args.prefix + '_' + attack + '_' + modeln + '_' + dataset
	log_dir_path = os.getcwd() + '/outputs/attacks/' + exp_run_time + '/' + prefix.split('/')[-1].split('.')[0] + '/'
	try: 
		os.makedirs(log_dir_path, exist_ok=True)
	except OSError:  
		print ("Creation of the main log directory %s failed, already exists" % log_dir_path)
	else:  
		print ("Created the main log directory %s " % log_dir_path)
	attack_log = open(log_dir_path + "_attack.log", "w")
	attack_log.write(args.log_data)
	attack_log.write(prefix)
	attack_log.write(model_file)
	setproctitle.setproctitle(prefix)

	#dataset
	_, test_loader = datasets.load_dataset(dataset, args.batch_size, dim=dim, segm=args.segm)
	#model
	model = models.get_model(modeln, dataset, Ni=dim, Nc=num_class, Nic=num_channels)
	model.load_state_dict(torch.load(model_file, map_location='cpu')['state_dict'][0])
	model.eval()
	#training
	kwargs = {'parallel':(args.cuda_ids is not None)}

	if args.make_dataset==True and eps!=0:
		dataset_path = log_dir_path+'dataset'+'_'+str(eps)+'/'
		try:  
			os.mkdir(dataset_path)
		except OSError:  
			print ("Creation of this dataset directory %s failed" % dataset_path)
		else:  
			print ("Created attack images dataset directory %s " % dataset_path)
	else:
		dataset_path = None
	
	adv_exs = []
	stop_iter = 500 if DEBUG else len(test_loader)
	N = stop_iter*args.batch_size
	stats = {'baseline_acc':N, 'adv_acc':N,
				'adv_noise_acc':[N]*len(args.attack_noise), 'baseline_noise_acc':[0]*len(args.attack_noise),
				'baseline_adv_noise_corr':[0]*len(args.attack_noise),
				'baseline_noise_corr_adv_aff':[0]*len(args.attack_noise),
				'baseline_adv_corr_noise_aff':[0]*len(args.attack_noise),
				'baseline_corr_adv_aff_noise_help':[0]*len(args.attack_noise), 
				'baseline_corr_adv_aff_noise_steady':[0]*len(args.attack_noise) }
	
	if not args.segm:
		for i,(X,y) in enumerate(test_loader):
			X,y = X.cuda(), y.cuda().long()
			if y.dim() == 2: 
				y = y.squeeze(1)
			X.requires_grad = True
			err, err_atk, err_noise_atk, _ = get_attack(attack, model, X, y, eps, 'CE', stats, adv_exs, dataset_path, args.attack_noise, norm=args.attack_norm, ball=args.attack_ball, fig=None, dataset=dataset)
			stats['baseline_acc'] = stats['baseline_acc'] - err
			stats['adv_acc'] = stats['adv_acc'] - err_atk
			for ii in range(len(args.attack_noise)):
				stats['adv_noise_acc'][ii] = stats['adv_noise_acc'][ii] - err_noise_atk[ii]
			if i==(stop_iter-1):
				break
	else:
		print('segm')
		for i, (X,y) in enumerate(zip(test_loader[0], test_loader[1])):
			adv_y = torch.from_numpy(np.zeros(y.shape).astype('float32'))
			visualize.show_data(fig, X, y, adv_y)
			Y, adv_Y = convert_to_one_hot(y), convert_to_one_hot(adv_y)
			X, Y, adv_Y = X.unsqueeze(0).cuda(), torch.from_numpy(Y).unsqueeze(0).cuda().float(), torch.from_numpy(adv_Y).unsqueeze(0).cuda().float()
			err, err_atk, _, _ = get_attack(attack, model, X, Y, eps, save_path=dataset_path, adv_y=adv_Y, segm=True, fig=fig, dataset=dataset)
			if i==(stop_iter-1):
				break
	stats['baseline_acc'] = stats['baseline_acc']/float(N)
	stats['adv_acc'] = stats['adv_acc']/float(N)
	if args.segm == False:
		stats['adv_noise_acc'] = [l/float(N) for l in stats['adv_noise_acc'] ]
		#for l in stats['baseline_noise_acc']:
		#	print('see            ', l, float(N))
		stats['baseline_noise_acc'] = [l/float(N) for l in stats['baseline_noise_acc'] ]	
	return stats, adv_exs, log_dir_path, attack_log
		
	
if __name__ == '__main__':
	#logging
	exp_run_time = datetime.datetime.now().strftime("%m_%d_%H%M")
	args = configs.argparser_attack()
	if not args.show:
		fig=plt.figure(figsize=(9, 9))
		fig.suptitle('Attacks')
	else:
		fig=None
	
	dataset_list, model_list, attack_list, epsilon_list = args.datasets, args.models, args.attacks, args.epsilons
	
	model_file_dict = {}
	if args.model_files == []:
		combinations = [[d,m] for d in dataset_list for m in model_list]
		for combo in combinations:
			model_path = train_session(*combo, 'baseline', configs.argparser_train(), exp_run_time)
			model_file_dict[combo[0] + '_' + combo[1]] = model_path
	else:
		# if providing a list of files, then (#_of_datasets x #_of_models) model_files are epected in the order as explained below:
		# --datasets d1 d2 --models m1 m2 m3 --model_files file_d1m1 file_d1m2 file_d1m3 file_d2m1 file_d2m2 file_d2m3 
		i = 0
		for di in range(len(dataset_list)):
			for mi in range(len(model_list)):
				model_file_dict[dataset_list[di] + '_' + model_list[mi]] = args.model_files[i]
				i = i + 1
		
	combinations = [[d,m,a] for d in dataset_list for m in model_list for a in attack_list]
	for combo in combinations:
		stats_eps, adv_examples = [], []
		print(epsilon_list)
		for e in epsilon_list:
			print('Processing {}'.format(combo))
			stats, adv_exs, savepath, attack_log = prepare_attack(*combo, e, model_file_dict[combo[0] + '_' + combo[1]], args, exp_run_time, fig)
			print(stats)
			attack_log.write(json.dumps(stats))
			stats_eps.append(stats)
			adv_examples.append(adv_exs)
			
			
			
		
		
		#plotting and visualization part
		baseline_noise_acc = []
		for i in range(len(args.attack_noise)):
			noise=[]
			for l in stats_eps:
				noise.append(l['baseline_noise_acc'][i])
			baseline_noise_acc.append(noise)
		adv_noise_acc = []
		for i in range(len(args.attack_noise)):
			noise=[]
			for l in stats_eps:
				noise.append(l['adv_noise_acc'][i])
			adv_noise_acc.append(noise)
		Y = [ [l['baseline_acc'] for l in stats_eps], [l['adv_acc'] for l in stats_eps], *[l for l in adv_noise_acc], *[l for l in baseline_noise_acc] ]
		legends = ["baseline_acc", "adv_acc", *['adv_noise_{}_acc'.format(n) for n in args.attack_noise], *['baseline_noise_{}_accs'.format(n) for n in args.attack_noise] ]
		visualize.make_2Dplot("Accuracies vs Epsilon", epsilon_list, Y, legends, ["Epsilon","Accuracies"], savepath)

		baseline_corr_adv_aff_noise_help = []
		for i in range(len(args.attack_noise)):
			noise=[]
			for l in stats_eps:
				noise.append(l['baseline_corr_adv_aff_noise_help'][i])
			baseline_corr_adv_aff_noise_help.append(noise)
		baseline_corr_adv_aff_noise_steady = []
		for i in range(len(args.attack_noise)):
			noise=[]
			for l in stats_eps:
				noise.append(l['baseline_corr_adv_aff_noise_steady'][i])
			baseline_corr_adv_aff_noise_steady.append(noise)
		Y = [ *[l for l in baseline_corr_adv_aff_noise_help], *[l for l in baseline_corr_adv_aff_noise_steady] ]
		legends = [*['BNA_{}'.format(n) for n in args.attack_noise], *['BNS_{}'.format(n) for n in args.attack_noise] ]
		visualize.make_2Dplot("Misclassifications vs Epsilon", epsilon_list, Y, legends, ["Epsilon","Misclassifications"], savepath)
		visualize.show_results(epsilon_list, adv_examples, savepath)




'''
USAGE:
python generate_adv_examples.py --datasets mnist --models conv_2layer --model_files <MODEL_FILE .pth>
python generate_adv_examples.py --datasets mnist --models conv_2layer --model_files <MODEL_FILE .pth> --make_dataset True --attacks PGD --attack_norm wasserstein --attack_ball wasserstein
'''
