import argparse
import numpy as np

def argparser_train(batch_size=4, epochs=50, seed=0, verbose=300, lr=5e-4, #5e-2
              epsilon=0.2, starting_epsilon=0.0005, 
              proj=None, 
              norm_train='l1', norm_test='l1', 
              opt='sgd', momentum=0.9, weight_decay=1e-4): 
	parser = argparse.ArgumentParser()
	parser.add_argument('--segm', type=bool, default=False)
	parser.add_argument('--params', type=bool, default=False)
	parser.add_argument('--useA', type=bool, default=False)
	parser.add_argument('--specific_train', type=str, default='augmentation', help='None')
	parser.add_argument('--intermediate', type=bool, default=False)
	#dataset
	parser.add_argument('--datasets', '--dlist', nargs='+', default=['mnist'], help='dataset can be {mnist, fmnist, cifar or specify a data_dir for custom dataset}')
	parser.add_argument('--num_class', type=int, default=2)
	parser.add_argument('--training_noise', type=float, default=0.000)
	parser.add_argument('--dataset_file', type = str, default=None, help='dataset file pickle to load')
	parser.add_argument('--image_size', type=int, default=512)
	# optimizer settings
	parser.add_argument('--opt', default=opt)
	parser.add_argument('--momentum', type=float, default=momentum)
	parser.add_argument('--weight_decay', type=float, default=weight_decay)
	parser.add_argument('--batch_size', type=int, default=batch_size)
	parser.add_argument('--epochs', type=int, default=epochs)
	parser.add_argument("--lr", type=float, default=lr)
	# epsilon settings
	parser.add_argument("--epsilon", type=float, default=epsilon)
	parser.add_argument("--nonuniform", type=bool, default=False)
	parser.add_argument("--starting_epsilon", type=float, default=starting_epsilon)
	parser.add_argument('--schedule_length', type=int, default=10)
	parser.add_argument('--warm_start', type=int, default=1)
	parser.add_argument('--plateau_length', type=int, default=1)
	# projection settings
	parser.add_argument('--proj', type=int, default=proj)
	parser.add_argument('--pca_var', type=float, default=1.)
	parser.add_argument('--norm_train', type=str, default='l1', help='can be {l1, l2, wasserstein}')
	# model arguments
	parser.add_argument('--models', '--mlist', nargs='+', default=['conv_2layer'], help='model can be {conv_2layer, conv_4layer, conv_wide, conv_deep, resnet18, resnet34}')
	parser.add_argument('--model_files', '--mflist', nargs='+', default=[], help='model file to finetune or train more')
	parser.add_argument('--model_expansion', type=int, default=1)
	parser.add_argument('--methods', '--mdlist', nargs='+', default=['wong'], help='method of training can be {baseline, madry, wong}')
	# other arguments
	parser.add_argument('--prefix', type=str, default='')
	parser.add_argument('--log_data', type=str, default='')
	parser.add_argument('--load')
	parser.add_argument('--real_time', action='store_true')
	parser.add_argument('--seed', type=int, default=seed)
	parser.add_argument('--verbose', type=int, default=verbose)
	parser.add_argument('--cuda_ids', default=None)
	parser.add_argument('--pretrained', type=bool, default=False)
	#attack
	parser.add_argument('--attacks', '--alist', nargs='+', default=['FGSM'], help='attacks can be {FGSM, PGD}')
	parser.add_argument('--target_class', type=int, default=None)
	# output arguments
	parser.add_argument('--make_dataset', type=bool, default=False)

	args = parser.parse_args()
	if args.starting_epsilon is None:
		args.starting_epsilon = args.epsilon 
	banned = ['verbose', 'prefix',
						'resume', 'baseline', 'eval', 
						'method', 'model', 'cuda_ids', 'load', 'real_time']
	for arg in sorted(vars(args)): 
		if arg not in banned and getattr(args,arg) is not None: 
			args.log_data += '_' + arg + '_' +str(getattr(args, arg)) + '\n'
	if args.schedule_length > args.epochs: 
		raise ValueError('Schedule length for epsilon ({}) is greater than '
										'number of epochs ({})'.format(args.schedule_length, args.epochs))

	if args.cuda_ids is not None: 
		print('Setting CUDA_VISIBLE_DEVICES to {}'.format(args.cuda_ids))
		os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_ids

	return args


def argparser_evaluate(epsilons=[0, 0.05, 0.1, 0.15, 0.2, 0.3], norm='l1'): 
	parser = argparse.ArgumentParser()
	parser.add_argument('--segm', type=bool, default=False)
	parser.add_argument('--specific_verify', type=str, default='augmentation', help='None')
	parser.add_argument('--Ni', type=int, default=16, help='Ni^(2/dim_z) must be an integer')
	parser.add_argument('--dim_z', type=int, default=2, help='keep 2 or 3')
	parser.add_argument('--phase', type=int, default=1, help='keep 1 or 2')
	parser.add_argument('--intermediate', type=bool, default=False)
	
	parser.add_argument('--encoder_file', type=str, default=None)
	parser.add_argument('--decoder_file', type=str, default=None)
	
	#attack
	parser.add_argument('--attacks', '--alist', nargs='+', default=['PGD'], help='attacks can be {FGSM, PGD, IFGSM}')
	parser.add_argument('--target_class', type=int, default=None)
	parser.add_argument('--attack_noise', '--nlist', nargs='+', default=[0.001])

	parser.add_argument('--proj', type=int, default=None)
	parser.add_argument('--pca_var', type=float, default=1.)
	parser.add_argument('--norm', default=norm)
	parser.add_argument('--models', '--mlist', nargs='+', default=['conv_2layer'])
	parser.add_argument('--pretrained', type=bool, default=False)
	parser.add_argument('--image_size', type=int, default=512)
	parser.add_argument('--num_class', type=int, default=2)
	parser.add_argument('--load')
	parser.add_argument('--output')
	parser.add_argument('--batch_size', type=int, default=1)
	parser.add_argument('--train_batch_size', type=int, default=900)
	# epsilon settings
	parser.add_argument('--epsilons', '--elist', type=float, nargs='+', default=epsilons)
	parser.add_argument('--real_time', action='store_true')
	parser.add_argument('--datasets', '--dlist', nargs='+', default=['mnist'], help='dataset can be {mnist, fmnist, cifar or specify a data_dir for custom dataset}')
	parser.add_argument('--methods', '--mdlist', nargs='+', default=['wong'], help='method of training can be {baseline, madry, wong}')
	parser.add_argument('--model_files', '--mplist', nargs='+', default=[])
	parser.add_argument('--verbose', type=int, default=True)
	parser.add_argument('--cuda_ids', default=None)
	# other arguments
	parser.add_argument('--prefix', type=str, default='')
	parser.add_argument('--log_data', type=str, default='')

	args = parser.parse_args()
	banned = ['verbose', 'prefix',
						'resume', 'baseline', 'eval', 
						'method', 'model', 'cuda_ids', 'load', 'real_time']
	for arg in sorted(vars(args)): 
		if arg not in banned and getattr(args,arg) is not None: 
			args.log_data += '_' + arg + '_' +str(getattr(args, arg))

	if args.cuda_ids is not None: 
		print('Setting CUDA_VISIBLE_DEVICES to {}'.format(args.cuda_ids))
		os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_ids
	return args
	
	
#attacks
def argparser_attack(attack_method='FGSM', model=['conv_2layer'], dataset='mnist', epsilons=[0, .05, .1, .15, .2, .25, .3], target_class=None, use_cuda=True, batch_size=1, epochs=10, seed=0, verbose=300, lr=1e-3, opt='adam', momentum=0.9, weight_decay=5e-4, starting_epsilon=0.01, show=False): #, .1, .15, .2, .25, .3
	parser = argparse.ArgumentParser()
	parser.add_argument('--segm', type=bool, default=False)
	parser.add_argument('--show', type=bool, default=False)
	parser.add_argument('--e_threshold', type=int, default=1000)

	#attack
	parser.add_argument('--attacks', '--alist', nargs='+', default=['FGSM'], help='attacks can be {FGSM, PGD, IFGSM}')
	parser.add_argument('--target_class', type=int, default=None)
	parser.add_argument('--attack_noise', '--nlist', nargs='+', default=[0.001])
	parser.add_argument('--attack_norm', type=str, default='linf')
	parser.add_argument('--attack_ball', type=str, default='linf')
	#dataset
	parser.add_argument('--datasets', '--dlist', nargs='+', default=['mnist'], help='dataset can be {mnist, fmnist, cifar or specify a data_dir for custom dataset}')
	parser.add_argument('--num_class', type=int, default=2)
	# optimizer settings
	parser.add_argument('--opt', default=opt)
	parser.add_argument('--momentum', type=float, default=momentum)
	parser.add_argument('--weight_decay', type=float, default=weight_decay)
	parser.add_argument('--batch_size', type=int, default=batch_size)
	parser.add_argument('--epochs', type=int, default=epochs)
	parser.add_argument("--lr", type=float, default=lr)
	# epsilon settings
	parser.add_argument('--epsilons', '--elist', type=float, nargs='+', default=epsilons)
	# model arguments
	parser.add_argument('--models', '--mlist', nargs='+', default=['conv_2layer'], help='model can be {conv_2layer, conv_4layer, conv_wide, conv_deep, resnet18, resnet34}')
	parser.add_argument('--model_expansion', type=int, default=1)
	parser.add_argument('--model_files', '--mplist', nargs='+', default=[])
	parser.add_argument('--model_factor', type=int, default=8)
	#attack method
	parser.add_argument('--methods', '--mdlist', nargs='+', default=['wong'], help='method of training can be {baseline, madry, wong}')
	# other arguments
	parser.add_argument('--prefix', type=str, default='')
	parser.add_argument('--log_data', type=str, default='')
	parser.add_argument('--real_time', action='store_true')
	parser.add_argument('--seed', type=int, default=seed)
	parser.add_argument('--verbose', type=int, default=verbose)
	parser.add_argument('--cuda_ids', default=None)
	# output arguments
	parser.add_argument('--make_dataset', type=bool, default=False)
		
	args = parser.parse_args()
	banned = ['verbose', 'prefix',
						'resume', 'baseline', 'eval', 
						'method', 'model', 'cuda_ids', 'load', 'real_time']
	for arg in sorted(vars(args)): 
		if arg not in banned and getattr(args,arg) is not None: 
			args.log_data += '_' + arg + '_' +str(getattr(args, arg)) + '\n'
	if args.cuda_ids is not None: 
		print('Setting CUDA_VISIBLE_DEVICES to {}'.format(args.cuda_ids))
		os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_ids
	return args



def argparser_eps_boundaries():
	parser = argparse.ArgumentParser()

	#COMMON FOR BOUNDARIES AND EPS
	parser.add_argument('--method', type = str, default='ibp', help='method')

	# model arguments
	parser.add_argument('--models', '--mlist', nargs='+', default=['2D'], help='model can be {2D}')
	parser.add_argument('--model_expansion', type=int, default=1)
	parser.add_argument('--model_files', '--mplist', nargs='+', default=[])
	
	#dataset
	parser.add_argument('--datasets', '--dlist', nargs='+', default=['syn'], help='dataset can be {mnist, fmnist, cifar or specify a data_dir for custom dataset}')
	parser.add_argument('--batch_size', type = int, default = 4, help = 'the batch size, default = 500')
	parser.add_argument('--dataset_file', type = str, default=None, help='dataset file pickle to load')
	
	#BOUNDARIES BASED
	#scanning data
	parser.add_argument('--min_x', type = float, default = -1., help = 'the minimum value of x, default = -1.')
	parser.add_argument('--max_x', type = float, default = +1., help = 'the maximum value of x, default = +1.')
	parser.add_argument('--num_x', type = int, default = 300, help = 'the number of samples in x axis, default = 1000')
	parser.add_argument('--min_y', type = float, default = -1., help = 'the minimum value of y, default = -1.')
	parser.add_argument('--max_y', type = float, default = +1., help = 'the maximum value of y, default = +1.')
	parser.add_argument('--num_y', type = int, default = 300, help = 'the number of samples in y axis, default = 1000')
	parser.add_argument('--useA', type=bool, default=False)
	parser.add_argument('--use_dist_file', type=bool, default=True)
	
	#EPS BASED
	parser.add_argument('--max_iter', type = int, default = 400, help = 'the maximum iterations, default = 200/400')
	parser.add_argument('--beta', type = float, default = 1., help = 'the coefficient of the augment term, default = 1.')
	parser.add_argument('--inc_rate', type = float, default = 5., help = 'the ratio of increase in beta, default = 5') #10
	parser.add_argument('--inc_min', type = int, default = 0, help = 'the minimum iteration number for increasing the rho, default = 0')
	parser.add_argument('--inc_freq', type = int, default = 80, help = 'the frequency of the increase, default = 80') #100
	parser.add_argument('--update_dual_freq', type = int, default = 5, help = 'the frequency of updating dual variable, default = 5') #3
	parser.add_argument('--mode', type = str, default = 'nonuniform', help = 'the type of the certified bound, default = "nonuniform", supported = ["nonuniform", "uniform"]')
	parser.add_argument('--init_margin', type = float, default = 0.001, help = 'the bound initialization, default = 0.001')
	parser.add_argument('--norm', type = float, default = np.inf, help = 'the norm used for robustness, default = np.inf')
	parser.add_argument('--delta', type = float, default = 1e-4, help = 'the margin required to ensure the right prediction, default = 1e-4')
	parser.add_argument('--grad_clip', type = float, default = None, help = 'whether or not to apply gradient clipping, default = None')
	parser.add_argument('--final_decay', type = float, default = 0.99, help = 'the decay rate in the final search, default = 0.99')
	parser.add_argument('--pts_per_class', type = int, default = None, help = 'the number of pts per class, default = None, meaning no limitations.')
	# for learning
	parser.add_argument('--in_dim', type = int, default = 2, help = 'the number of input dimensions, default = 2')
	parser.add_argument('--out_dim', type = int, default = 2, help = 'the number of classes, default = 10')
	parser.add_argument('--optim', type = str, default = 'sgd', help = 'the type of the optimizer, default = "sgd"')
	
	parser.add_argument('--lr', type = float, default = 1.5e-3, help = 'the learning rate, default = 1e-3')
	parser.add_argument('--lr_w', type = float, default = 1e-3, help = 'the learning rate, default = 1e-2')
	parser.add_argument('--show_images', type=bool, default=False)

	args = parser.parse_args()
	return args


def argparser_demo(): 
	parser = argparse.ArgumentParser()
	
	#attack
	# parser.add_argument('--seed', type=int, default=seed)
	parser.add_argument('--dataset', type=str, default='mnist', help='dataset can be {mnist, fmnist, cifar or specify a data_dir for custom dataset}')
	parser.add_argument('--method', type=str, default='wong', help='method of training can be {baseline, madry, wong}')
	parser.add_argument('--model_files', '--mplist', nargs='+', default=[], required=True)
	parser.add_argument('--model', type=str, default='conv_2layer', help='model can be {2D}')
	parser.add_argument('--segm', type=bool, default=False)
	parser.add_argument('--epsilon', type=float, default=0.2)
	parser.add_argument('--save', type=bool, default=False)
	# other arguments
	parser.add_argument('--prefix', type=str, default='')
	parser.add_argument('--log_data', type=str, default='')
	parser.add_argument('--dataset_path', type=str, default='./')
	# output arguments
	parser.add_argument('--make_dataset', type=bool, default=False)
	parser.add_argument('--num_class', type=int, default=10)
	parser.add_argument('--attack_noise', '--anlist', nargs='+', default=[0.001])
	parser.add_argument('--pert_noise', '--pnlist', nargs='+', default=[0.001])
	parser.add_argument('--image_size', type=int, default=512)
	
	args = parser.parse_args()
	banned = ['verbose', 'prefix',
						'resume', 'baseline', 'eval', 
						'method', 'model', 'cuda_ids', 'load', 'real_time']
	banned += ['momentum', 'weight_decay']
	for arg in sorted(vars(args)): 
		if arg not in banned and getattr(args,arg) is not None: 
			args.log_data += '_' + arg + '_' +str(getattr(args, arg))
	return args


def argparser_adv_training(batch_size=4, epochs=100, seed=0, verbose=500, lr=5e-3, #5e-2
              epsilon=0.3, starting_epsilon=0.001, 
              opt='adam', momentum=0.9, weight_decay=5e-4): 
	parser = argparse.ArgumentParser()
	
	#attack
	# parser.add_argument('--seed', type=int, default=seed)
	parser.add_argument('--dataset', type=str, default='mnist', help='dataset can be {mnist, fmnist, cifar or specify a data_dir for custom dataset}')
	parser.add_argument('--method', type=str, default='madry', help='method of training can be {baseline, madry, wong}')
	parser.add_argument('--model_file', type=str, default=' ')
	parser.add_argument('--model', type=str, default='conv_2layer', help='model can be {2D}')
	parser.add_argument('--segm', type=bool, default=False)
	
	# other arguments
	parser.add_argument('--prefix', type=str, default='')
	parser.add_argument('--log_data', type=str, default='')
	parser.add_argument('--dataset_path', type=str, default='./')
	# output arguments
	parser.add_argument('--make_dataset', type=bool, default=False)
	parser.add_argument('--num_class', type=int, default=10)
	parser.add_argument('--attack_noise', '--anlist', nargs='+', default=[0.01])
	parser.add_argument('--pert_noise', '--pnlist', nargs='+', default=[0.01])
	
	parser.add_argument('--opt', default=opt)
	parser.add_argument('--momentum', type=float, default=momentum)
	parser.add_argument('--weight_decay', type=float, default=weight_decay)
	parser.add_argument('--batch_size', type=int, default=batch_size)
	parser.add_argument('--epochs', type=int, default=epochs)
	parser.add_argument("--lr", type=float, default=lr)
	# epsilon settings
	parser.add_argument("--epsilon", type=float, default=epsilon)
	parser.add_argument("--nonuniform", type=bool, default=False)
	parser.add_argument("--starting_epsilon", type=float, default=starting_epsilon)
	parser.add_argument('--schedule_length', type=int, default=30)
	parser.add_argument('--warm_start', type=int, default=1)
	parser.add_argument('--plateau_length', type=int, default=1)
	parser.add_argument('--verbose', type=int, default=verbose)
	
	parser.add_argument('--image_size', type=int, default=512)
	
	args = parser.parse_args()
	banned = ['verbose', 'prefix',
						'resume', 'baseline', 'eval', 
						'method', 'model', 'cuda_ids', 'load', 'real_time']
	banned += ['momentum', 'weight_decay']
	for arg in sorted(vars(args)): 
		if arg not in banned and getattr(args,arg) is not None: 
			args.log_data += '_' + arg + '_' +str(getattr(args, arg))
	return args