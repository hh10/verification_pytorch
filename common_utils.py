def get_standard_params(dataset, model, args, pretrained=False):
	if pretrained:
		dataset = 'imagenet'
		print("ATTENTION, since using pretrained model, dataset is IMAGENET..!")
	if dataset == 'syn':
		model = "2D"
		dim = 2
		num_class = args.num_class
		num_channels = 1
		print("Try only 2D for synthetic dataset") 
	elif dataset == 'mnist':
		model = model
		dim = 28
		num_class = 10
		num_channels = 1
	elif dataset == 'cifar':
		model = model
		dim = 32
		num_class = 10
		num_channels = 3
	else:
		model = model
		dim = args.image_size #default is 512.!, better to start with something small
		num_class = args.num_class
		num_channels = 1
		print("ATTENTION, using defaut settings for your CUSTOM DATASET..!")
	print("current configs are: ", dataset, model, dim, num_channels, num_class)
	return dataset, model, dim, num_class, num_channels

def make_params(methods):
	params_epoch_train, params_epoch_test = {}, {}
	params = [params_epoch_train, params_epoch_test]
	for params_epoch in params:
		for method in methods:
			if method== 'baseline':
				params_epoch['actual'] = []
			elif method == 'wong':
				params_epoch['wong'] = []
			elif method == 'wongIBP':
				params_epoch['wibp'] = []
				params_epoch['wibp_q'] = []
				params_epoch['wibp_s'] = []
			elif method == 'ibp':
				params_epoch['ibp'] = []
	return params_epoch_train, params_epoch_test

def prepare_logs_checkpoints(methods, prefix, log_dir_path, log_data):
	model_paths, checkpoint_paths = [[]]*len(methods), [[]]*len(methods)
	train_log_handles, test_log_handles = [], []
	for method in methods:
		train_log = open(log_dir_path + method +"_train.log", "w")
		test_log = open(log_dir_path + method + "_test.log", "w")
		for log in [train_log, test_log]:
			log.write(log_data)
			log.write(prefix)
			log.write(method)
		train_log_handles.append(train_log)
		test_log_handles.append(test_log)
	return train_log_handles, test_log_handles, model_paths, checkpoint_paths
