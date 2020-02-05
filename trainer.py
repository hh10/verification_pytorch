""" Contains functions defining the various large, small, wide, residual models being used. """
import numpy as np
import time
import math
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import pickle

#pytorch imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.cuda
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch import autograd

#our imports
from convex_adversarial import robust_loss as robust_loss_wong, get_dual_net
from visualize import discrete_cmap
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
    (255, 255, 255, 255): 1,  # Plane
    (0, 0, 0, 255): 0,  # Sky
}

PLOT_TSNE = False
kk = 1
	
def train(train_loader, method_list, model_list, opt_list, criterion, t, epsilon, train_log_handles, norm_type='l1', bounded_input=False, num_class=2, non_uniform=False, A=None, pca_var=1., proj=0, fig=None, K=1., M=None, warm_start=0, tot=0, DEBUG=True, stop_batch=0, save_path=None, plt_acc=None, plt_acc_avg=None, plt_loss=None, plt_rloss=None, plt_loss_avg=None, plt_rloss_avg=None, plt_sparsity=None, intermediate=False):
	global kk
	data_time = AverageMeter()
	end = time.time()
	batch_time, losses, errors, robust_losses, robust_errors, params = make_params(method_list, train_log_handles)
	
	for model in model_list:
		model.train()
	
	if t%5 == 0 and t!=0:
		kk = kk*1.1
		
	for i, (X,y) in enumerate(train_loader):
		print('batch {}/{}: '.format(i, tot))
		X, y = X.cuda(), y.cuda().long()
		if y.dim() == 2: 
			y = y.squeeze(1)
		data_time.update(time.time() - end)
		#train all models with one method on the same example
		for m in range(len(method_list)):
			train_method(X, y, model_list[m], opt_list[m], method_list[m], criterion, t, epsilon, train_log_handles[m], norm_type, i, 
						params, batch_time[m], losses[m], errors[m], robust_losses[m], robust_errors[m], clip_grad=None, num_class=num_class,
						non_uniform=non_uniform, A=A, pca_var=pca_var, proj=proj, M=M, fig = fig, K=K, warm_start=warm_start, tot=tot, DEBUG=DEBUG, save_path=save_path, intermediate=intermediate, kk=kk)
		if DEBUG and i==stop_batch and t>=warm_start: #2*num_class:
			break
	for m in range(len(method_list)):
		plt_acc[m].append(1-errors[m].val)
		plt_acc_avg[m].append(1-errors[m].avg)
		plt_loss[m].append(losses[m].val)
		plt_rloss[m].append(robust_losses[m].val)
		plt_loss_avg[m].append(losses[m].avg)
		plt_rloss_avg[m].append(robust_losses[m].avg)
		nn_weights = []
		for l in model_list[m]:
			try:
				if method_list[m] == 'wong':
					nn_weights.extend(l.layer.weight.view(-1).data.cpu().numpy())
				else:
					nn_weights.extend(l.weight.view(-1).data.cpu().numpy())
				print(l)
			except:
				print(l, ' has no weight, continuing')
				continue
		up = np.array(nn_weights)<0.01
		be = np.array(nn_weights)>-0.01
		sparsity = np.sum((up & be)*1.)/len(nn_weights)
		plt_sparsity[m].append(sparsity)
	print('training done')
	torch.cuda.empty_cache()
	return params

def train_method(X, y, model, opt, method, criterion, epoch, epsilon, log, norm_type, i, params, batch_time, losses, errors, robust_losses, robust_errors, clip_grad=None, num_class=2, non_uniform=False, A=None, pca_var=1., proj=0, M=None, fig=None, criterion0=None, K=1., warm_start=0, tot=0, DEBUG=True, save_path=None, intermediate=False, kk=1):	
	#gather params for the second class		
	test_class = 1
	end = time.time()
	if method == 'baseline':
		out = model(X)
		ce =  criterion(out, y)
		loss = ce
		err = (out.max(1)[1] != y).float().sum() / X.size(0)		
		update_params(method, params, y, num_class, test_class, [out.clone()], K)
		print('epoch: ', epoch, ' err: ', round(err.item(),4), 'nominal loss: ', round(ce.item(),4))
	else:
		with torch.no_grad(): 
			out = model(Variable(X))
			ce = criterion(out, Variable(y))
			err = (out.max(1)[1] != y).float().sum() / X.size(0)

		if method == 'wong':
			plot = True if epoch%4==0 else False
			loss, robust_err, lbs, ubs, _, _, _, _ = robust_loss_wong(model, epsilon, X, y, plot, pca_var=pca_var, non_uniform=non_uniform, proj=proj, intermediate=intermediate, kk=kk)
			if plot and DEBUG:
				update_params(method, params, y, num_class, test_class, [out.clone(), lbs.clone(), ubs.clone()], K)	
	opt.zero_grad()
	loss.backward()
	if clip_grad: 
		nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
	opt.step()
	batch_time.update(time.time()-end)
	
	#updates
	losses.update(ce.detach().cpu().item(), X.size(0))
	errors.update(err.detach().cpu().item(), X.size(0))
	if method != 'baseline':
		robust_losses.update(loss.detach().cpu().item(), X.size(0))
		robust_errors.update(robust_err, X.size(0))		
	
	if i%50==0 and method != 'baseline':
		print('Epoch: [{0}][{1}/{2}]\t''Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
				'Loss {loss.val:.4f} ({loss.avg:.4f})\t''Error {errors.val:.3f} ({errors.avg:.3f})\t'
				'Robust loss {rloss.val:.4f} ({rloss.avg:.4f})\t''Robust error {rerrors.val:.3f} ({rerrors.avg:.3f})'.format(
				epoch, i, tot, batch_time=batch_time, loss=losses, errors=errors, rloss = robust_losses, rerrors = robust_errors), file=log)
	elif i%50==0 and method == 'baseline':
		print('Epoch: [{0}][{1}/{2}]\t''Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
				'Loss {loss.val:.4f} ({loss.avg:.4f})\t''Error {errors.val:.3f} ({errors.avg:.3f})'.format(
				epoch, i, tot, batch_time=batch_time, loss=losses, errors=errors), file=log)
	log.flush()
	del X, y, out, err, ce, losses
	if method != 'baseline':
		del robust_err
	torch.cuda.empty_cache()


def evaluate(test_loader, method_list, model_list, criterion, t, epsilon, test_log_handles, num_class=2, norm_type='l1', non_uniform=False, A=None, pca_var=1., proj=0, tot=0, DEBUG=True, stop_batch=0, save_path=None):
	batch_time, losses, errors, robust_losses, robust_errors, params = make_params(method_list, test_log_handles)
	bound_size = []
	for i in range(len(method_list)):
		bound_size.append(AverageMeter()) 
	lower_bounds, upper_bounds = [[]] * (len(method_list)), [[]] * (len(method_list))
	
	for model in model_list:
		model.eval()
	torch.set_grad_enabled(False)
	for i, (X,y) in enumerate(test_loader):
		X,y = X.cuda(), y.cuda().long()
		if y.dim() == 2: 
			y = y.squeeze(1)
		
		for m in range(len(method_list)):
			lb, ub = evaluate_method(X, y, model_list[m], method_list[m], criterion, t, epsilon, test_log_handles[m], num_class, norm_type, i, 
						params, batch_time[m], losses[m], errors[m], robust_losses[m], robust_errors[m], bound_size[m], non_uniform=non_uniform, A=A, pca_var=pca_var, proj=proj, tot=tot, DEBUG=DEBUG, save_path=save_path)
			lower_bounds[m].append(lb)
			upper_bounds[m].append(ub)
		if DEBUG and i==stop_batch: 
			break
	#xx = input()
	torch.set_grad_enabled(True)
	torch.cuda.empty_cache()
	return params, errors, robust_errors, lower_bounds, upper_bounds
		
def evaluate_method(X, y, model, method, criterion, epoch, epsilon, log, num_class, norm_type, i, params, batch_time, losses, errors, robust_losses, robust_errors, bound_size, non_uniform=False, A=None, pca_var=1., proj=0, tot=0, DEBUG=True,  save_path=None):
	test_class = 1
	end = time.time()
	
	out = model(Variable(X))
	ce =  criterion(out, Variable(y))
	err = (out.max(1)[1] != y).float().sum() / X.size(0)
	losses.update(ce.item(), X.size(0))
	errors.update(err.item(), X.size(0))
	
	lb, ub = 0, 0
	if method == 'baseline':
		update_params(method, params, y, num_class, test_class, [out.clone()], K=0)
	elif method == 'wong':
		robust_ce, robust_err, lb, ub, _, _, _ = robust_loss_wong(model, epsilon, X, y, True, proj=proj, pca_var=pca_var, non_uniform=non_uniform)
		update_params(method, params, y, num_class, test_class, [out.clone(), lb.clone(), ub.clone()], K=0)
		robust_losses.update(robust_ce.detach().cpu().item(), X.size(0))
		robust_errors.update(robust_err, X.size(0))
		bound_size.update(np.average((ub-lb).cpu().float()), X.size(0))
	batch_time.update(time.time()-end)
		
	if i%50==0:
		print('Test_Epoch: [{0}][{1}/{2}]\t''Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
				'Loss {loss.val:.4f} ({loss.avg:.4f})\t''Error {error.val:.3f} ({error.avg:.3f})\t'
				'Robust loss {rloss.val:.3f} ({rloss.avg:.3f})\t''Robust error {rerrors.val:.3f} ({rerrors.avg:.3f})'.format(
				epoch, i, tot, batch_time=batch_time, loss=losses, error=errors, rloss = robust_losses, rerrors = robust_errors), file=log)
	log.flush()
	
	del X, y, out, ce
	if method != 'baseline':
		del robust_ce, robust_err
	if method == 'baseline':
		print(' * Error {error.avg:.3f}'.format(error=errors))
	else:
		print(' * Robust error {rerror.avg:.3f}\t Error {error.avg:.3f}\t Bound_size {b_size.avg:.3f}'.format(rerror=robust_errors, error=errors, b_size=bound_size))
	return lb, ub


def evaluate_verifier(test_loader, method_list, model_list, criterion, t, epsilon, test_log_handles, num_class=2, norm_type='l1', non_uniform=False, A=None, pca_var=1., proj=0, tot=0, DEBUG=True, stop_batch=200,  save_path=None, pretrained=False, fig=None, pca=None, embed_train=None, X_train=None, y_train=None, intermediate=False, **kwargs):
	batch_time, losses, errors, robust_losses, robust_errors, params = make_params(method_list, test_log_handles)
	bound_size = []
	for i in range(len(method_list)):
		bound_size.append(AverageMeter()) 
	lower_bounds, upper_bounds = [[]] * (len(method_list)), [[]] * (len(method_list))
	if pca_var<1.:
		bound_size_pca, robust_errors_pca = [], []
		for i in range(len(method_list)):
			bound_size_pca.append(AverageMeter()) 
			robust_errors_pca.append(AverageMeter())
		lower_bounds_pca, upper_bounds_pca = [[]] * (len(method_list)), [[]] * (len(method_list))
	
	for model in model_list:
		model.eval()
	torch.set_grad_enabled(False)

	#for i, li in enumerate(test_loader):
	#	X,y = li[0].cuda(), li[1].cuda().long()	
	for i, (X,y) in enumerate(test_loader):
		X, y = X.cuda(), y.cuda().long()		
		if y.dim() == 2: 
			y = y.squeeze(1)
		for m in range(len(method_list)):
			end = time.time()
	
			out = model(X)
			if PLOT_TSNE:
				#to plot tsne vector for this in the same fig
				embd = pca.transform(out.clone().cpu().numpy())
				closest_embd_ind = find_closest_embd(embed_train, embd)
				closest_l1_ind = find_closest_l1(X_train, X)
				plt.figure(fig.number)
				plt.scatter(embd[:, 0], embd[:, 1], c=discrete_cmap(num_class, 'jet')(y.clone().cpu().numpy()), marker='o', edgecolor='none')
				plt.draw()
				plt.pause(0.2)
				fig_img = plt.figure(figsize=(8,8))
				plt.figure(fig_img.number)
				fig_img.add_subplot(1, 3, 1)		
				plt.imshow(X.clone().squeeze().cpu().numpy(), cmap='gray', vmin=0, vmax=1)
				fig_img.add_subplot(1, 3, 2)		
				plt.imshow(X_train[closest_embd_ind].squeeze().cpu().numpy(), cmap='gray', vmin=0, vmax=1)
				fig_img.add_subplot(1, 3, 3)		
				plt.imshow(X_train[closest_l1_ind].squeeze().cpu().numpy(), cmap='gray', vmin=0, vmax=1)
				plt.draw()
				plt.pause(0.2)
				xx=input()
				plt.close()

			ce = criterion(out, y)
			err = (out.max(1)[1] != y).float().sum() / X.size(0)
			losses[m].update(ce.item(), X.size(0))
			errors[m].update(err.item(), X.size(0))
	
			lb, ub = 0, 0
			if method_list[m] == 'wong':
				robust_ce, robust_err, lb, ub, lb_pca, ub_pca, pca_err, f = robust_loss_wong(model, epsilon, X, y, True, proj=proj, pca_var=pca_var, non_uniform=non_uniform, intermediate=intermediate,**kwargs)
				# to store the output logits from model to a file, uncomment the block below
				'''
				file1 = open("./f_cifar_resnet.txt", 'ab') #adversarial_03_005.txt","ab") 
				pickle.dump([y.item(), f.data.cpu().numpy()[0]], file1)
				file1.close() 
				'''
				# to store the weight distribution as a hist per layer, uncomment the block below
				'''
				if save_path is not None:
		            nn_weights = []
		            for l in get_dual_net(model, epsilon, X, y, True, proj=proj, pca_var=pca_var, non_uniform=non_uniform, **kwargs):
		                try:
		                    nn_weights.append(l.layer.weight.view(-1).data.cpu().numpy())
		                except:
		                    continue
		            plt.hist(nn_weights, bins=1000)
		            plt.title('Weights distribution')
		            plt.gcf().savefig(save_path + '/weights_distr_' + str(epoch) + '.png')
        		'''
				if pca_var<1.:
					bound_size_pca[m].update(np.average((ub_pca-lb_pca).cpu().float()), X.size(0))
					robust_errors_pca[m].update(pca_err, X.size(0))
					lower_bounds_pca[m].append(lb_pca)
					upper_bounds_pca[m].append(ub_pca)
			robust_losses[m].update(robust_ce.detach().cpu().item(), X.size(0))
			robust_errors[m].update(robust_err, X.size(0))
			bound_size[m].update(np.average((ub-lb).cpu().float()), X.size(0))
			batch_time[m].update(time.time()-end)

			del X, y, out, ce, robust_ce, robust_err
			if pca_var<1.:
				print('* {ii}/{tt} * Error {error.avg:.3f}\t Robust error {rerror.avg:.3f}\t PCARobust error {pcarerror.avg:.3f}\t Bound_size {b_size.avg:.3f}\t PCABound_size {pcab_size.avg:.3f}'.format(ii=i, tt=t, rerror=robust_errors[m], error=errors[m], b_size=bound_size[m], pcab_size=bound_size_pca[m], pcarerror=robust_errors_pca[m]))
			else:
				print('* {ii}/{tt} * Error {error.avg:.3f}\t Robust error {rerror.avg:.3f}\t Bound_size {b_size.avg:.3f}'.format(ii=i, tt=t, rerror=robust_errors[m], error=errors[m], b_size=bound_size[m]))
			lower_bounds[m].append(lb)
			upper_bounds[m].append(ub)
		if DEBUG and i==stop_batch: 
			break
	torch.set_grad_enabled(True)
	torch.cuda.empty_cache()
	if pca_var<1:
		return errors, robust_errors, robust_errors_pca, lower_bounds, upper_bounds, lower_bounds_pca, upper_bounds_pca
	else:
		return errors, robust_errors, 0, lower_bounds, upper_bounds, 0, 0
			

def segm(mode, train_loader, method_list, model_list, opt_list, t, epsilon, train_log_handles, verbose, norm_type='l1', bounded_input=False, fig=None, **kwargs):
	#structure of train_loader : [train_images, train_labels] 
	data_time = AverageMeter()
	end = time.time()

	batch_time, losses, errors = [], [], []
	for i in range(len(method_list)):
		batch_time.append(AverageMeter()) 
		losses.append(AverageMeter())
		errors.append(AverageMeter())

	for i, (X,y) in enumerate(zip(train_loader[0], train_loader[1])):
		#checks
		columns, rows = 4, 1
		plt.clf()
		fig.suptitle('Training')
		fig.add_subplot(rows, columns, 1)
		plt.imshow(transforms.ToPILImage()(X))
		fig.add_subplot(rows, columns, 2)
		plt.imshow(transforms.ToPILImage()(y))
		#checks over
		Y = convert_to_one_hot(y)
		X, Y = X.unsqueeze(0).cuda(), torch.from_numpy(Y).unsqueeze(0).cuda().float()
		#assert Y.size() == torch.Size([1, 10, 160, 160]) #will do batches later  
		#assert X.size() == torch.Size([1, 3, 160, 160]) #will do batches later  
		data_time.update(time.time() - end)
		
		for m in range(len(method_list)):
			if mode=='train':
				ce, Y_hat, Y_labels = segm_method('train', X, Y, model_list[m], opt_list[m], method_list[m], t, epsilon, train_log_handles[m], verbose, i, batch_time[m], losses[m], errors[m], clip_grad=None)
			elif mode=='test':
				Y_labels = segm_method('test', X, Y, model_list[m], opt_list[m], method_list[m], t, epsilon, train_log_handles[m], verbose, i, batch_time[m], losses[m], errors[m], clip_grad=None)
			y_ori_hat = convert_to_label(Y, y, 'Y_original after converting to one hot')
			y_hat = convert_to_label(Y_labels, y, 'Y_output from training/testing')
			fig.add_subplot(rows, columns, 3)
			plt.imshow(transforms.ToPILImage()(y_hat))
			fig.add_subplot(rows, columns, 4)
			plt.imshow(transforms.ToPILImage()(y_ori_hat))
			plt.title('check')
			plt.draw()
			plt.pause(0.01)
		if DEBUG and i==10:
			break
	if mode=='train':
		print('training done')
		return
	elif mode=='test':
		print('testing done')
		return errors

def segm_method(mode, X, Y, model, opt, method, t, epsilon, log, verbose, i, batch_time, losses, errors, clip_grad=None):
	if mode=='train':
		model.train()
	elif mode=='test':
		model.eval()

	end = time.time()
	out = torch.sigmoid(model(X))
	assert out.shape==Y.shape
	main_labels = np.argmax(out.data.cpu().numpy(), 1)
	Y_labels = np.zeros((len(label2class), out.shape[2], out.shape[3]))
	for label in range(len(label2class)):
		mask = (main_labels[:,:,:]==label)
		Y_labels[label,:,:] = mask*1.
	print(Y.shape)
	err = np.absolute(Y_labels-Y.data.cpu().numpy()).sum()/(Y.size(0)*Y.size(2)*Y.size(3))
	ce = nn.BCELoss(reduce=True)(out, Y)
	#ce = ce.permute(0,2,3,1)
	# different losses for different classes
	#err_matrix = torch.FloatTensor([0.1, 0.05, 5, 0.1, 0.05, 0.1, 0.1, 0.1, 0.1, 0.1]).cuda()
	#ce = (ce*err_matrix).sum()/torch.ones(1, out.shape[2], out.shape[3], len(label2class)).cuda().sum()
	
	losses.update(ce.data.cpu().item(), X.size(0))
	errors.update(err, X.size(0))
	if mode=='train':
		opt.zero_grad()
		ce.backward()
		opt.step()
	batch_time.update(time.time()-end)
	print(' * ', mode, ' Error {error.avg:.3f}'.format(error=errors))
	
	if verbose and i % verbose == 0: 
		print('Epoch: [{0}][{1}/{2}]\t''Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t''Loss {loss.val:.4f} ({loss.avg:.4f})\t''Error {errors.val:.3f} ({errors.avg:.3f})'.format(t, i, len(X), batch_time=batch_time, loss=losses, errors=errors), file=log)
	log.flush()
	if mode=='train':
		del X, Y, err
		torch.cuda.empty_cache()	
		return ce, out, torch.from_numpy(Y_labels).unsqueeze(0).cuda().float()	
	elif mode=='test':
		del X, Y, out, ce
		return torch.from_numpy(Y_labels).unsqueeze(0).cuda().float()	


#utils
def wasserstein_cost2(ks, p=2):
    C = torch.zeros((ks*ks, ks*ks))
    for i in range(ks): 
        for j in range(ks): 
            for ii in range(ks): 
                for jj in range(ks):
                    C[i*ks+j, ii*ks+jj] = (abs(i-ii)**2 + abs(j-jj)**2)**(p/2)
    print('made C of shape', C.shape)
    return C

def make_params(method_list, log_handles):
	batch_time, losses, errors, robust_losses, robust_errors = [], [], [], [], []
	for i in range(len(method_list)):
		batch_time.append(AverageMeter()) 
		losses.append(AverageMeter())
		errors.append(AverageMeter())
		robust_losses.append(AverageMeter())
		robust_errors.append(AverageMeter())
		print("\nEpoch, Batch Time, Batch Number, Loss, Err", file=log_handles[i])
		print(method_list[i], file=log_handles[i])
	params = {}
	params['corr_lbound_baseline'] = []
	params['wrong_ubound_baseline'] = []
	for method in method_list:
		if method == 'wong':
			params['corr_wong'] = []
			params['wrong_wong'] = []
			params['corr_lbound_wong'] = []
			params['wrong_ubound_wong'] = []
		elif method == 'wongIBP':
			params['corr_wibp'] = []
			params['wrong_wibp'] = []
			params['corr_lbound_wibp'] = []
			params['wrong_ubound_wibp'] = []
			params['corr_lbound_wibp_q'] = []
			params['wrong_ubound_wibp_q'] = []
			params['corr_lbound_wibp_s'] = []
			params['wrong_ubound_wibp_s'] = []
		elif method == 'ibp':
			params['corr_ibp'] = []
			params['wrong_ibp'] = []
			params['corr_lbound_ibp'] = []
			params['wrong_ubound_ibp'] = []
	return batch_time, losses, errors, robust_losses, robust_errors, params

def update_params(method, params, y, num_class, test_class, list, K):
	if test_class is None or test_class in y.data.cpu():
		if method == 'baseline':
			cl, wu = bound_var(list[0], list[0], y, num_class, test_class)	
			params['corr_lbound_baseline'].append(cl)
			params['wrong_ubound_baseline'].append(wu)
		elif method == 'wong':
			cl, wu = bound_var(list[0], list[0], y, num_class, test_class)	
			params['corr_wong'].append(cl)
			params['wrong_wong'].append(wu)
			cl, wu = bound_var(list[1], list[2], y, num_class, test_class)
			params['corr_lbound_wong'].append(cl)
			params['wrong_ubound_wong'].append(wu)
		elif method == 'wongIBP':
			cl, wu = bound_var(list[0], list[0], y, num_class, test_class)	
			params['corr_v2'].append(cl)
			params['wrong_v2'].append(wu)
			if K!=1:
				cl, wu = bound_var(list[1], list[2], y, num_class, test_class)
				cl_q, wu_q = bound_var(list[3], list[4], y, num_class, test_class)
				cl_s, wu_s = bound_var(list[5], list[6], y, num_class, test_class)
				params['corr_lbound_v2'].append(cl)
				params['wrong_ubound_v2'].append(wu)
				params['corr_lbound_v2q'].append(cl_q)
				params['wrong_ubound_v2q'].append(wu_q)
				params['corr_lbound_v2s'].append(cl_s)
				params['wrong_ubound_v2s'].append(wu_s)
		elif method == 'ibp':
			cl, wu = bound_var(list[0], list[0], y, num_class, test_class)	
			params['corr_ibp'].append(cl)
			params['wrong_ibp'].append(wu)
			if K!=1:
				cl, wu = bound_var(list[1], list[2], y, num_class, test_class)
				params['corr_lbound_ibp'].append(cl)
				params['wrong_ubound_ibp'].append(wu)

def unique_pixels(y, name):
	pixs=list(transforms.ToPILImage()(y).getdata())
	pix=[]
	for x in pixs:
		if x not in pix:
			pix.append(x)
	#print('Unique pixels in {}: {}'.format(name, pix))
	#assert len(pix)<=10
	
def convert_to_one_hot(y):
	#[3, N, N]
	unique_pixels(y, 'labels original')
	Y = np.zeros((len(label2class), y.size()[1], y.size()[2]))
	for k,v in label2class.items():
		r = (np.floor((y[0,:,:])*100)/100 == np.floor((k[0]/255.0)*100)/100)
		g = (np.floor((y[1,:,:])*100)/100 == np.floor((k[1]/255.0)*100)/100)
		b = (np.floor((y[2,:,:])*100)/100 == np.floor((k[2]/255.0)*100)/100)
		c = (r*g*b)
		Y[v,:,:] = c
	#[10, N, N]
	return Y
	
def convert_to_label(Y, y_ideal, name):
	# [1, 10, N, N]
	y = np.zeros((3, Y.size()[2], Y.size()[3]), dtype="uint8")
	for k,v in label2class.items():
		c = (Y[0,v,:,:]==1.).cpu().numpy()
		y[0,:,:] = np.add(y[0,:,:], k[0]*c)
		y[1,:,:] = np.add(y[1,:,:], k[1]*c)
		y[2,:,:] = np.add(y[2,:,:], k[2]*c)
	y = torch.from_numpy(y)
	unique_pixels(y_ideal, name+' ideal')
	unique_pixels(y, name)
	#[3, N, N]
	return y

def log_GPU(here):
	print("GPU: {} ({}), {} ({}) at {}".format(torch.cuda.memory_allocated(torch.cuda.current_device()), torch.cuda.max_memory_allocated(torch.cuda.current_device()), torch.cuda.memory_cached(torch.cuda.current_device()),  torch.cuda.max_memory_cached(torch.cuda.current_device()), here))
	
class AverageMeter(object):
	"""Computes and stores the average and current value"""
	def __init__(self):
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count
		
def bound_var(lbs, ubs, y, num_class, test_class=None):
	corr_lbound, wrong_ubound = [], []
	for i in range(len(y)):
		if test_class is None or y[i]==test_class:
			corr_lbound.append(lbs[i][y[i]].detach().cpu().item())
			wl = []
			for j in range(num_class):
				if j != y[i]:
					wl.append(ubs[i][j])
			wrong_ubound.append(max(wl).detach().cpu().item())
	return min(corr_lbound), max(wrong_ubound)

def find_closest_embd(embd_train, embd_test):
	diff = embd_train - embd_test
	assert diff.shape==embd_train.shape
	sum_diff = np.sum(np.absolute(diff), axis=1)
	return np.argmin(sum_diff)

def find_closest_l1(X_train, X_test):
	diff = X_train.cpu().numpy().reshape(X_train.shape[0], -1) - X_test.cpu().numpy().reshape(X_test.shape[0], -1)
	assert diff.shape[0]==X_train.shape[0]
	sum_diff = np.sum(np.absolute(diff), axis=1)
	return np.argmin(sum_diff)