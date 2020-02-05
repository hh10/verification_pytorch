import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from .utils import Dense, DenseSequential
from .dual_inputs import select_input
from .dual_layers import select_layer
from .utils import PCA 

import numpy as np
import matplotlib.pyplot as plt
import warnings

class DualNetwork(nn.Module):   
    def __init__(self, net, X, epsilon, 
                 proj=None, norm_type='l1', bounded_input=False, 
                 data_parallel=True, pca_var=1., non_uniform=False, M=None, phase=1, y=None, intermediate=False):
        """  
        This class creates the dual network. 

        net : ReLU network
        X : minibatch of examples
        epsilon : size of l1 norm ball to be robust against adversarial examples
        alpha_grad : flag to propagate gradient through alpha
        scatter_grad : flag to propagate gradient through scatter operation
        l1 : size of l1 projection
        l1_eps : the bound is correct up to a 1/(1-l1_eps) factor
        m : number of probabilistic bounds to take the max over
        """
        super(DualNetwork, self).__init__()
        # need to change that if no batchnorm, can pass just a single example
        if not isinstance(net, (nn.Sequential, DenseSequential)): 
            raise ValueError("Network must be a nn.Sequential or DenseSequential module")
        with torch.no_grad(): 
            if any('BatchNorm2d' in str(l.__class__.__name__) for l in net): 
                zs = [X] #[[BS,1,28,28]]
            else:
                zs = [X[:1]] #[[1,1,28,28]]
            nf = [zs[0].size()]  #[[1,1,28,28]]
            for l in net:
                if isinstance(l, Dense): 
                    zs.append(l(*zs))
                else:
                    zs.append(l(zs[-1]))
                nf.append(zs[-1].size())

        # Use the bounded boxes
        dual_net = [select_input(X, epsilon, proj, norm_type, bounded_input, non_uniform=non_uniform, M=M)]

        for i,(in_f,out_f,layer) in enumerate(zip(nf[:-1], nf[1:], net)):
            dual_layer = select_layer(layer, dual_net, X, proj, norm_type, in_f, out_f, zs[i], pca_var=pca_var, phase=phase, intermediate=intermediate, y=y)
            # recursively apply upcoming layers on the previous layers; skip last layer as nothing more to apply on it
            if i < len(net)-1: 
                for l in dual_net:
                    l.apply(dual_layer)
                dual_net.append(dual_layer)
            else: 
                self.last_layer = dual_layer
        self.X = X
        self.dual_net = dual_net
        return 

    def forward(self, c):
        """ For the constructed given dual network, compute the objective for some given vector c """
        nu = [-c]
        nu.append(self.last_layer.T(*nu))
        for l in reversed(self.dual_net[1:]):
            nu.append(l.T(*nu))
        dual_net = self.dual_net + [self.last_layer]
        loss, reg_loss = 0, 0
        xp = self.X
        for i,l in enumerate(dual_net):
            if "InfBall" not in str(l):
                xp = l.forward([xp], simple=True)
            if "InfBall" not in str(l) and "ReLU" not in str(l) and "Reshape" not in str(l):
                reg_loss = reg_loss + torch.sum(torch.abs(l.layer.weight))
            if "ReLU" in str(l):
                loss += l.rd
        return sum(l.objective(*nu[:min(len(dual_net)-i+1, len(dual_net))]) for i,l in enumerate(dual_net)), loss, reg_loss, xp

class DualNetBounds(DualNetwork): 
    def __init__(self, *args, **kwargs):
        warnings.warn("DualNetBounds is deprecated. Use the proper "
                      "PyTorch module DualNetwork instead. ")
        super(DualNetBounds, self).__init__(*args, **kwargs)

    def g(self, c):
        return self(c)

class RobustBounds(nn.Module): 
    def __init__(self, net, epsilon, non_uniform=False, pca_var=1., norm_type='l1', **kwargs): 
        super(RobustBounds, self).__init__()
        self.net = net
        self.epsilon = epsilon
        self.kwargs = kwargs
        self.non_uniform = non_uniform
        self.pca_var = pca_var
        self.norm_type = norm_type
        
    def forward(self, X,y, get_bounds): 
        num_classes = self.net[-1].out_features
        dual = DualNetwork(self.net, X, self.epsilon, non_uniform=self.non_uniform, pca_var=self.pca_var, norm_type=self.norm_type, y=y, **self.kwargs)
        if get_bounds:
            zl, zu = robust_bounds(dual)
        else:
            zl, zu = 0, 0
        c = Variable(torch.eye(num_classes).type_as(X)[y].unsqueeze(1) - torch.eye(num_classes).type_as(X).unsqueeze(0))
        if X.is_cuda:
            c = c.cuda()
        f, floss, rloss, yhat = dual(c)
        return -f, floss, rloss, yhat, zl, zu

    def get_net(X):
        dual = DualNetwork(self.net, X, self.epsilon, non_uniform=self.non_uniform, pca_var=self.pca_var, **self.kwargs)
        return dual.dual_net

def get_RobustBounds(net, epsilon, X, y, get_bounds, non_uniform=False, pca_var=1., **kwargs):
    num_classes = net[-1].out_features
    dual = DualNetwork(net, X, epsilon, non_uniform=non_uniform, pca_var=1., **kwargs)
    zl, zu = robust_bounds(dual)
    if pca_var<1.:
        dual_pca = DualNetwork(net, X, epsilon, non_uniform=non_uniform, pca_var=pca_var, **kwargs)
        zl_pca, zu_pca = robust_bounds(dual_pca)
        return zl, zu, zl_pca, zu_pca    
    else:
        return zl, zu, zl, zu    

def robust_bounds(NN):
    NN_net = NN.dual_net
    for l in NN_net:
        l.apply(NN.last_layer)
    qzl, qzu = zip(*[l.bounds() for l in NN_net])
    zl, zu = sum(qzl), sum(qzu)
    zll, zul = NN.last_layer.bounds()
    zl = zl + zll
    zu = zu + zul
    
    szl, szu = NN_net[0].lb_simple, NN_net[0].ub_simple
    for l in NN_net[1:]:
        szl, szu = l.simp_bounds(szl, szu)
    szl, szu = NN.last_layer.simp_bounds(szl, szu)
    zl = torch.max(zl, szl)
    zu = torch.min(zu, szu)
    return zl, zu


def robust_loss(net, epsilon, X, y, get_bounds, non_uniform=False,
                size_average=True, device_ids=None, parallel=False, pca_var=1., norm_type='l1', intermediate=False, kk=1, **kwargs):
    if intermediate:
        with torch.no_grad(): 
            RB_net = RobustBounds(net, epsilon=0.0001, non_uniform=non_uniform, norm_type=norm_type, intermediate=True, **kwargs)      
            f_dontu, _, _, _, zl_dontu, zu_dontu = RB_net(X,y,get_bounds)
            del RB_net, f_dontu, zl_dontu, zu_dontu   
        torch.cuda.empty_cache()
    if parallel: 
        f = nn.DataParallel(RobustBounds(net, epsilon, **kwargs))(X,y)
    else: 
        f, floss, rloss, yhat, zl, zu = RobustBounds(net, epsilon, non_uniform=non_uniform, norm_type=norm_type, intermediate=False,**kwargs)(X,y,get_bounds)
    '''
    rd = 0
    for i in range(1,18*3+1):
        try:
            rdiffs = np.load('./exp_res/dummy/linf_diff_filter_'+str(i)+'.npy')
            rd += np.linalg.norm(rdiffs.reshape(-1)) 
        except Exception as e:
            pass
    '''
    err = (f.max(1)[1] != y)
    if size_average: 
        err = err.sum().item()/X.size(0)
    ce_loss = nn.CrossEntropyLoss(reduce=size_average)(f, y)
    '''
    if err == 1.:
        ce_loss = nn.CrossEntropyLoss(reduce=size_average)(yhat, y)+ kk*0.001*rloss
    else:
        ce_loss = kk*0.005*floss + nn.CrossEntropyLoss(reduce=size_average)(yhat, y) + 2*0.001*rloss
    ce_loss += rd
    '''
    if pca_var<1.:
        pca_f, floss, rloss, yhat, zl_pca, zu_pca = RobustBounds(net, epsilon, non_uniform=non_uniform, pca_var=pca_var, norm_type=norm_type, intermediate=False, **kwargs)(X,y,get_bounds)
        pca_err = (pca_f.max(1)[1] != y)        
        if size_average: 
            pca_err = pca_err.sum().item()/X.size(0)
        return ce_loss, err, zl, zu, zl_pca, zu_pca, pca_err, f
    else:
        return ce_loss, err, zl, zu, 0, 0, 0, f   


def get_dual_net(net, epsilon, X, y, get_bounds, non_uniform=False,
                size_average=True, device_ids=None, parallel=False, pca_var=1., **kwargs):
    return RobustBounds(net, epsilon, non_uniform=non_uniform, norm_type=norm_type, **kwargs).get_net(X)

class InputSequential(nn.Sequential): 
    def __init__(self, *args, **kwargs): 
        self.i = 0
        super(InputSequential, self).__init__(*args, **kwargs)

    def set_start(self, i): 
        self.i = i

    def forward(self, input): 
        """ Helper class to apply a sequential model starting at the ith layer """
        xs = [input]
        for j,module in enumerate(self._modules.values()): 
            if j >= self.i: 
                if 'Dense' in type(module).__name__:
                    xs.append(module(*xs))
                else:
                    xs.append(module(xs[-1]))
        return xs[-1]


# Data parallel versions of the loss calculation
def robust_loss_parallel(net, epsilon, X, y, proj=None, 
                 norm_type='l1', bounded_input=False, size_average=True): 
    if any('BatchNorm2d' in str(l.__class__.__name__) for l in net): 
        raise NotImplementedError
    if bounded_input: 
        raise NotImplementedError('parallel loss for bounded input spaces not implemented')
    if X.size(0) != 1: 
        raise ValueError('Only use this function for a single example. This is '
            'intended for the use case when a single example does not fit in '
            'memory.')
    zs = [X[:1]]
    nf = [zs[0].size()]
    for l in net: 
        if isinstance(l, Dense): 
            zs.append(l(*zs))
        else:
            zs.append(l(zs[-1]))
        nf.append(zs[-1].size())

    dual_net = [select_input(X, epsilon, proj, norm_type, bounded_input)]

    for i,(in_f,out_f,layer) in enumerate(zip(nf[:-1], nf[1:], net)): 
        if isinstance(layer, nn.ReLU): 
            # compute bounds
            D = (InputSequential(*dual_net[1:]))
            Dp = nn.DataParallel(D)
            zl,zu = 0,0
            for j,dual_layer in enumerate(dual_net): 
                D.set_start(j)
                out = dual_layer.bounds(network=Dp)
                zl += out[0]
                zu += out[1]

            dual_layer = select_layer(layer, dual_net, X, proj, norm_type,
                in_f, out_f, zs[i], zl=zl, zu=zu)
        else:
            dual_layer = select_layer(layer, dual_net, X, proj, norm_type,
                in_f, out_f, zs[i])
        
        dual_net.append(dual_layer)

    num_classes = net[-1].out_features
    c = Variable(torch.eye(num_classes).type_as(X)[y].unsqueeze(1) - torch.eye(num_classes).type_as(X).unsqueeze(0))
    if X.is_cuda:
        c = c.cuda()
    
    # same as f = -dual.g(c)
    nu = [-c]
    for l in reversed(dual_net[1:]): 
        nu.append(l.T(*nu))
    
    f = -sum(l.objective(*nu[:min(len(dual_net)-i+1, len(dual_net))]) 
             for i,l in enumerate(dual_net))

    err = (f.max(1)[1] != y)

    if size_average: 
        err = err.sum().item()/X.size(0)
    ce_loss = nn.CrossEntropyLoss(reduce=size_average)(f, y)
    return ce_loss, err