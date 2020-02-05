import torch
import torch.nn as nn
import torch.nn.functional as F

from .dual import DualLayer
from .utils import full_bias, Dense, PCA, get_PFA
import matplotlib.pyplot as plt
import numpy as np
import json as simplejson
import seaborn as sns

dict_activations = {}
epoch = {}
index = 0
relu_index = 0
max_relu_index = 4 #change as per model
max_index = 19 #change as per model
CH = False
OOD = False
STABILITY = False
ZERO_SPAN_STAT = False
PRUNE = False
MAX_NUM_HYPERPLANES = 300
PRUNE_PLOT = False
PRUNE_FILTERS = False

act_change_layers = {}
for i in range(1, max_index):
    act_change_layers[i] = []
act_change_layers_per_class = {}
for i in range(1, max_index):
    act_change_layers_per_class[i] = {}
#for characteristic
act_layers = {}
for i in range(1, max_index):
    act_layers[i] = []
act_layers_per_class = {}
for i in range(1, max_index):
    act_layers_per_class[i] = {}
#for ood
act_max = {}
for i in range(1, max_relu_index):
    act_max[i] = []
act_max_per_class = {}
for i in range(1, max_relu_index):
    act_max_per_class[i] = {}
act_min = {}
for i in range(1, max_relu_index):
    act_min[i] = []
act_min_per_class = {}
for i in range(1, max_relu_index):
    act_min_per_class[i] = {}

def select_layer(layer, dual_net, X, proj, norm_type, in_f, out_f, zsi, pca_var, phase=1, intermediate=False, zl=None, zu=None, y=None):
    global index
    if CH:
        if zl is None and zu is None:
            zl, zuq = zip(*[l.bounds() for l in dual_net])
            zl, zuq = sum(zl), sum(zuq)
            zua = 0
            if ('Conv' in str(dual_net[-1]) ) or ('Linear' in str(dual_net[-1]) ):
                zua = dual_net[-1].affine_bounds()    
            zu = torch.min(zuq, zua)
    if isinstance(layer, nn.Linear): 
        return DualLinear(layer, out_f, pca_var, phase, zl, zu)
    elif isinstance(layer, nn.Conv2d):
        return DualConv2d(layer, out_f, pca_var, phase, intermediate, zl, zu)
    elif isinstance(layer, nn.ReLU):
        #simple bounds addition
        if not CH:
            if zl is None and zu is None:
                zl, zu = zip(*[l.bounds() for l in dual_net])
                zl, zu = sum(zl), sum(zu)
            
            szl, szu = dual_net[0].lb_simple, dual_net[0].ub_simple
            for l in dual_net[1:]:
                szl, szu = l.simp_bounds(szl, szu)
            zl = torch.max(zl, szl)
            zu = torch.min(zu, szu)
        if zl is None or zu is None: 
            raise ValueError("Must either provide both l,u bounds or neither.")
        # vector of activation indices spanning zero 
        I = ((zu > 0).detach() * (zl < 0).detach())
        if proj is not None and (norm_type=='l1_median' or norm_type=='l2_normal') and I.sum().item() > proj:
            return DualReLUProj(zl, zu, proj, pca_var)
        else:
            return DualReLU(zl, zu, pca_var, intermediate, y)
    elif 'Flatten' in (str(layer.__class__.__name__)): 
        return DualReshape(in_f, out_f, pca_var)
    elif isinstance(layer, Dense):
        return DualDense(layer, dual_net, out_f, pca_var, phase, intermediate, zl, zu)
    elif isinstance(layer, nn.BatchNorm2d):
        return DualBatchNorm2d(layer, zsi, out_f, pca_var)
    else:
        print(layer)
        raise ValueError("No module for layer {}".format(str(layer.__class__.__name__)))

def batch(A, n): 
    return A.view(n, -1, *A.size()[1:])
def unbatch(A): 
    return A.view(-1, *A.size()[2:])

class DualLinear(DualLayer): 
    def __init__(self, layer, out_features, pca_var, phase, zl=None, zu=None): 
        super(DualLinear, self).__init__()
        if not isinstance(layer, nn.Linear):
            raise ValueError("Expected nn.Linear input.")
        self.layer = layer
        if layer.bias is None: 
            self.bias = None
        else: 
            self.bias = [full_bias(layer, out_features[1:])]
        self.pca_var = pca_var
        self.phase = phase
        if CH:
            self.zl, self.zu = zl, zu
            self.C, self.M, self.C1= [], [], []
            W = self.layer.weight
            Ctotal, Mtotal, C1int = [], [], []
            z0_t = F.linear(zu[0], self.layer.weight) + (self.layer.bias if self.layer.bias is not None else 0)
            print('constructor : ', zl.shape)
            for i in range(min(MAX_NUM_HYPERPLANES, zl.shape[1])):
                zi = zu[0]
                Z = zi.unsqueeze(0)
                for j in range(zl.shape[1]):
                    if j!=i:
                        zi = zu[0]
                        zi[j] = zl[0][j]
                        Z = torch.cat([Z, zi.unsqueeze(0)])
                zi_t = F.linear(Z, self.layer.weight) + (self.layer.bias if self.layer.bias is not None else 0)
                #check that there is no sign change in zi_t-z0_t
                del_t = zi_t-z0_t
                if not ((torch.sum(del_t.clamp(min=0))==0) or (torch.sum(del_t.clamp(max=0))==0)): 
                    zi_t = zi_t.clamp(min=0)
                    z0t = z0_t.clamp(min=0)
                    del_t = zi_t-z0t
                Mtotal.append(del_t)
                C1int.append(z0_t)
                Ctotal.append(zi_t)
            self.M.append(Mtotal)
            self.C1.append(C1int)
            self.C.append(Ctotal)

    def apply(self, dual_layer):
        if self.bias is not None: 
            self.bias.append(dual_layer(*self.bias, W=True))
        if CH and (not isinstance(dual_layer, nn.ReLU)):
            if self.M is not []:
                Mtotal = []
                for i, m in enumerate(self.M[-1]):
                    Mint = []
                    for ii in range(len(m)):
                        Mint.append(dual_layer(m[ii], W=True))
                    Mtotal.append(Mint)
                self.M.append(Mtotal)
                
                Ctotal = []
                for i, c in enumerate(self.C[-1]):
                    Cint = []
                    for ii in range(len(c)):
                        Cint.append(dual_layer(c[ii], W=True))
                    Ctotal.append(Cint)
                self.C.append(Ctotal)                

                C1int = []
                for c1 in self.C1[-1]:
                    C1int.append(dual_layer(c1, W=True))
                self.C1.append(C1int)
                
    def bounds(self, network=None):
        if self.bias is None: 
            return 0,0
        else: 
            if network is None: 
                b = self.bias[-1]
            else:
                b = network(self.bias[0])
            if b is None:
                return 0,0
            return b,b

    def affine_bounds(self):
        # M is a vector of dim zout #(2^(n-1)) such vectors #n such vectors
        # C is a vector of dim zout 
        # alpha1, 2 are vectorsof dim zin, total combinations
        Cprop, Mprop, C1prop = self.C[-1], self.M[-1], self.C1[-1]
        zu = None
        print('mprop ', len(Mprop))
        for i in range(len(Mprop)):
            zuint = C1prop[0]
            for j in range(len(Mprop[i])):
                if ((torch.sum(Mprop[i][j].clamp(min=0))==0) or (torch.sum(Mprop[i][j].clamp(max=0))==0)): 
                    zuint = torch.max(zuint, zuint + Mprop[i][j])
                else:
                    Cnew = Cprop[i][j].clamp(min=0)
                    del_t = Cnew - C1prop[0].clamp(min=0)
                    zuint = torch.max(zuint, zuint + del_t.cuda())
            if zu is None and zuint is not None:
                zu = zuint
            elif zu is not None and zuint is not None:
                zuint = zuint
                if torch.all(torch.max(zu, zuint)==zuint):
                    zu = zuint
            elif zuint is None:
                zu = torch.ones(Mprop[i][j].shape)*1000000
        return zu.cuda()

    def objective(self, *nus): 
        if self.bias is None: 
            return 0
        else:
            nu = nus[-2]
            nu = nu.view(nu.size(0), nu.size(1), -1)
            return -nu.matmul(self.bias[0].view(-1))

    def forward(self, *xs, W=False, simple=False):
        if simple:
            return F.linear(xs[0][0], self.layer.weight)
        global dict_activations, epoch
        x = xs[-1]
        if x is None: 
            return None
        if self.pca_var < 1.:
            if W==False and PRUNE:
                key = 1
                for i in range(1,x.dim()):
                    key = key*x.shape[i]
                if key not in dict_activations:
                    dict_activations[key] = [0]*key
                    epoch[key] = 0
            if self.phase==1:
                x = PCA(x, W, self.pca_var, ' Linear')
                if W==False:
                    key = 1
                    for i in range(1,x.dim()):
                        key = key*x.shape[i]
                    if key not in dict_activations:
                        dict_activations[key] = [0]*key
                        epoch[key] = 0
                    _, dict_activations[key] = get_PFA(x, dict_activations[key])
                    f = open('./res/'+str(key) + "_linear.txt","w")
                    simplejson.dump(dict_activations[key], f)
                    f.close()
                    epoch[key] = epoch[key] + 1 
                    if PRUNE_PLOT:
                        plt.plot(np.arange(key), dict_activations[key])  # arguments are passed to np.histogram
                        plt.title("Histogram of kept activations with for key " + str(key) + " after examples " + str(epoch[key]))
                        plt.gcf().savefig('./res/'+str(key) + "_linear.png")
                        plt.pause(0.2)
                        plt.close()
            elif self.phase==2:
                    key = 1
                    for i in range(1, x.dim()):
                        key = key*x.shape[i]
                    if W==False:
                        f = open('./res/'+str(key)+'_linear.txt',"r").read().split(", ")
                        act_matrix_indices = np.array([int(fff) for fff in f[0:len(f)-1]])
                        zero_ind = np.where(act_matrix_indices == 0)[0]
                        act_matrix = torch.ones((x.shape[0], key)).cuda()
                        for i in zero_ind:
                            act_matrix[:,i] = 0
                        xx = x.reshape(x.shape[0], -1) * act_matrix
                        del act_matrix
                        x = xx.reshape(x.shape)
        return F.linear(x, self.layer.weight)

    def simp_bounds(self, zl, zu):
        W_pos = self.layer.weight.clamp(min=0)
        W_neg = self.layer.weight.clamp(max=0)
        bias = 0
        if self.bias is not None: 
            bias = self.bias[0]
        l = F.linear(zl, W_pos) + F.linear(zu, W_neg) + bias
        u = F.linear(zu, W_pos) + F.linear(zl, W_neg) + bias
        return l, u

    def T(self, *xs):
        global dict_activations
        x = xs[-1]
        if x is None:
            return None
        out = F.linear(x, self.layer.weight.t())
        
        if PRUNE:
            key = 1
            for i in range(1, out.dim()):
                key = key*out.shape[i]
            if self.phase==2 and key in dict_activations and False:
                f = open('./res/'+str(key)+'_linear.txt',"r").read().split(", ")
                act_matrix_indices = np.array([int(fff) for fff in f[0:len(f)-1]])
                zero_ind = np.where(act_matrix_indices == 0)[0]
                act_matrix = torch.ones((out.shape[0],key)).cuda()
                for i in zero_ind:
                    act_matrix[:,i] = 0
                xx = out.reshape(out.shape[0],-1) * act_matrix
                del act_matrix
                out = xx.reshape(out.shape) 
        return out

# Convolutional helper functions to minibatch large inputs for CuDNN
def conv2d(x, *args, **kwargs): 
    """ Minibatched inputs to conv2d """
    i = 0
    out = []
    batch_size = 1000
    while i < x.size(0): 
        out.append(F.conv2d(x[i:min(i+batch_size, x.size(0))], *args, **kwargs))
        i += batch_size
    return torch.cat(out, 0)

def conv_transpose2d(x, *args, **kwargs):
    i = 0
    out = []
    batch_size = 1000
    while i < x.size(0): 
        out.append(F.conv_transpose2d(x[i:min(i+batch_size, x.size(0))], *args, **kwargs))
        i += batch_size
    return torch.cat(out, 0)

class DualConv2d(DualLinear): 
    def __init__(self, layer, out_features, pca_var, phase, intermediate, zl=None, zu=None): 
        super(DualLinear, self).__init__()
        if not isinstance(layer, nn.Conv2d):
            raise ValueError("Expected nn.Conv2d input.")
        self.layer = layer
        if layer.bias is None: 
            self.bias = None
        else: 
            self.bias = [full_bias(layer, out_features[1:]).contiguous()]
        self.pca_var = pca_var
        self.phase = phase
        self.intermediate = intermediate
        if CH:
            self.zl, self.zu = zl, zu
            self.M, self.C1= [], []
            W = self.layer.weight
            #change t here to get different Zouts 
            Mint, C1int = [], []
            z0_t = conv2d(zu[0].unsqueeze(0), self.layer.weight, stride=self.layer.stride, padding=self.layer.padding) 
            for i in range(zl[0].shape[0]):
                for j in range(zl[0].shape[1]):
                    for k in range(zl[0].shape[2]):
                        zi = zu[0]
                        zi[i, j, k] = zl[0, i, j, k]
                        zi_t = conv2d(zi.unsqueeze(0), self.layer.weight, stride=self.layer.stride, padding=self.layer.padding) 
                        del_t = zi_t-z0_t
                        #check that there is no sign change in zi_t-z0_t
                        if not ((torch.sum(del_t.clamp(min=0))==0) or (torch.sum(del_t.clamp(max=0))==0)): 
                            zit = zi_t.clamp(min=0)
                            z0t = z0_t.clamp(min=0)
                            del_t = zit-z0t
                        Mint.append(del_t)
                        C1int.append(z0_t)
            self.M.append(np.array(Mint))
            self.C1.append(np.array(C1int))

    def forward(self, *xs, W=False, simple=False): 
        if simple:
            return conv2d(xs[0][0], self.layer.weight, 
                       stride=self.layer.stride,
                       padding=self.layer.padding)
        global dict_activations, epoch, index
        x = xs[-1]
        if x is None: 
            return None
        if xs[-1].dim() == 5:  
            n = x.size(0)
            x = unbatch(x)
        if self.pca_var < 1.:
            if W==False and PRUNE:
                key = 1
                for i in range(1,x.dim()):
                    key = key*x.shape[i]
                if key not in dict_activations:
                    dict_activations[key] = [0]*key
                    epoch[key] = 0
            if self.phase==1:
                x = PCA(x, W, self.pca_var, 'Conv2D')
                if W==False and False:
                    _, dict_activations[key] = get_PFA(x, dict_activations[key])
                    f = open('./res/'+str(key) + "_conv2d.txt","w")
                    simplejson.dump(dict_activations[key], f)
                    f.close()
                    epoch[key] = epoch[key] + 1 
                    plt.plot(np.arange(key), dict_activations[key])  # arguments are passed to np.histogram
                    _ = plt.hist(dict_activations[key], bins=np.arange(key))  # arguments are passed to np.histogram
                    plt.title("Histogram of kept activations with for key " + str(key) + " after examples " + str(epoch[key]))
                    plt.gcf().savefig('./res/'+str(key) + "_conv2d.png")
                    plt.pause(0.2)
                    plt.close()
            elif self.phase==2:
                    key = 1
                    for i in range(1,x.dim()):
                        key = key*x.shape[i]
                    if W==False:
                        f = open('./res/'+str(key)+'_conv2d.txt',"r").read().split(", ")
                        act_matrix_indices = np.array([int(fff) for fff in f[0:len(f)-1]])
                        zero_ind = np.where(act_matrix_indices == 0)[0]
                        act_matrix = torch.ones((x.shape[0], key)).cuda()
                        for i in zero_ind:
                            act_matrix[:,i] = 0
                        xx = x.reshape(x.shape[0], -1) * act_matrix
                        del act_matrix
                        x = xx.reshape(x.shape)    
        out = conv2d(x, self.layer.weight, 
                       stride=self.layer.stride,
                       padding=self.layer.padding)
        if PRUNE_FILTERS:
            if out.shape==torch.Size([1, 16, 14, 14]):
                S = [0,2,5,6,7,9,10,11,12,13,14,15,1,3]
                for i in S:
                    out[:,i,:,:] = 0
            if out.shape==torch.Size([1, 32, 7, 7]):
                S = [2,7,8,9,10,11,12,13,14,16,19,21,22,24,28,30,20] #30
                for i in S:
                    out[:,i,:,:] = 0
        if xs[-1].dim() == 5:  
            out = batch(out, n)
        return out

    def affine_bounds(self):
        # M is a vector of dim zout #(2^(n-1)) such vectors #n such vectors
        # C is a vector of dim zout 
        # alpha1, 2 are vectorsof dim zin, total combinations
        Mprop, C1prop = self.M[-1], self.C1[-1]
        zu = None
        for i in range(Mprop.shape[0]):
            zuint = torch.max(Mprop[i]  + C1prop[i], C1prop[i])
            if zu is None:
                zu = zuint
            else:
                zu = torch.max(zu, zuint)                
        return zu

    def simp_bounds(self, zl, zu):
        W_pos = self.layer.weight.clamp(min=0)
        W_neg = self.layer.weight.clamp(max=0)
        bias = 0
        if self.bias is not None:
            bias = self.bias[0]
        l = conv2d(zl, W_pos, stride=self.layer.stride, padding=self.layer.padding) + conv2d(zu, W_neg, stride=self.layer.stride, padding=self.layer.padding) + bias
        u = conv2d(zu, W_pos, stride=self.layer.stride, padding=self.layer.padding) + conv2d(zl, W_neg, stride=self.layer.stride, padding=self.layer.padding) + bias
        return l, u        

    def T(self, *xs): 
        global dict_activations
        x = xs[-1]
        if x is None:
            return None
        if xs[-1].dim() == 5:  
            n = x.size(0)
            x = unbatch(x)
        out = conv_transpose2d(x, self.layer.weight, 
                                 stride=self.layer.stride,
                                 padding=self.layer.padding)
        if PRUNE:
            key = 1
            for i in range(1,out.dim()):
                key = key*out.shape[i]
            if self.phase==2 and key in dict_activations and key==3136 and False:
                print('yes conv2d '+str(key))
                f = open('./res/'+str(key)+'_conv2d.txt',"r").read().split(", ")
                act_matrix_indices = np.array([int(fff) for fff in f[0:len(f)-1]])
                zero_ind = np.where(act_matrix_indices == 0)[0]
                act_matrix = torch.ones((out.shape[0],key)).cuda()
                for i in zero_ind:
                    act_matrix[:,i] = 0
                xx = out.reshape(out.shape[0],-1) * act_matrix
                del act_matrix
                out = xx.reshape(out.shape) 
        if xs[-1].dim() == 5:  
            out = batch(out, n)
        return out

class DualReshape(DualLayer): 
    def __init__(self, in_f, out_f, pca_var): 
        super(DualReshape, self).__init__()
        self.in_f = in_f[1:]
        self.out_f = out_f[1:]
        self.pca_var = pca_var

    def forward(self, *xs, W=False, simple=False):
        if simple:
            x = xs[0][0]
            shape = x.size()[:-len(self.in_f)] + self.out_f
            return x.view(shape) 
        x = xs[-1]
        if x is None: 
            return None
        shape = x.size()[:-len(self.in_f)] + self.out_f
        return x.view(shape)

    def T(self, *xs): 
        x = xs[-1]
        if x is None: 
            return None
        shape = x.size()[:-len(self.out_f)] + self.in_f
        return x.view(shape)

    def apply(self, dual_layer):
        pass

    def bounds(self, network=None): 
        return 0,0

    def simp_bounds(self, zl, zu):
        shape = zl.size()[:-len(self.in_f)] + self.out_f
        return zl.view(shape), zu.view(shape)        

    def objective(self, *nus): 
        return 0

class DualReLU(DualLayer): 
    def __init__(self, zl, zu, pca_var, intermediate, y): 
        super(DualReLU, self).__init__()
        global relu_index, max_relu_index, max_act_per_class, act_max

        d = (zl >= 0).detach().type_as(zl) #[BS,1,28,28]
        I = ((zu > 0).detach() * (zl < 0).detach())
        if I.sum().item() > 0:
            d[I] += zu[I]/(zu[I] - zl[I])

        #this is for OOD analysis
        if OOD:
            relu_index += 1
            if relu_index == max_relu_index:
                relu_index = 1
            if act_max[relu_index] == []:
                act_max[relu_index] = zu.clamp(min=0).clone().detach().cpu().numpy()
                act_min[relu_index] = zl.clamp(min=0).clone().detach().cpu().numpy()
            else:
                act_max[relu_index] = np.maximum(act_max[relu_index], zu.clamp(min=0).clone().detach().cpu().numpy())
                act_min[relu_index] = np.minimum(act_min[relu_index], zl.clamp(min=0).clone().detach().cpu().numpy())
            np.save('./exp_res/OOD/max_act_'+str(relu_index)+'.npy', act_max[relu_index])                
            np.save('./exp_res/OOD/min_act_'+str(relu_index)+'.npy', act_min[relu_index])                
            
            if y.item() not in act_max_per_class[relu_index]:
                act_max_per_class[relu_index][y.item()] = zu.clamp(min=0).clone().detach().cpu().numpy()
                act_min_per_class[relu_index][y.item()] = zl.clamp(min=0).clone().detach().cpu().numpy()
            else:
                act_max_per_class[relu_index][y.item()] = np.maximum(act_max_per_class[relu_index][y.item()], zu.clamp(min=0).clone().detach().cpu().numpy())
                act_min_per_class[relu_index][y.item()] = np.minimum(act_max_per_class[relu_index][y.item()], zl.clamp(min=0).clone().detach().cpu().numpy())
            np.save('./exp_res/OOD/max_act_per_class'+str(relu_index)+'.npy', act_max_per_class[relu_index])
            np.save('./exp_res/OOD/min_act_per_class'+str(relu_index)+'.npy', act_min_per_class[relu_index])
            
        n = d[0].numel()
        if I.sum().item() > 0: 
            self.I_empty = False
            self.I_ind = I.view(-1,n).nonzero()

            self.nus = [zl.new(I.sum().item(), n).zero_()]
            self.nus[-1].scatter_(1, self.I_ind[:,1,None], d[I][:,None])
            self.nus[-1] = self.nus[-1].view(-1, *(d.size()[1:])) #[non-zero d,1,28,28]
            self.I_collapse = zl.new(self.I_ind.size(0),zl.size(0)).zero_()
            self.I_collapse.scatter_(1, self.I_ind[:,0][:,None], 1)
        else: 
            self.I_empty = True

        self.d = d
        self.I = I
        self.zl = zl
        self.zu = zu
        self.pca_var = pca_var
        self.intermediate = intermediate
        if intermediate or True:
            self.y = y
        self.rd = torch.tensor(0., device='cuda', requires_grad=True)
        
    def apply(self, dual_layer): 
        global index, max_index
        if self.I_empty:
            if str(dual_layer) == 'DualReLU()': 
                index += 1
                if index == max_index:
                    index = 1
            return
        if isinstance(dual_layer, DualReLU): 
            self.nus.append(dual_layer(*self.nus, I_ind=self.I_ind, W=False))
        else: 
            self.nus.append(dual_layer(*self.nus, W=False))

    def bounds(self, network=None): #[non-zero I,1,28,28]
        if self.I_empty: 
            return 0,0
        if network is None: 
            nu = self.nus[-1]
        else:
            nu = network(self.nus[0])
        if nu is None: 
            return 0,0
        size = nu.size()
        nu = nu.view(nu.size(0), -1)
        zlI = self.zl[self.I]
        zl = (zlI * (-nu.t()).clamp(min=0)).mm(self.I_collapse).t().contiguous()
        zu = -(zlI * nu.t().clamp(min=0)).mm(self.I_collapse).t().contiguous()

        zl = zl.view(-1, *(size[1:]))
        zu = zu.view(-1, *(size[1:]))
        return zl,zu

    def objective(self, *nus): 
        nu_prev = nus[-1]
        if self.I_empty: 
            return 0
        n = nu_prev.size(0)
        nu = nu_prev.view(n, nu_prev.size(1), -1)
        zl = self.zl.view(n, -1)
        I = self.I.view(n, -1)
        return (nu.clamp(min=0)*zl.unsqueeze(1)).matmul(I.type_as(nu).unsqueeze(2)).squeeze(2)

    def simp_bounds(self, zl, zu):
        return zl.clamp(min=0), zu.clamp(min=0)

    def forward(self, *xs, I_ind=None, W=False, simple=False): 
        if simple:
            return xs[0][0].clamp(min=0)
        global index, act_change_layers, act_change_layers_per_class, max_index
        index += 1
        if index == max_index:
            index = 1
        x = xs[-1]
        if x is None:
            return None
        if self.d.is_cuda:
            d = self.d.cuda(device=x.get_device())
        else:
            d = self.d
        if x.dim() > d.dim():
            d = d.unsqueeze(1)

        if self.intermediate:
            dd = d.clone().detach().cpu().numpy()
            np.save('./exp_res/charac/activations_'+str(index)+'.npy', dd)
            if act_layers[index] == []:
                act_layers[index] = dd
            else:
                act_layers[index] += dd
            np.save('./exp_res/charac/l1_act_'+str(index)+'.npy', act_layers[index])                
            
            if self.y.item() not in act_layers_per_class[index]:
                act_layers_per_class[index][self.y.item()] = dd
            else:
                act_layers_per_class[index][self.y.item()] += dd
            np.save('./exp_res/charac/l1_act_per_class'+str(index)+'.npy', act_layers_per_class[index])
        else:
            #this is for stability analysis
            if STABILITY:
                dd_cpu = d.clone().detach().cpu().numpy()
                try:
                    dd_clean = np.load('./exp_res/charac/activations_'+str(index)+'.npy')
                except Exception as e:
                    dd_clean = np.zeros(dd_cpu.shape)
                if d.dim()==4:
                    diff_vector = np.reshape(np.absolute(dd_clean-dd_cpu), (dd_cpu.shape[0], dd_cpu.shape[1], -1))
                elif d.dim()==5:
                    diff_vector = np.reshape(np.absolute(dd_clean-dd_cpu), (dd_cpu.shape[0], dd_cpu.shape[1], dd_cpu.shape[2], -1))
                else:
                    diff_vector = np.reshape(np.absolute(dd_clean-dd_cpu), -1)
                    
                if act_change_layers[index] == []:
                    act_change_layers[index] = diff_vector
                else:
                    act_change_layers[index] += diff_vector
                np.save('./exp_res/stability/l1_act_change_filter_'+str(index)+'.npy', act_change_layers[index])                

                if self.y.item() not in act_change_layers_per_class[index]:
                    act_change_layers_per_class[index][self.y.item()] = diff_vector
                else:
                    act_change_layers_per_class[index][self.y.item()] += diff_vector
                np.save('./exp_res/stability/l1_act_change_filter_per_class'+str(index)+'.npy', act_change_layers_per_class[index])
                
        if I_ind is not None: 
            I_ind = I_ind.to(dtype=torch.long, device=x.device)
            dd = d[I_ind[:,0]]
            if ZERO_SPAN_STAT: 
                if dd.dim()==4:
                    print('I_ind=', I_ind.shape, ' image=', dd[0,0,:,:].shape)
                    sns.heatmap(dd[-1,-1,:,:].clone().detach().cpu().numpy(), linewidth=0.5)
                elif dd.dim()==2:
                    print('I_ind=', I_ind.shape, ' image=', dd.shape)
                    sns.heatmap(dd.clone().detach().cpu().numpy(), linewidth=0.5)                
                plt.title('layer shape: '+str(d.shape) + ' ' + '   # of 0 spanning activations: '+str(I_ind.shape))
                plt.pause(0.5)
                plt.savefig('./activations_'+str(index)+'.png', dpi=300, bbox_inches='tight')
                plt.close()
            return dd*x
        else:
            return d*x

    def T(self, *xs): 
        return self(*xs)


class DualReLUProj(DualReLU): 
    def __init__(self, zl, zu, k, pca_var): 
        DualLayer.__init__(self)
        d = (zl >= 0).detach().type_as(zl) #[BS,1,28,28]
        I = ((zu > 0).detach() * (zl < 0).detach())
        if I.sum().item() > 0:
            d[I] += zu[I]/(zu[I] - zl[I])

        n = I.size(0)

        self.d = d
        self.I = I
        self.zl = zl
        self.zu = zu
        self.pca_var = pca_var

        if I.sum().item() == 0: 
            warnings.warn('ReLU projection has no origin crossing activations')
            self.I_empty = True
            return
        else:
            self.I_empty = False

        nu = zl.new(n, k, *(d.size()[1:])).zero_() #[BS,k,1,28,28]
        nu_one = zl.new(n, *(d.size()[1:])).zero_() #[BS,1,28,28]
        if  I.sum() > 0: 
            nu[I.unsqueeze(1).expand_as(nu)] = nu.new(I.sum().item()*k).cauchy_()
            nu_one[I] = 1
        nu = zl.unsqueeze(1)*nu
        nu_one = zl*nu_one

        self.nus = [d.unsqueeze(1)*nu]
        self.nu_ones = [d*nu_one]

    def apply(self, dual_layer): 
        if self.I_empty: 
            return
        self.nus.append(dual_layer(*self.nus, W=False))
        self.nu_ones.append(dual_layer(*self.nu_ones, W=False))

    def bounds(self, network=None): 
        if self.I_empty: 
            return 0,0

        if network is None: 
            nu = self.nus[-1]
            no = self.nu_ones[-1]
        else: 
            nu = network(self.nus[0])
            no = network(self.nu_ones[0])

        n = torch.median(nu.abs(), 1)[0]

        # From notes: 
        # \sum_i l_i[nu_i]_+ \approx (-n + no)/2
        # which is the negative of the term for the upper bound
        # for the lower bound, use -nu and negate the output, so 
        # (n - no)/2 since the no term flips twice and the l1 term
        # flips only once. 
        zl = (-n - no)/2
        zu = (n - no)/2
        return zl,zu

class DualDense(DualLayer): 
    def __init__(self, dense, net, out_features, pca_var, phase, intermediate, zl, zu): 
        super(DualDense, self).__init__()
        self.duals = nn.ModuleList([])
        for i,W in enumerate(dense.Ws): 
            if isinstance(W, nn.Conv2d):
                dual_layer = DualConv2d(W, out_features, pca_var, phase, intermediate, zl, zu)
            elif isinstance(W, nn.Linear): 
                dual_layer = DualLinear(W, out_features, pca_var, phase, zl, zu)
            elif isinstance(W, nn.Sequential) and len(W) == 0: 
                dual_layer = Identity()
            elif W is None:
                dual_layer = None
            else:
                print(W)
                raise ValueError("Don't know how to parse dense structure")
            self.duals.append(dual_layer)

            if i < len(dense.Ws)-1 and W is not None: 
                idx = i-len(dense.Ws)+1
                # dual_ts needs to be len(dense.Ws)-i long
                net[idx].dual_ts = nn.ModuleList([dual_layer] + [None]*(len(dense.Ws)-i-len(net[idx].dual_ts)-1) + list(net[idx].dual_ts))
        self.pca_var=pca_var
        self.dual_ts = nn.ModuleList([self.duals[-1]])

    def forward(self, *xs, W=False): 
        duals = list(self.duals)[-min(len(xs),len(self.duals)):]
        if all(W is None for W in duals): 
            return None
        # recursively apply the dense sub-layers
        out = [W(*xs[:i+1]) 
            for i,W in zip(range(-len(duals) + len(xs), len(xs)),
                duals) if W is not None]
        
        # remove the non applicable outputs
        out = [o for o in out if o is not None]

        # if no applicable outputs, return None
        if len(out) == 0: 
            return None
        # otherwise, return the sum of the outputs
        return sum(o for o in out if o is not None)

    def T(self, *xs): 
        dual_ts = list(self.dual_ts)[-min(len(xs),len(self.dual_ts)):]
        if all(W is None for W in dual_ts): 
            return None

        # recursively apply the dense sub-layers
        out = [W.T(*xs[:i+1]) 
            for i,W in zip(range(-len(dual_ts) + len(xs), len(xs)),
                dual_ts) if W is not None]
        # remove the non applicable outputs
        out = [o for o in out if o is not None]

        # if no applicable outputs, return None
        if len(out) == 0: 
            return None
        # otherwise, return the sum of the outputs
        return sum(o for o in out if o is not None)


    def apply(self, dual_layer): 
        for W in self.duals: 
            if W is not None: 
                W.apply(dual_layer)

    def bounds(self, network=None): 
        fvals = list(W.bounds(network=network) for W in self.duals 
                        if W is not None)
        l,u = zip(*fvals)
        return sum(l), sum(u)

    def objective(self, *nus): 
        fvals = list(W.objective(*nus) for W in self.duals if W is not None)
        return sum(fvals)

class DualBatchNorm2d(DualLayer): 
    def __init__(self, layer, minibatch, out_features, pca_var): 
        if layer.training: 
            minibatch = minibatch.data.transpose(0,1).contiguous()
            minibatch = minibatch.view(minibatch.size(0), -1)
            mu = minibatch.mean(1)
            var = minibatch.var(1)
        else: 
            mu = layer.running_mean
            var = layer.running_var
        
        eps = layer.eps

        weight = layer.weight
        bias = layer.bias
        denom = torch.sqrt(var + eps)

        self.D = (weight/denom).unsqueeze(1).unsqueeze(2)
        self.ds = [((bias - weight*mu/denom).unsqueeze(1).unsqueeze
            (2)).expand(out_features[1:]).contiguous()]
        self.pca_var = pca_var

    def forward(self, *xs, W=False): 
        x = xs[-1]
        if x is None:
            return None
        return self.D*x

    def T(self, *xs): 
        if x is None: 
            return None
        return self(*xs)

    def apply(self, dual_layer): 
        self.ds.append(dual_layer(*self.ds))

    def bounds(self, network=None):
        if network is None:
            d = self.ds[-1]
        else:
            d = network(self.ds[0])
        return d, d

    def objective(self, *nus): 
        nu = nus[-2]
        d = self.ds[0].view(-1)
        nu = nu.view(nu.size(0), nu.size(1), -1)
        return -nu.matmul(d)

class Identity(DualLayer): 
    def forward(self, *xs, W=False): 
        return xs[-1]

    def T(self, *xs): 
        return xs[-1]

    def apply(self, dual_layer): 
        pass

    def bounds(self, network=None):
        return 0,0

    def objective(self, *nus): 
        return 0
