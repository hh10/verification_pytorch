import torch.nn as nn
import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA as PCAsk
from sklearn.cluster import KMeans
from collections import defaultdict
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import StandardScaler

PLOT = True
KK = 0
###########################################
# Helper function to extract fully        #
# shaped bias terms                       #
###########################################

def full_bias(l, n=None): 
    # expands the bias to the proper size. For convolutional layers, a full
    # output dimension of n must be specified. 
    if isinstance(l, nn.Linear): 
        return l.bias.view(1,-1)
    elif isinstance(l, nn.Conv2d): 
        if n is None: 
            raise ValueError("Need to pass n=<output dimension>")
        b = l.bias.unsqueeze(1).unsqueeze(2)
        if isinstance(n, int): 
            k = int((n/(b.numel()))**0.5)
            return b.expand(1,b.numel(),k,k).contiguous().view(1,-1)
        else: 
            return b.expand(1,*n)
    elif isinstance(l, Dense): 
        return sum(full_bias(layer, n=n) for layer in l.Ws if layer is not None)
    elif isinstance(l, nn.Sequential) and len(l) == 0: 
        return 0
    else:
        raise ValueError("Full bias can't be formed for given layer.")

###########################################
# Sequential models with skip connections #
###########################################

class DenseSequential(nn.Sequential): 
    def forward(self, x):
        xs = [x]
        for module in self._modules.values():
            if 'Dense' in type(module).__name__:
                xs.append(module(*xs))
            else:
                xs.append(module(xs[-1]))
        return xs[-1]

class Dense(nn.Module): 
    def __init__(self, *Ws): 
        super(Dense, self).__init__()
        self.Ws = nn.ModuleList(list(Ws))
        if len(Ws) > 0 and hasattr(Ws[0], 'out_features'): 
            self.out_features = Ws[0].out_features

    def forward(self, *xs): 
        xs = xs[-len(self.Ws):]
        out = sum(W(x) for x,W in zip(xs, self.Ws) if W is not None)
        return out

#######################################
# Epsilon for high probability bounds #
#######################################
import numpy as np
import time

def GR(epsilon): 
    return (epsilon**2)/(-0.5*np.log(1+(2/np.pi*np.log(1+epsilon))**2) 
                        + 2/np.pi*np.arctan(2/np.pi*np.log(1+epsilon))*np.log(1+epsilon))

def GL(epsilon): 
    return (epsilon**2)/(-0.5*np.log(1+(2/np.pi*np.log(1-epsilon))**2) 
                        + 2/np.pi*np.arctan(2/np.pi*np.log(1-epsilon))*np.log(1-epsilon))

def p_upper(epsilon, k): 
    return np.exp(-k*(epsilon**2)/GR(epsilon))

def p_lower(epsilon, k): 
    return np.exp(-k*(epsilon**2)/GL(epsilon))

def epsilon_from_model(model, X, k, delta, m): 
    if k is None or m is None: 
        raise ValueError("k and m must not be None. ")
    if delta is None: 
        print('No delta specified, not using probabilistic bounds.')
        return 0
        
    X = X[0].unsqueeze(0)
    out_features = []
    for l in model: 
        X = l(X)
        if isinstance(l, (nn.Linear, nn.Conv2d)): 
            out_features.append(X.numel())

    num_est = sum(n for n in out_features[:-1] if k*m < n)

    num_est += sum(n*i for i,n in enumerate(out_features[:-1]) if k*m < n)
    print(num_est)

    sub_delta = (delta/num_est)**(1/m)
    l1_eps = get_epsilon(sub_delta, k)

    if num_est == 0: 
        return 0
    if l1_eps > 1: 
        raise ValueError('Delta too large / k too small to get probabilistic bound')
    return l1_eps

def get_epsilon(delta, k, alpha=1e-2): 
    """ Determine the epsilon for which the estimate is accurate
    with probability >(1-delta) and k projection dimensions. """
    epsilon = 0.001
    # probability of incorrect bound 
    start_time = time.time()
    p_max = max(p_upper(epsilon, k), p_lower(epsilon,k))
    while p_max > delta: 
        epsilon *= (1+alpha)
        p_max = max(p_upper(epsilon, k), p_lower(epsilon,k))
    if epsilon > 1: 
        raise ValueError('Delta too large / k too small to get probabilistic bound (epsilon > 1)')
    return epsilon

def get_PCAk(S, perc_var, tell=False):
    var_cons = 0.
    var_total = torch.sum(S)
    for i in range(S.shape[0]):
        var_cons = var_cons + S[i]
        if (var_cons/var_total > perc_var):
            if tell:
                print('Considering ',i+1,' PCA components', var_cons.item(), var_total.item(), (var_cons/var_total).item(), perc_var, S.shape)
            return i+1
    if tell:
        print('Considering all PCA components', var_cons.item(), var_total.item(), (var_cons/var_total).item(), perc_var, S.shape)    
    return S.shape[0]

class PFA(object):
    def __init__(self, n_features, q=None):
        self.q = q
        self.n_features = n_features

    def fit(self, X):
        if not self.q:
            self.q = X.shape[1]
        sc = StandardScaler()
        X = sc.fit_transform(X)
        pca = PCAsk(n_components=self.q).fit(X)
        A_q = pca.components_.T

        kmeans = KMeans(n_clusters=self.n_features).fit(A_q)
        clusters = kmeans.predict(A_q)
        cluster_centers = kmeans.cluster_centers_
        dists = defaultdict(list)
        for i, c in enumerate(clusters):
            dist = euclidean_distances([A_q[i, :]], [cluster_centers[c, :]])[0][0]
            dists[c].append((i, dist))
        self.indices_ = [sorted(f, key=lambda x: x[1])[0][0] for f in dists.values()]
        self.features_ = X[:, self.indices_]

def get_PFA(X, list_activations):
    global KK
    X_pca =  X.reshape(X.shape[0], -1) 
    print('components : ', KK)
    pfa = PFA(n_features=KK, q=min(X_pca.shape[0], X_pca.shape[1]))
    pfa.fit(X_pca.detach().cpu().numpy())
    del X_pca
    # To get the column indices of the kept features
    column_indices = pfa.indices_
    for i in column_indices:
        list_activations[i] = list_activations[i]+1 
    return column_indices, list_activations

def PCA(x, W, pca_var=1., text="", normalize=False, dict_activations=None):
    global KK
    x_pca = x
    if W==False:
        if normalize:
            x_mean = torch.mean(x,0)
            x_pca = x - x_mean.expand_as(x)
        if x_pca.dim()>2:
            x_pca = x_pca.reshape(x.shape[0], -1)
        assert x_pca.dim()==2
        Ux,Sx,Vx = torch.svd(x_pca)
        KK = get_PCAk(Sx, pca_var, False)
        print(KK)
        x_pcar = torch.mm(Ux[:,:KK], torch.mm(torch.diag(Sx[:KK]), Vx.t()[:KK,:]))
        x_pcar = x_pcar.reshape(*x.shape)
        assert x.shape == x_pcar.shape
        return Variable(x_pcar, requires_grad=True)
    else:
        return x

#function to find the l-inf ball from the wasserstein epsilon
def get_WtoLINF_eps(X, eps, M, fig=None):
    if PLOT:
        if fig is None:
            fig = plt.figure()
    Z_min, Z_max = X.new_zeros((*X.shape)), X.new_zeros((*X.shape))
    N = X.shape[-1] * X.shape[-1]
    
    for I in range(X.shape[0]):
        epsilon = eps
        x = X[I,:,:,:]
        
        P_min, P_max = torch.zeros((N, N)).cuda(), torch.zeros((N, N)).cuda()
        for i in range(N):
            P_max[:,i] = x.view(-1,).cpu()
            sub = ((P_max[:,i]*M[:,i].cuda()) >= epsilon*torch.ones(N,).cuda())
            P_max[sub, i] = epsilon/M[sub, i].cuda()
            P_max[i,i] = x.view(-1,).cpu()[i]
        z_max = torch.sum(P_max, dim=0).clamp(max=1).float().cuda() 
        Z_max[I,:,:,:] = z_max.view(*X.size()[1:])
        
        epsilon = eps*0.1
        Mmin = M.clone() 
        for j in range(N):
            P_min[j,:] = z_max.cpu()
            sub = ((P_min[j,:]*M[j,:].cuda()) >= epsilon*torch.ones(N,).cuda())
            Mmin[j, j] = 10
            P_min[j, sub] = epsilon/Mmin[j, sub].cuda()
            P_min[j,j] = (torch.sum(P_min[j,sub]) - epsilon/Mmin[j,j]).clamp(min=0.)
        z_min = (x.view(-1,) - torch.sum(P_min, dim=1)).float().cuda() # 
        Z_min[I,:,:,:] = z_min.view(*X.size()[1:])
        del Mmin
        
        if PLOT:
            columns, rows = 3, 1
            fig.suptitle('W_to_Lp')
            fig.add_subplot(rows, columns, 1)       
            plt.imshow(X[I,0,:,:].detach().cpu().numpy(), cmap='gray', vmin=0, vmax=1)
            fig.add_subplot(rows, columns, 2)       
            plt.imshow(Z_max[I,0,:,:].detach().cpu().numpy(), cmap='gray', vmin=0, vmax=1)
            fig.add_subplot(rows, columns, 3)       
            plt.imshow(Z_min[I,0,:,:].detach().cpu().numpy(), cmap='gray', vmin=0, vmax=1)
            plt.draw()
            plt.pause(1)
    if PLOT:
        plt.close()
    return Z_min, Z_max


def get_correlation(X, transform=None, extent=3):
    if transform is None:
        return
    BS = X.shape[0]
    n = X.shape[-1]
    N = X[0].numel() #NxN = 28*28
    A, cov_A = torch.zeros((BS,N,N)), torch.zeros((BS,N,N))
    for k in range(1,extent):
        Xp, Xn = torch.zeros(X.size()), torch.zeros(X.size())
        Xp[:,:,:n-k,:] = X[:,:,k:,:]
        Xn[:,:,k:,:] = X[:,:,:n-k,:]        
        Xp_vector, Xn_vector = Xp.view(BS,-1), Xn.view(BS,-1)
        for i in range(N):
            for j in range(BS):
                if Xp_vector[j,i]!=0:
                    A[j,i,:] += Xp_vector[j,:]
                    A[j,:,i] += Xp_vector[j,:]
                if Xn_vector[j,i]!=0:
                    A[j,i,:] += Xn_vector[j,:]
                    A[j,:,i] += Xn_vector[j,:]
    A_mean = torch.sum(A, dim=0)/extent
    for j in range(BS):
        cov_A += ((A[j,:,:] - A_mean)**2)/extent

    if PLOT:
        fig = plt.figure()
        fig.suptitle('Correlation')
        fig.add_subplot(1, 4, 1)       
        plt.imshow(Xp_vector.view(n,n).detach().cpu().numpy(), cmap='gray', vmin=0, vmax=1)
        fig.add_subplot(1, 4, 2)       
        plt.imshow(Xn_vector.view(n,n).detach().cpu().numpy(), cmap='gray', vmin=0, vmax=1)
        fig.add_subplot(1, 4, 3)       
        plt.imshow(X[0,0,:,:].detach().cpu().numpy(), cmap='gray', vmin=0, vmax=1)
        fig.add_subplot(1, 4, 4)       
        plt.imshow(cov_A.squeeze(0).detach().cpu().numpy(), cmap='gray', vmin=0, vmax=1)
        plt.draw()
        plt.pause(0.1)
        xx=input()
        plt.close()
    print(cov_A.max(), cov_A.min(), cov_A.shape)
    return cov_A.squeeze().cuda()