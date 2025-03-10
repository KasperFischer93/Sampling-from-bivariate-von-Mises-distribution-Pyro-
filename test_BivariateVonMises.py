# Parameter estimation / test for pyro-dev
import math
import torch
import pyro
from torch.distributions import constraints
from torch.distributions.utils import broadcast_all
from pyro.distributions import *
from torch.autograd import Variable

import os
import warnings
warnings.filterwarnings("ignore")
from torch.distributions import *
from torch import optim

from BivariateVonMises import BivariateVonMises

def circ_mean(theta):
    C = theta.cos().mean(); S = theta.sin().mean()
    return torch.atan2(S, C)

def moments(theta1, theta2, mean1, mean2):
    S1 = (torch.sin(theta1 - mean1)**2).mean()
    S2 = (torch.sin(theta2 - mean2)**2).mean()
    S12 = (torch.sin(theta1 - mean1) * torch.sin(theta2 - mean2)).mean()
    return torch.tensor([S2, S1, S12]) / (S1*S2 - S12**2)


def _fit_params_from_data(phi, psi, n_iter):

    loc = torch.tensor([circ_mean(phi), circ_mean(psi)])
    w = moments(phi, psi, loc[0], loc[1])
    w.requires_grad = True

    lr = 1e-5; n_iter=50
    bfgs = optim.LBFGS([w], lr=lr)
    def bfgs_closure():
        bfgs.zero_grad()
        obj = -BivariateVonMises(loc[0], loc[1], 
                                 w[0], w[1], w[2]).log_prob(phi, psi).sum()
        obj.backward()
        return obj

    for i in range(n_iter):
        bfgs.step(bfgs_closure)
        
    return loc, w.detach()


def _test_fit(loc_true, w_true, n_samples=int(1e6), n_iter=50):
    bvm = BivariateVonMises(loc_true[0], loc_true[1], w_true[0], w_true[1], w_true[2])
    phi, psi = bvm.sample(n_samples)
    loc_est, w_est = _fit_params_from_data(phi, psi, n_iter)
    print(f"loc_est {loc_est}")
    print(f"w_est {w_est}")
    print( abs(loc_est - loc_true) < 0.1  )
    print( abs( w_est - w_true) < w_true * 0.1 )
    #assert all( abs(loc_est - loc_true) < 0.1 )
    #assert all( abs( w_est - w_true) < w_true * 0.1 )

loc_true = torch.tensor([0., 0.])
w_true = torch.tensor([5000., 700., 500.])
print(f"loc true: {loc_true}")
print(f"w true: {w_true}")
_test_fit(loc_true, w_true)