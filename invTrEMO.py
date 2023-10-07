import dtlz
import matplotlib.pyplot as plt
import numpy as np
from Utils import Tchebycheff, Inv_Transform
import gpytorch
import torch
from gpytorch.models import IndependentModelList
from InvGPList import InvTGPList, ForwardGP
import nondomination
import copy
import pickle
from pymoo.util.ref_dirs import get_reference_directions
import random
import os
from datetime import datetime
from pymoo.core.problem import Problem

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

class OPT(Problem):
    def __init__(self,input_func,n_var,n_obj,xl=0,xu=1):
        super().__init__(n_var=n_var,n_obj=n_obj,xl=xl,xu=xu)
        self.input_func = input_func

    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = self.input_func(x)

def trans_to_tensor(X,Y):
    max_Y = np.max(Y,axis=0).reshape(1,-1)
    min_Y = np.min(Y,axis=0).reshape(1,-1)
    Y = (Y - min_Y)/(10 - min_Y)
    return torch.tensor(X),torch.tensor(Y), Y

def invTrEMO(test_problem,source_data,init_pop=None):
    # Parameter Setting
    pop_size = 20
    max_iter = 80
    sigma_0 = 0.01

    # Generate preference weights
    w_all = get_reference_directions("energy", test_problem.obj_num, 50)

    # Load source inverse models
    Pareto_WS = source_data[0]
    Pareto_XS = source_data[1]

    # Initilize the population and Pareto optimal solutions
    if init_pop is None:
        Pop = np.random.rand(pop_size,test_problem.dim)
    else:
        Pop = init_pop
    y_Pop = test_problem(Pop)

    # Evolution
    Recorded_IGD = []
    for iter in range(max_iter):
        # Calculate IGD
        Current_IGD = test_problem.IGD(y_Pop)
        Recorded_IGD.append(Current_IGD)
        print("invTrEMO IGD: "+str(Current_IGD))

        # Build forward model
        idx_sel = np.random.choice(50)
        w = w_all[idx_sel]
        ww = 1/(w+1e-10)
        ww = ww/np.sum(ww)
        Dataset_XT,Dataset_YT,y_Pop_norm = trans_to_tensor(Pop,y_Pop)
        Y_tch = Tchebycheff(Dataset_YT.numpy(),ww)
        Y_tch_norm = torch.tensor((Y_tch - np.mean(Y_tch))/np.std(Y_tch))
        model = ForwardGP(Dataset_XT,Y_tch_norm)
        model.train()

        # Generate target dataset
        is_efficient = nondomination.is_pareto_efficient_simple(y_Pop_norm)
        Pareto_X = Pop[is_efficient]
        Pareto_Y = y_Pop_norm[is_efficient]
        Pareto_W, Pareto_X = Inv_Transform(Pareto_X, Pareto_Y)

        # Build inverse model
        train_WT = torch.tensor(Pareto_W).cpu()
        train_XT = torch.tensor(Pareto_X).cpu()
        train_WS = torch.tensor(Pareto_WS).cpu()
        train_XS = torch.tensor(Pareto_XS).cpu()
        InvTGP = InvTGPList(train_XT, train_WT, train_XS, train_WS,sigma_0)
        mu_t = torch.tensor( np.mean(test_problem.standard_bounds,axis=0) )
        InvTGP.set_model(mu_t)
        InvTGP.train_gp()

        # Generate offspring
        try:
            InvTGP.train_tgp()
            mu_inv, std_inv = InvTGP.predict(w.reshape(1,-1))
            x_sample = mu_inv + std_inv*np.random.randn(10000,test_problem.dim)
            x_sample = (x_sample >= test_problem.standard_bounds[0:1,:])*x_sample + (x_sample < test_problem.standard_bounds[0:1,:])*test_problem.standard_bounds[0:1,:]
            x_sample = (x_sample <= test_problem.standard_bounds[1:2,:])*x_sample + (x_sample > test_problem.standard_bounds[1:2,:])*test_problem.standard_bounds[1:2,:]
            f_hat_sample = model.LCB(x_sample)
            idx_best = np.argmin(f_hat_sample)
            x_new = x_sample[idx_best:idx_best+1]
        except:
            mu_inv, std_inv = InvTGP.predict_gp(w.reshape(1,-1))
            x_sample = mu_inv + std_inv*np.random.randn(10000,test_problem.dim)
            x_sample = (x_sample >= test_problem.standard_bounds[0:1,:])*x_sample + (x_sample < test_problem.standard_bounds[0:1,:])*test_problem.standard_bounds[0:1,:]
            x_sample = (x_sample <= test_problem.standard_bounds[1:2,:])*x_sample + (x_sample > test_problem.standard_bounds[1:2,:])*test_problem.standard_bounds[1:2,:]
            f_hat_sample = model.LCB(x_sample)
            idx_best = np.argmin(f_hat_sample)
            x_new = x_sample[idx_best:idx_best+1]

        # Evaluate query solutions
        y_new = test_problem(x_new)
        Pop = np.concatenate((Pop,x_new),axis=0)
        y_Pop = np.concatenate((y_Pop,y_new),axis=0)

    # Train the final inverse TGPs
    is_efficient = nondomination.is_pareto_efficient_simple(y_Pop)
    Pareto_X = Pop[is_efficient]
    Pareto_Y = y_Pop[is_efficient]
    Pareto_W, Pareto_X = Inv_Transform(Pareto_X, Pareto_Y)
    train_WT = torch.tensor(Pareto_W).cpu()
    train_XT = torch.tensor(Pareto_X).cpu()
    train_WS = torch.tensor(Pareto_WS).cpu()
    train_XS = torch.tensor(Pareto_XS).cpu()
    InvTGP = InvTGPList(train_XT, train_WT, train_XS, train_WS)
    InvTGP.set_model(mu_t)
    InvTGP.train_gp()
    InvTGP.train_tgp()

    return InvTGP, Recorded_IGD, [Pop, y_Pop]
