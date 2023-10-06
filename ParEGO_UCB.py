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
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.optimize import minimize

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

class OPT(Problem):
    def __init__(self,input_func,n_var,n_obj,xl=0,xu=1):
        super().__init__(n_var=n_var,n_obj=n_obj,xl=xl,xu=xu)
        self.input_func = input_func

    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = self.input_func(x)

def normalize_(X,Y):
    max_Y = np.max(Y,axis=0).reshape(1,-1)
    min_Y = np.min(Y,axis=0).reshape(1,-1)
    Y = (Y - min_Y)/(max_Y - min_Y)
    return torch.tensor(X),torch.tensor(Y)

def ParEGO_UCB(test_problem,source_data, init_pop=None):
    # Parameter Setting
    pop_size = 20
    max_iter = 80

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
        print("ParEGO IGD: " + str(Current_IGD))

        # Build forward model
        idx_sel = np.random.choice(50)
        w = w_all[idx_sel]
        Dataset_XT,Dataset_YT = normalize_(Pop,y_Pop)
        Y_tch = Tchebycheff(Dataset_YT.numpy(),1/(w + 1e-10))
        Y_tch_norm = torch.tensor((Y_tch - np.mean(Y_tch))/np.std(Y_tch))
        model = ForwardGP(Dataset_XT,Y_tch_norm)
        model.train()

        # Obtain current pareto solutions
        is_efficient = nondomination.is_pareto_efficient_simple(y_Pop)
        Pareto_X = Pop[is_efficient]
        Pareto_Y = y_Pop[is_efficient]

        # Solve acquisition function
        obj_func = model.LCB
        obj_problem = OPT(obj_func,n_var=test_problem.dim,n_obj=1)
        algorithm = GA(
                        pop_size = 50,
                    )
        res = minimize(obj_problem,
               algorithm,
               ('n_gen', 200))
        x_new = res.X
        x_new = x_new.reshape(1,-1)

        # Evaluate query solutions
        y_new = test_problem(x_new)
        Pop = np.concatenate((Pop,x_new),axis=0)
        y_Pop = np.concatenate((y_Pop,y_new),axis=0)   
    
    # Obtain current pareto solutions
    is_efficient = nondomination.is_pareto_efficient_simple(y_Pop)
    Pareto_X = Pop[is_efficient]
    Pareto_Y = y_Pop[is_efficient]
    
    # Creat target data
    Pareto_W, Pareto_X = Inv_Transform(Pareto_X, Pareto_Y)  

    # Build inverse model
    train_WT = torch.tensor(Pareto_W)
    train_XT = torch.tensor(Pareto_X)
    train_WS = torch.tensor(Pareto_WS)
    train_XS = torch.tensor(Pareto_XS)

    InvTGP = InvTGPList(train_XT, train_WT, train_XS, train_WS)
    mu_t = torch.tensor( np.mean(test_problem.standard_bounds,axis=0) )
    InvTGP.set_model(mu_t)
    InvTGP.train_gp()
    InvTGP.train_tgp()
      
    return InvTGP, Recorded_IGD, [Pop,y_Pop]