import nondomination
import numpy as np
import copy

def Tchebycheff(f,w):
    if f.ndim == 1:
        f = np.array([f])
    if w.ndim == 1:
        w = np.array([w])
    # z = np.min(f,axis=0)
    # z_max = np.max(f,axis=0)
    # f = (f-z)/(z_max - z)
    f_agg = np.max(f*w,axis=1) + 0.05*np.sum(f*w,axis=1)
    # f_agg = np.max(w*(f-z),axis=1)
    return f_agg

def Inv_Transform(X,Y):
    # y_max = np.max(Y,axis=0)
    y_min = np.min(Y,axis=0)
    y_delta = ((Y + 1e-10) - y_min)#/(y_max - y_min)
    # y_delta = (Y + 1e-10)
    W = y_delta/np.sum(y_delta,axis=1).reshape(-1,1)
    return W,X

def generate_w(obj_num,mc):
    x = np.random.rand(mc,obj_num - 1)
    x = x.reshape(mc,1,-1).repeat(obj_num-1,axis=1)
    triu_mat = np.fliplr(np.triu(np.ones([obj_num-1,obj_num-1]))).reshape(1,obj_num-1,obj_num-1).repeat(mc,axis=0)
    tril_mat = 1-triu_mat
    f_diversity1 = np.prod(x*triu_mat+tril_mat,axis=2)
    f_diversity1 = np.concatenate((f_diversity1,np.ones([mc,1])),axis=1)
    inv_identity = np.fliplr(np.eye(obj_num-1)).reshape(1,obj_num-1,obj_num-1).repeat(mc,axis=0)
    f_diversity2 = np.sum((1-x)*inv_identity,axis=2)
    f_diversity2 = np.concatenate((np.ones([mc,1]),f_diversity2),axis=1)
    f = f_diversity1*f_diversity2
    return f