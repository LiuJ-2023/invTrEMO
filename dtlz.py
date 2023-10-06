import numpy as np

class DTLZ1():
    def __init__(self, obj_num = 2, n_var = 10, delta1 = 1, delta2 = 0):
        self.dim = n_var
        self.obj_num = obj_num
        self.standard_bounds = np.array([ np.zeros(n_var), np.ones(n_var) ])
        self.norm_for_hv = np.array([np.zeros(obj_num),1000*np.ones(obj_num)])
        self.delta2 = delta2
        self.delta1 = delta1
        self.pareto_f = self.cal_pareto_f()
    
    def __call__(self, x):
        if x.ndim == 1:
            x = np.array([x])
        elif x.ndim >= 3:
            Exception("The dimension of the input array should be small than 3.")
        pop_size = x.shape[0]
        M = self.obj_num
        x1 = x[:,0:M-1]**self.delta1
        x2 = x[:,M-1:self.dim]
        
        triu_mat = np.fliplr(np.triu(np.ones([M-1,M-1]))).reshape(1,M-1,M-1).repeat(pop_size,axis=0)
        x1_repeat = x1.reshape(pop_size,1,-1).repeat(M-1,axis=1)
        tril_mat = 1-triu_mat
        f_diversity1 = np.prod(x1_repeat*triu_mat+tril_mat,axis=2)
        f_diversity1 = np.concatenate((f_diversity1,np.ones([pop_size,1])),axis=1)
        inv_identity = np.fliplr(np.eye(M-1)).reshape(1,M-1,M-1).repeat(pop_size,axis=0)
        f_diversity2 = np.sum((1-x1_repeat)*inv_identity,axis=2)
        f_diversity2 = np.concatenate((np.ones([pop_size,1]),f_diversity2),axis=1)
        g = ((self.dim - M + 1)*1 + np.sum( (x[:,M-1:] - 0.5 - self.delta2 )**2 - 1*np.cos(2*np.pi*(x[:,M-1:] - 0.5 - self.delta2)), axis=1 ))
        g = g.reshape(pop_size,1)

        f = 0.5*f_diversity1*f_diversity2*(1+g)
        return f
    
    def cal_pareto_f(self):
        mc = 10000
        x = np.random.rand(mc,self.obj_num - 1)
        x = x.reshape(mc,1,-1).repeat(self.obj_num-1,axis=1)
        triu_mat = np.fliplr(np.triu(np.ones([self.obj_num-1,self.obj_num-1]))).reshape(1,self.obj_num-1,self.obj_num-1).repeat(mc,axis=0)
        tril_mat = 1-triu_mat
        f_diversity1 = np.prod(x*triu_mat+tril_mat,axis=2)
        f_diversity1 = np.concatenate((f_diversity1,np.ones([mc,1])),axis=1)
        inv_identity = np.fliplr(np.eye(self.obj_num-1)).reshape(1,self.obj_num-1,self.obj_num-1).repeat(mc,axis=0)
        f_diversity2 = np.sum((1-x)*inv_identity,axis=2)
        f_diversity2 = np.concatenate((np.ones([mc,1]),f_diversity2),axis=1)
        f = 0.5*f_diversity1*f_diversity2
        return f

    def IGD(self,x):
        m,n = x.shape
        p,q = self.pareto_f.shape
        x = x.reshape(m,1,n)
        x = x.repeat(p,axis=1)
        pareto_f = self.pareto_f.reshape(1,p,q)
        pareto_f = pareto_f.repeat(m,axis=0)
        distance = np.sqrt(np.sum((x - pareto_f)**2,axis=2))
        min_distance = np.min(distance,axis=0)
        igd = np.mean(min_distance)
        return igd

class DTLZ2():
    def __init__(self, obj_num = 2, n_var = 10, delta1 = 1, delta2=0):
        self.dim = n_var
        self.obj_num = obj_num
        self.standard_bounds = np.array([ np.zeros(n_var), np.ones(n_var) ])
        self.norm_for_hv = np.array([[0,0],[2,2]])
        self.delta1 = delta1
        self.delta2 = delta2
        self.pareto_f = self.cal_pareto_f()
    
    def __call__(self, x):
        if x.ndim == 1:
            x = np.array([x])
        elif x.ndim >= 3:
            Exception("The dimension of the input array should be small than 3.")

        pop_size = x.shape[0]
        M = self.obj_num
        x1 = x[:,0:M-1]**self.delta1
        x2 = x[:,M-1:self.dim]

        triu_mat = np.fliplr(np.triu(np.ones([M-1,M-1]))).reshape(1,M-1,M-1).repeat(pop_size,axis=0)
        x1_repeat = x1.reshape(pop_size,1,-1).repeat(M-1,axis=1)
        tril_mat = 1-triu_mat
        f_diversity1 = np.prod(np.cos(x1_repeat*np.pi/2)*triu_mat+tril_mat,axis=2)
        f_diversity1 = np.concatenate((f_diversity1,np.ones([pop_size,1])),axis=1)
        inv_identity = np.fliplr(np.eye(M-1)).reshape(1,M-1,M-1).repeat(pop_size,axis=0)
        f_diversity2 = np.sin(np.sum(x1_repeat*inv_identity,axis=2)*np.pi/2)
        f_diversity2 = np.concatenate((np.ones([pop_size,1]),f_diversity2),axis=1)
        g = np.sum( (x[:,M-1:] - 0.5 - self.delta2 )**2, axis=1 )
        g = g.reshape(pop_size,1)

        f = f_diversity1*f_diversity2*(1+g)
        return f
    
    def cal_pareto_f(self):
        mc = 10000
        x = np.random.rand(mc,self.obj_num - 1)
        x = x.reshape(mc,1,-1).repeat(self.obj_num-1,axis=1)
        triu_mat = np.fliplr(np.triu(np.ones([self.obj_num-1,self.obj_num-1]))).reshape(1,self.obj_num-1,self.obj_num-1).repeat(mc,axis=0)
        tril_mat = 1-triu_mat
        f_diversity1 = np.prod(np.cos(x*np.pi/2)*triu_mat+tril_mat,axis=2)
        f_diversity1 = np.concatenate((f_diversity1,np.ones([mc,1])),axis=1)
        inv_identity = np.fliplr(np.eye(self.obj_num-1)).reshape(1,self.obj_num-1,self.obj_num-1).repeat(mc,axis=0)
        f_diversity2 = np.sin(np.sum(x*inv_identity,axis=2)*np.pi/2)
        f_diversity2 = np.concatenate((np.ones([mc,1]),f_diversity2),axis=1)
        f = f_diversity1*f_diversity2
        return f

    def IGD(self,x):
        m,n = x.shape
        p,q = self.pareto_f.shape
        x = x.reshape(m,1,n)
        x = x.repeat(p,axis=1)
        pareto_f = self.pareto_f.reshape(1,p,q)
        pareto_f = pareto_f.repeat(m,axis=0)
        distance = np.sqrt(np.sum((x - pareto_f)**2,axis=2))
        min_distance = np.min(distance,axis=0)
        igd = np.mean(min_distance)
        return igd
    
class DTLZ3():
    def __init__(self, obj_num = 2, n_var = 10, delta1 = 1, delta2 = 0):
        self.dim = n_var
        self.obj_num = obj_num
        self.standard_bounds = np.array([ np.zeros(n_var), np.ones(n_var) ])
        self.norm_for_hv = np.array([[0,0],[2000,2000]])
        self.delta1 = delta1
        self.delta2 = delta2
        self.pareto_f = self.cal_pareto_f()
    
    def __call__(self, x):
        if x.ndim == 1:
            x = np.array([x])
        elif x.ndim >= 3:
            Exception("The dimension of the input array should be small than 3.")

        pop_size = x.shape[0]
        M = self.obj_num
        x1 = x[:,0:M-1]**(2*self.delta1)
        x2 = x[:,M-1:self.dim]

        triu_mat = np.fliplr(np.triu(np.ones([M-1,M-1]))).reshape(1,M-1,M-1).repeat(pop_size,axis=0)
        x1_repeat = x1.reshape(pop_size,1,-1).repeat(M-1,axis=1)
        tril_mat = 1-triu_mat
        f_diversity1 = np.prod(np.cos(x1_repeat*np.pi/2)*triu_mat+tril_mat,axis=2)
        f_diversity1 = np.concatenate((f_diversity1,np.ones([pop_size,1])),axis=1)
        inv_identity = np.fliplr(np.eye(M-1)).reshape(1,M-1,M-1).repeat(pop_size,axis=0)
        f_diversity2 = np.sin(np.sum(x1_repeat*inv_identity,axis=2)*np.pi/2)
        f_diversity2 = np.concatenate((np.ones([pop_size,1]),f_diversity2),axis=1)
        g = ((self.dim - M + 1)*0.1 + np.sum( (x[:,M-1:] - 0.5 - self.delta2 )**2 - 0.1*np.cos(2*np.pi*(x[:,M-1:] - 0.5 - self.delta2)), axis=1 ))
        g = g.reshape(pop_size,1)

        f = f_diversity1*f_diversity2*(1+g)
        return f
    
    def cal_pareto_f(self):
        mc = 10000
        x = np.random.rand(mc,self.obj_num - 1)
        x = x.reshape(mc,1,-1).repeat(self.obj_num-1,axis=1)
        triu_mat = np.fliplr(np.triu(np.ones([self.obj_num-1,self.obj_num-1]))).reshape(1,self.obj_num-1,self.obj_num-1).repeat(mc,axis=0)
        tril_mat = 1-triu_mat
        f_diversity1 = np.prod(np.cos(x*np.pi/2)*triu_mat+tril_mat,axis=2)
        f_diversity1 = np.concatenate((f_diversity1,np.ones([mc,1])),axis=1)
        inv_identity = np.fliplr(np.eye(self.obj_num-1)).reshape(1,self.obj_num-1,self.obj_num-1).repeat(mc,axis=0)
        f_diversity2 = np.sin(np.sum(x*inv_identity,axis=2)*np.pi/2)
        f_diversity2 = np.concatenate((np.ones([mc,1]),f_diversity2),axis=1)
        f = f_diversity1*f_diversity2
        return f

    def IGD(self,x):
        m,n = x.shape
        p,q = self.pareto_f.shape
        x = x.reshape(m,1,n)
        x = x.repeat(p,axis=1)
        pareto_f = self.pareto_f.reshape(1,p,q)
        pareto_f = pareto_f.repeat(m,axis=0)
        distance = np.sqrt(np.sum((x - pareto_f)**2,axis=2))
        min_distance = np.min(distance,axis=0)
        igd = np.mean(min_distance)
        return igd
    
class DTLZ4():
    def __init__(self, obj_num = 2, n_var = 10, delta1 = 1, delta2 = 0):
        self.dim = n_var
        self.obj_num = obj_num
        self.standard_bounds = np.array([ np.zeros(n_var), np.ones(n_var) ])
        self.norm_for_hv = np.array([[0,0],[2,2]])
        self.delta1 = delta1
        self.delta2 = delta2
        self.pareto_f = self.cal_pareto_f()
    
    def __call__(self, x):
        if x.ndim == 1:
            x = np.array([x])
        elif x.ndim >= 3:
            Exception("The dimension of the input array should be small than 3.")

        pop_size = x.shape[0]
        M = self.obj_num
        x1 = x[:,0:M-1]**(2*self.s)
        x2 = x[:,M-1:self.dim]

        triu_mat = np.fliplr(np.triu(np.ones([M-1,M-1]))).reshape(1,M-1,M-1).repeat(pop_size,axis=0)
        x1_repeat = x1.reshape(pop_size,1,-1).repeat(M-1,axis=1)
        tril_mat = 1-triu_mat
        f_diversity1 = np.prod(np.cos(x1_repeat*np.pi/2)*triu_mat+tril_mat,axis=2)
        f_diversity1 = np.concatenate((f_diversity1,np.ones([pop_size,1])),axis=1)
        inv_identity = np.fliplr(np.eye(M-1)).reshape(1,M-1,M-1).repeat(pop_size,axis=0)
        f_diversity2 = np.sin(np.sum(x1_repeat*inv_identity,axis=2)*np.pi/2)
        f_diversity2 = np.concatenate((np.ones([pop_size,1]),f_diversity2),axis=1)
        g = np.sum( (x[:,M-1:] - 0.5 - self.delta2 )**2, axis=1 )
        g = g.reshape(pop_size,1)

        f = f_diversity1*f_diversity2*(1+g)
        return f
    
    def cal_pareto_f(self):
        mc = 10000
        x = np.random.rand(mc,self.obj_num - 1)
        x = x.reshape(mc,1,-1).repeat(self.obj_num-1,axis=1)
        triu_mat = np.fliplr(np.triu(np.ones([self.obj_num-1,self.obj_num-1]))).reshape(1,self.obj_num-1,self.obj_num-1).repeat(mc,axis=0)
        tril_mat = 1-triu_mat
        f_diversity1 = np.prod(np.cos(x*np.pi/2)*triu_mat+tril_mat,axis=2)
        f_diversity1 = np.concatenate((f_diversity1,np.ones([mc,1])),axis=1)
        inv_identity = np.fliplr(np.eye(self.obj_num-1)).reshape(1,self.obj_num-1,self.obj_num-1).repeat(mc,axis=0)
        f_diversity2 = np.sin(np.sum(x*inv_identity,axis=2)*np.pi/2)
        f_diversity2 = np.concatenate((np.ones([mc,1]),f_diversity2),axis=1)
        f = f_diversity1*f_diversity2
        return f

    def IGD(self,x):
        m,n = x.shape
        p,q = self.pareto_f.shape
        x = x.reshape(m,1,n)
        x = x.repeat(p,axis=1)
        pareto_f = self.pareto_f.reshape(1,p,q)
        pareto_f = pareto_f.repeat(m,axis=0)
        distance = np.sqrt(np.sum((x - pareto_f)**2,axis=2))
        min_distance = np.min(distance,axis=0)
        igd = np.mean(min_distance)
        return igd