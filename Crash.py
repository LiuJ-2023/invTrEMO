import numpy as np

class Crash_Source_Case1():
    def __init__(self):
        self.dim = 4
        self.obj_num = 3
        self.standard_bounds = np.array([ 0*np.ones(4), 1*np.ones(4) ])
        self.norm_for_hv = np.array([[0,0],[2,2]])
    
    def __call__(self, x):
        if x.ndim == 1:
            x = np.array([x])
        elif x.ndim >= 3:
            Exception("The dimension of the input array should be small than 3.")
        x = 1.5*x + 0.5
        x = np.concatenate((0.5*np.ones([x.shape[0],1]),x),axis=1)
        
        f1 = 1640.2823 + 2.3573285*x[:,0] + 2.3220035*x[:,1] + 4.5688768*x[:,2] + 7.7213633*x[:,3] + 4.4559504*x[:,4]
        f2 = 6.5856 + 1.15*x[:,0] - 1.0427*x[:,1] + 0.9738*x[:,2] + 0.8364*x[:,3] - 0.3695*x[:,0]*x[:,3] + 0.0861*x[:,0]*x[:,4] + 0.3628*x[:,1]*x[:,3] - 0.1106*x[:,0]**2 - 0.3437*x[:,2]**2 + 0.1764*x[:,3]**3
        f3 = -0.0551 + 0.0181*x[:,0] + 0.1024*x[:,1] + 0.0421*x[:,2] - 0.0073*x[:,0]*x[:,1] + 0.024*x[:,1]*x[:,2] - 0.0118*x[:,1]*x[:,3] - 0.0204*x[:,2]*x[:,3] - 0.008*x[:,2]*x[:,4] - 0.0241*x[:,1]**2 + 0.0109*x[:,3]**2
        f = np.concatenate((f1.reshape(-1,1),f2.reshape(-1,1),f3.reshape(-1,1)),axis=1)
        # f = (f - np.array([1650,6.180,0.01]).reshape(1,-1))/np.array([30,5,0.2]).reshape(1,-1)
        return f

class Crash_Target_Case1():
    def __init__(self):
        self.dim = 5
        self.obj_num = 3
        self.standard_bounds = np.array([ 0*np.ones(5), 1*np.ones(5) ])
        self.norm_for_hv = np.array([[0,0],[2,2]])
    
    def __call__(self, x):
        if x.ndim == 1:
            x = np.array([x])
        elif x.ndim >= 3:
            Exception("The dimension of the input array should be small than 3.")
        xx1 = x[:,:4]
        xx2 = x[:,4:]
        x = np.concatenate((xx2,xx1),axis=1)
        x = 1.5*x + 0.5
        f1 = 1640.2823 + 2.3573285*x[:,0] + 2.3220035*x[:,1] + 4.5688768*x[:,2] + 7.7213633*x[:,3] + 4.4559504*x[:,4]
        f2 = 6.5856 + 1.15*x[:,0] - 1.0427*x[:,1] + 0.9738*x[:,2] + 0.8364*x[:,3] - 0.3695*x[:,0]*x[:,3] + 0.0861*x[:,0]*x[:,4] + 0.3628*x[:,1]*x[:,3] - 0.1106*x[:,0]**2 - 0.3437*x[:,2]**2 + 0.1764*x[:,3]**3
        f3 = -0.0551 + 0.0181*x[:,0] + 0.1024*x[:,1] + 0.0421*x[:,2] - 0.0073*x[:,0]*x[:,1] + 0.024*x[:,1]*x[:,2] - 0.0118*x[:,1]*x[:,3] - 0.0204*x[:,2]*x[:,3] - 0.008*x[:,2]*x[:,4] - 0.0241*x[:,1]**2 + 0.0109*x[:,3]**2
        f = np.concatenate((f1.reshape(-1,1),f2.reshape(-1,1),f3.reshape(-1,1)),axis=1)
        # f = (f - np.array([1650,6.180,0.01]).reshape(1,-1))/np.array([30,5,0.2]).reshape(1,-1)
        return f

class Crash_Source_Case2():
    def __init__(self):
        self.dim = 4
        self.obj_num = 3
        self.standard_bounds = np.array([ 0*np.ones(4), 1*np.ones(4) ])
        self.norm_for_hv = np.array([[0,0],[2,2]])
    
    def __call__(self, x):
        if x.ndim == 1:
            x = np.array([x])
        elif x.ndim >= 3:
            Exception("The dimension of the input array should be small than 3.")
        x = 1.5*x + 0.5
        x = np.concatenate((x,0.5*np.ones([x.shape[0],1])),axis=1)

        f1 = 1640.2823 + 2.3573285*x[:,0] + 2.3220035*x[:,1] + 4.5688768*x[:,2] + 7.7213633*x[:,3] + 4.4559504*x[:,4]
        f2 = 6.5856 + 1.15*x[:,0] - 1.0427*x[:,1] + 0.9738*x[:,2] + 0.8364*x[:,3] - 0.3695*x[:,0]*x[:,3] + 0.0861*x[:,0]*x[:,4] + 0.3628*x[:,1]*x[:,3] - 0.1106*x[:,0]**2 - 0.3437*x[:,2]**2 + 0.1764*x[:,3]**3
        f3 = -0.0551 + 0.0181*x[:,0] + 0.1024*x[:,1] + 0.0421*x[:,2] - 0.0073*x[:,0]*x[:,1] + 0.024*x[:,1]*x[:,2] - 0.0118*x[:,1]*x[:,3] - 0.0204*x[:,2]*x[:,3] - 0.008*x[:,2]*x[:,4] - 0.0241*x[:,1]**2 + 0.0109*x[:,3]**2
        f = np.concatenate((f1.reshape(-1,1),f2.reshape(-1,1),f3.reshape(-1,1)),axis=1)
        # f = (f - np.array([1650,6.180,0.01]).reshape(1,-1))/np.array([30,5,0.2]).reshape(1,-1)
        return f

class Crash_Target_Case2():
    def __init__(self):
        self.dim = 4
        self.obj_num = 3
        self.standard_bounds = np.array([ 0*np.ones(4), 1*np.ones(4) ])
        self.norm_for_hv = np.array([[0,0],[2,2]])
    
    def __call__(self, x):
        if x.ndim == 1:
            x = np.array([x])
        elif x.ndim >= 3:
            Exception("The dimension of the input array should be small than 3.")
        x = 1.5*x + 0.5
        x = np.concatenate((0.5*np.ones([x.shape[0],1]),x),axis=1)
        
        f1 = 1640.2823 + 2.3573285*x[:,0] + 2.3220035*x[:,1] + 4.5688768*x[:,2] + 7.7213633*x[:,3] + 4.4559504*x[:,4]
        f2 = 6.5856 + 1.15*x[:,0] - 1.0427*x[:,1] + 0.9738*x[:,2] + 0.8364*x[:,3] - 0.3695*x[:,0]*x[:,3] + 0.0861*x[:,0]*x[:,4] + 0.3628*x[:,1]*x[:,3] - 0.1106*x[:,0]**2 - 0.3437*x[:,2]**2 + 0.1764*x[:,3]**3
        f3 = -0.0551 + 0.0181*x[:,0] + 0.1024*x[:,1] + 0.0421*x[:,2] - 0.0073*x[:,0]*x[:,1] + 0.024*x[:,1]*x[:,2] - 0.0118*x[:,1]*x[:,3] - 0.0204*x[:,2]*x[:,3] - 0.008*x[:,2]*x[:,4] - 0.0241*x[:,1]**2 + 0.0109*x[:,3]**2
        f = np.concatenate((f1.reshape(-1,1),f2.reshape(-1,1),f3.reshape(-1,1)),axis=1)
        # f = (f - np.array([1650,6.180,0.01]).reshape(1,-1))/np.array([30,5,0.2]).reshape(1,-1)
        return f

class Crash_Source_Case3():
    def __init__(self):
        self.dim = 5
        self.obj_num = 3
        self.standard_bounds = np.array([ 0*np.ones(5), 1*np.ones(5) ])
        self.norm_for_hv = np.array([[0,0],[2,2]])
    
    def __call__(self, x):
        if x.ndim == 1:
            x = np.array([x])
        elif x.ndim >= 3:
            Exception("The dimension of the input array should be small than 3.")
        xx1 = x[:,:4]
        xx2 = x[:,4:]
        x = np.concatenate((xx2,xx1),axis=1)
        x = 1.5*x + 0.5
        f1 = 1640.2823 + 2.3573285*x[:,0] + 2.3220035*x[:,1] + 4.5688768*x[:,2] + 7.7213633*x[:,3] + 4.4559504*x[:,4]
        f2 = 6.5856 + 1.15*x[:,0] - 1.0427*x[:,1] + 0.9738*x[:,2] + 0.8364*x[:,3] - 0.3695*x[:,0]*x[:,3] + 0.0861*x[:,0]*x[:,4] + 0.3628*x[:,1]*x[:,3] - 0.1106*x[:,0]**2 - 0.3437*x[:,2]**2 + 0.1764*x[:,3]**3
        f3 = -0.0551 + 0.0181*x[:,0] + 0.1024*x[:,1] + 0.0421*x[:,2] - 0.0073*x[:,0]*x[:,1] + 0.024*x[:,1]*x[:,2] - 0.0118*x[:,1]*x[:,3] - 0.0204*x[:,2]*x[:,3] - 0.008*x[:,2]*x[:,4] - 0.0241*x[:,1]**2 + 0.0109*x[:,3]**2
        f = np.concatenate((f1.reshape(-1,1),f2.reshape(-1,1),f3.reshape(-1,1)),axis=1)
        # f = (f - np.array([1650,6.180,0.01]).reshape(1,-1))/np.array([30,5,0.2]).reshape(1,-1)
        return f

class Crash_Target_Case3():
    def __init__(self):
        self.dim = 4
        self.obj_num = 3
        self.standard_bounds = np.array([ 0*np.ones(4), 1*np.ones(4) ])
        self.norm_for_hv = np.array([[0,0],[2,2]])
    
    def __call__(self, x):
        if x.ndim == 1:
            x = np.array([x])
        elif x.ndim >= 3:
            Exception("The dimension of the input array should be small than 3.")
        x = 1.5*x + 0.5
        x = np.concatenate((0.5*np.ones([x.shape[0],1]),x),axis=1)
        
        f1 = 1640.2823 + 2.3573285*x[:,0] + 2.3220035*x[:,1] + 4.5688768*x[:,2] + 7.7213633*x[:,3] + 4.4559504*x[:,4]
        f2 = 6.5856 + 1.15*x[:,0] - 1.0427*x[:,1] + 0.9738*x[:,2] + 0.8364*x[:,3] - 0.3695*x[:,0]*x[:,3] + 0.0861*x[:,0]*x[:,4] + 0.3628*x[:,1]*x[:,3] - 0.1106*x[:,0]**2 - 0.3437*x[:,2]**2 + 0.1764*x[:,3]**3
        f3 = -0.0551 + 0.0181*x[:,0] + 0.1024*x[:,1] + 0.0421*x[:,2] - 0.0073*x[:,0]*x[:,1] + 0.024*x[:,1]*x[:,2] - 0.0118*x[:,1]*x[:,3] - 0.0204*x[:,2]*x[:,3] - 0.008*x[:,2]*x[:,4] - 0.0241*x[:,1]**2 + 0.0109*x[:,3]**2
        f = np.concatenate((f1.reshape(-1,1),f2.reshape(-1,1),f3.reshape(-1,1)),axis=1)
        # f = (f - np.array([1650,6.180,0.01]).reshape(1,-1))/np.array([30,5,0.2]).reshape(1,-1)
        return f