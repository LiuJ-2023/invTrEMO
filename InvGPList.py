import torch
import gpytorch
from gpytorch.mlls import SumMarginalLogLikelihood
from TransferKernel import TransferKernel
from gpytorch.kernels.index_kernel import IndexKernel
from gpytorch.constraints import Interval
import numpy as np
from torchmin import Minimizer

class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean().cpu()#constant_constraint=Interval(lower_bound=-20,upper_bound=20)
        self.covar_module = gpytorch.kernels.RBFKernel(lengthscale_constraint=Interval(lower_bound=1e-4,upper_bound=10)).cpu()

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
class TransferGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(TransferGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean().cpu()
        self.covar_module = gpytorch.kernels.RBFKernel().cpu()
        self.task_covar_module = TransferKernel()
        # self.task_covar_module = IndexKernel(num_tasks=2).cpu()

    def forward(self,x,i):
        mean_x = self.mean_module(x)
        # Get input-input covariance
        covar_x = self.covar_module(x)
        # Get task-task covariance
        covar_i = self.task_covar_module(i)
        # Multiply the two together to get the covariance we want
        covar = covar_x.mul(covar_i)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar)

class ForwardGP():
    def __init__(self, train_xT, train_yT):        
        # Built single task GP for the target
        self.train_xT = train_xT
        self.train_yT = train_yT
        self.beta = 0.5
        self.data_size,self.dim = self.train_xT.shape
        self.build_gp()

    # Built a single-task GP
    def build_gp(self):
        likelihood_t = gpytorch.likelihoods.GaussianLikelihood()
        self.model_t = ExactGPModel(self.train_xT, self.train_yT, likelihood_t)

    # Train a single-task GP
    def train(self):
        # Train single task GP for target
        self.model_t.train()
        self.model_t.likelihood.train()
        optimizer = Minimizer(self.model_t.parameters(),
                        method='l-bfgs',
                        tol=1e-6,
                        max_iter=200,
                        disp=0)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.model_t.likelihood, self.model_t)       
        def closure():
            optimizer.zero_grad()
            output = self.model_t(self.train_xT)
            loss = -mll(output,self.train_yT)
            # loss.backward()
            return loss
        optimizer.step(closure = closure)

    def LCB(self,x):
        if x.ndim == 1:
            x = np.array([x])
        elif x.ndim >= 3:
            Exception("The dimension of the input array should be small than 3.")

        X = torch.tensor(np.array(x))
        self.model_t.eval()         
        prediction_t = self.model_t(X)
        predicted_std = prediction_t.variance.sqrt()
        predicted_mean = prediction_t.mean
        return (predicted_mean - self.beta*predicted_std).detach().numpy()
    
    def set_weight(self,w):
        self.weight_ = w

    def set_gaussian(self,mu,sigma):
        self.mu_inv = mu
        self.std_inv = sigma

    def TrLCB(self,x, mode = 'max'):
        if x.ndim == 1:
            x = np.array([x])
        elif x.ndim >= 3:
            Exception("The dimension of the input array should be small than 3.")

        X = torch.tensor(np.array(x))
        self.model_t.eval()         
        prediction_t = self.model_t(X)
        predicted_std = prediction_t.variance.sqrt()
        predicted_mean = prediction_t.mean
        psi = np.log( (1/((self.std_inv + 0.1)*np.sqrt(2*np.pi)))*np.exp(-((x-self.mu_inv)**2)/(2*(self.std_inv + 0.1)**2)) + 1)
        return (predicted_mean - self.beta*predicted_std).detach().numpy() - 0.5*np.mean(psi[:,:8],axis=1)

class InvGPList():
    def __init__(self,train_X, train_W, lb_noise = 1e-4):
        self.train_X = train_X
        self.train_W = train_W
        self.lb_noise = lb_noise
        self.data_size,self.dim = self.train_X.shape
        self.build_model()
    
    def build_model(self):
        models = []
        likelihoods = []
        for i in range(self.train_X.size(1)):
            likelihood_i = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=Interval(lower_bound=self.lb_noise,upper_bound=0.2))
            models.append(ExactGPModel(self.train_W,self.train_X[:,i],likelihood_i))
            likelihoods.append(models[i].likelihood)
        self.likelihoods_list = gpytorch.likelihoods.LikelihoodList(*likelihoods)
        self.model_list = gpytorch.models.IndependentModelList(*models)

    def train_model(self):
        self.model_list.train()
        self.likelihoods_list.train() 
        mll = SumMarginalLogLikelihood(self.likelihoods_list,self.model_list)
        # optimizer = torch.optim.Adam(self.model_list.parameters(),lr=0.1)
        # for i in range(100):
        #     optimizer.zero_grad()
        #     output = self.model_list(*self.model_list.train_inputs)
        #     loss = -mll(output, self.model_list.train_targets)
        #     loss.backward()
        #     # print('Iter %d/%d - Loss: %.3f' % (i + 1, 200, loss.item()))
        #     optimizer.step()
        optimizer = Minimizer(self.model_list.parameters(),
                                method='l-bfgs',
                                tol=1e-6,
                                max_iter=200,
                                disp=0)
        def closure():
            optimizer.zero_grad()
            output = self.model_list(*self.model_list.train_inputs)
            loss = -mll(output, self.model_list.train_targets)
            # loss.backward()
            return loss
        optimizer.step(closure = closure)
        
    def set_model(self):
        for i in range(self.train_X.size(1)):
            self.model_list.models[i].likelihood.noise_covar.initialize(noise=1e-4)
            self.model_list.models[i].covar_module._set_lengthscale(1)


    def predict(self,w,pred_mode = 'vector'):
        if w.ndim == 1:
            w = np.array([w])
        self.model_list.eval()
        self.likelihoods_list.eval()
        input_w = []
        for i in range(self.dim):
            input_w.append(torch.tensor(w))
        predictions = self.model_list(*input_w)
        pred_X = []
        pred_X_variance = []
        for i in range(self.dim):
            pred_X.append(predictions[i].mean)
            pred_X_variance.append(predictions[i].variance.sqrt())
        pred_X = torch.stack(pred_X,dim=1)
        pred_X_variance = torch.stack(pred_X_variance,dim=1)
    
        return pred_X.detach().numpy(), pred_X_variance.detach().numpy()
    
    def predict_tensor(self,w,pred_mode = 'vector'):
        if w.ndim == 1:
            w = np.array([w])
        self.model_list.eval()
        self.likelihoods_list.eval()
        input_w = []
        for i in range(self.dim):
            input_w.append(torch.tensor(w))
        predictions = self.model_list(*input_w)
        pred_X = []
        pred_X_variance = []
        for i in range(self.dim):
            pred_X.append(predictions[i].mean)
            pred_X_variance.append(predictions[i].variance.sqrt())
        pred_X = torch.stack(pred_X,dim=1)
        pred_X_variance = torch.stack(pred_X_variance,dim=1)
    
        return pred_X

class InvTGPList():
    def __init__(self,train_XT, train_WT, train_XS, train_WS,lb_noise = 1e-4):
        self.train_XT = train_XT
        self.train_WT = train_WT
        self.train_XS = train_XS
        self.train_WS = train_WS
        self.data_size,self.dim = self.train_XT.shape
        self.data_size_s,self.dim_s = self.train_XS.shape
        self.dim_overlap = np.minimum(self.dim,self.dim_s)
        self.lb_noise = lb_noise
        self.build_model_gp()
        self.build_model_tgp()
    
    def build_model_gp(self):
        models = []
        likelihoods = []
        for i in range(self.train_XT.size(1)):
            likelihood_i = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=Interval(lower_bound=self.lb_noise,upper_bound=10)).cpu()
            models.append(ExactGPModel(self.train_WT,self.train_XT[:,i],likelihood_i))
            likelihoods.append(models[i].likelihood)
        self.likelihoods_list = gpytorch.likelihoods.LikelihoodList(*likelihoods).cpu()
        self.model_list = gpytorch.models.IndependentModelList(*models).cpu()

    def build_model_tgp(self):
        tgp_models = []
        tgp_likelihoods = []
        self.full_train_x = []
        self.full_train_y = []
        self.full_train_i = []
        self.tgp_likelihoods = []
        self.tgp_models = []   
         
        train_i_taskS = torch.zeros(self.train_WS.size(0),dtype=torch.long).cpu()
        train_i_taskT = torch.ones(self.train_WT.size(0),dtype=torch.long).cpu()
        full_train_x = torch.cat([self.train_WS, self.train_WT],0).cpu()
        full_train_i = torch.cat([train_i_taskS, train_i_taskT],0).cpu()
        full_train_y = torch.cat([self.train_XS, self.train_XT[:,:self.dim_overlap]],0).cpu()
        for j in range(self.dim_overlap):
            likelihood_ij = gpytorch.likelihoods.GaussianLikelihood().cpu()
            model_ij = TransferGPModel((full_train_x, full_train_i), full_train_y[:,j], likelihood = likelihood_ij)
            model_ij.likelihood.noise_covar.register_constraint("raw_noise", Interval(self.lb_noise,10))       
            tgp_models.append(model_ij)
            tgp_likelihoods.append(model_ij.likelihood)
        self.tgp_likelihood_list = gpytorch.likelihoods.LikelihoodList(*tgp_likelihoods).cpu()
        self.tgp_model_list = gpytorch.models.IndependentModelList(*tgp_models).cpu()

    def set_model(self,set_mean):     
        self.set_mean = set_mean
        self.gp_optimizer = Minimizer(self.model_list.parameters(),
                                        method='l-bfgs',
                                        tol=1e-6,
                                        max_iter=200,
                                        disp=0)         
        self.tgp_optimizer = Minimizer(self.tgp_model_list.parameters(),
                                        method='l-bfgs',
                                        tol=1e-6,
                                        max_iter=200,
                                        disp=0)   

    def train_gp(self):
        for i in range(self.dim):
            self.model_list.models[i].mean_module.constant.data = torch.tensor(self.set_mean[i])
            self.model_list.models[i].mean_module.constant.requires_grad = False
        self.model_list.train()
        self.likelihoods_list.train() 
        mll = SumMarginalLogLikelihood(self.likelihoods_list,self.model_list)
        # for i in range(200):
        #     self.gp_optimizer.zero_grad()
        #     output = self.model_list(*self.model_list.train_inputs)
        #     loss = -mll(output, self.model_list.train_targets)
        #     loss.backward()
        #     # print('Iter %d/%d - Loss: %.3f' % (i + 1, 200, loss.item()))
        #     self.gp_optimizer.step()
        def closure():
            self.gp_optimizer.zero_grad()
            output = self.model_list(*self.model_list.train_inputs)
            loss = -mll(output, self.model_list.train_targets)
            # loss.backward()
            return loss
        self.gp_optimizer.step(closure = closure)

    def train_tgp(self):
        self.tgp_model_list.train()
        self.tgp_likelihood_list.train() 
        t = 0
        for j in range(self.dim_overlap):
            self.tgp_model_list.models[t].mean_module.constant.data = self.model_list.models[j].mean_module.constant.data
            self.tgp_model_list.models[t].covar_module.lengthscale = self.model_list.models[j].covar_module.lengthscale
            self.tgp_model_list.models[t].likelihood.noise_covar.raw_noise = self.model_list.models[j].likelihood.noise_covar.raw_noise
            self.tgp_model_list.models[t].mean_module.constant.requires_grad = False
            self.tgp_model_list.models[t].covar_module.raw_lengthscale.requires_grad = False
            self.tgp_model_list.models[t].likelihood.noise_covar.raw_noise.requires_grad = False
            t = t+1

        mll = SumMarginalLogLikelihood(self.tgp_likelihood_list,self.tgp_model_list)
        # for i in range(200):
        #     self.tgp_optimizer.zero_grad()
        #     output = self.tgp_model_list(*self.tgp_model_list.train_inputs)
        #     loss = -mll(output, self.tgp_model_list.train_targets)
        #     loss.backward()
        #     # print('Iter %d/%d - Loss: %.3f' % (i + 1, 200, loss.item()))
        #     self.tgp_optimizer.step()
        def closure():
            self.tgp_optimizer.zero_grad()
            output = self.tgp_model_list(*self.tgp_model_list.train_inputs)
            loss = -mll(output, self.tgp_model_list.train_targets)
            # loss.backward()
            return loss
        self.tgp_optimizer.step(closure = closure)

    def predict_gp(self,w):
        self.model_list.eval()
        self.likelihoods_list.eval()
        input_w = []
        for i in range(self.dim):
            input_w.append(torch.tensor(w).cpu())
        predictions = self.model_list(*input_w)
        pred_X = []
        pred_X_variance = []
        for i in range(self.dim):
            pred_X.append(predictions[i].mean)
            pred_X_variance.append(predictions[i].variance.sqrt())
        pred_X = torch.stack(pred_X,dim=1)
        pred_X_variance = torch.stack(pred_X_variance,dim=1)
    
        return pred_X.cpu().detach().numpy(), pred_X_variance.cpu().detach().numpy()
    
    def predict_tgp(self,w):
        self.tgp_model_list.eval()
        self.tgp_likelihood_list.eval()
        input_w = []
        for i in range(self.dim_overlap):
            input_w.append((torch.tensor(w).cpu(),torch.ones(torch.tensor(w).size(0), dtype=torch.long).cpu()))
        predictions = self.tgp_model_list(*input_w)
        pred_X = []
        pred_X_variance = []
        for i in range(self.dim_overlap):
            pred_X.append(predictions[i].mean)
            pred_X_variance.append(predictions[i].variance.sqrt())
        pred_X = torch.stack(pred_X,dim=1)
        pred_X_variance = torch.stack(pred_X_variance,dim=1)
    
        return pred_X.cpu().detach().numpy(), pred_X_variance.cpu().detach().numpy()

    def predict(self,w):
        # Predictions of inverse GP models
        pred_GP_mean, pred_GP_var = self.predict_gp(w)
        
        # Predictions of inverse TGP models
        pred_TGP_mean, pred_TGP_var = self.predict_tgp(w)

        # Overall Predictions
        if self.dim_overlap < self.dim:
            pred_var = pred_GP_var
            pred_mean = pred_GP_mean
            pred_var[:,:self.dim_overlap] = pred_TGP_var
            pred_mean[:,:self.dim_overlap] = pred_TGP_mean
        else:
            pred_var = pred_TGP_std
            pred_mean = pred_TGP_mean

        return pred_mean, pred_var