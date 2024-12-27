import numpy as np
from scipy.optimize import fmin_tnc
from numpy.linalg import inv
import random
from scipy.optimize import minimize, NonlinearConstraint

# ofu_mnl_plus
class ofu_mnl_plus(object):
    def __init__(self, N, K, d, kappa = 0.5, beta = None, vzero = 1.0):
        """
        :param N: number of items
        :param d: dimension of the context vectors and unknown parameter w
        :param K: maximum assortment size
        :param S: upper bound of L2 norm of each w
        :param eta: step-size parameter for online update
        :param r_lambda: regularization parameter
        """
        # immediate attributes from the constructor
        self.N = N
        self.d = d
        self.S = 1
        self.K = K
        self.eta = (self.S + 1) + np.log(self.K+1)/2
        self.r_lambda = 84 * np.sqrt(2) * self.eta * self.d
        # init parameter
        self.t = 1
        self.W_t = np.zeros(self.d)[:, None]  # estimated parameter
        self.H = self.r_lambda * np.identity(self.d)  # hessian of loss matrix
        self.inv_H = 1/self.r_lambda * np.identity(self.d)  # inverse of H
        self.beta = beta  # confidence radius
        self.vzero = vzero  # utility for the outside option
        
    def choose_S(self, t, x):
        """
        choose the optimistic assortment
        """
        if self.beta is None:
            ## calculate beta
            self.beta = np.sqrt(2 *self.eta *( (3*np.log(1 + (self.K +1)*t ) +3)
            *(17/16*self.r_lambda + 2* np.sqrt(self.r_lambda)*np.log(2*np.sqrt(1+2*t)) + 16*(np.log(2*np.sqrt(1+2*t)))**2 )
            + 2 + np.sqrt(6)*7/6*self.eta*self.d*np.log(1 + (t+1)/(2*self.r_lambda))) + 4*self.r_lambda)   
        means = np.squeeze(np.dot(x,self.W_t))
        xv = np.sqrt((np.matmul(x, self.inv_H) * x).sum(axis = 1))
        u = means + self.beta * xv
        self.assortment = np.argsort(u)[::-1][:self.K]
        self.chosen_vectors = x[self.assortment,:]
        return(self.assortment)

    def update_state(self, y):
        """
        update state
        """
        X = self.chosen_vectors
        assert isinstance(X, np.ndarray), 'np.array required'
        self.update_estimator(X, y)
        probs = self.Sigma(self.W_t, X)
        gg_Wt_fst_term = np.sum(probs[:, None, None] * np.einsum('ki,kj->kij', X, X), axis=0) 
        gg_Wt_snd_term = np.sum(np.einsum('i,j->ij', probs, probs)[:, :, np.newaxis, np.newaxis] 
                        * np.einsum('ik,jl->ijkl', X, X), axis=(0, 1))
        gg_Wt = gg_Wt_fst_term - gg_Wt_snd_term 
        self.H += gg_Wt
        self.inv_H = inv(self.H)
        self.t += 1
        
    def update_estimator(self, X, y):
        """
        update parameter
        """
        y = np.squeeze(y[:-1])
        W_estimate = self.W_t
        probs = np.squeeze(self.Sigma(self.W_t, X))
        g_Wt = np.sum(np.multiply(np.repeat((probs - y)[...,np.newaxis], self.d, axis=1), X), axis=0) # d dimension
        gg_Wt_fst_term = np.sum(probs[:, None, None] * np.einsum('ki,kj->kij', X, X), axis=0) 
        gg_Wt_snd_term = np.sum(np.einsum('i,j->ij', probs, probs)[:, :, np.newaxis, np.newaxis] 
                        * np.einsum('ik,jl->ijkl', X, X), axis=(0, 1))
        gg_Wt = gg_Wt_fst_term - gg_Wt_snd_term
        M_t = 1/(2*self.eta) * self.H + 1/2 * gg_Wt
        inv_Mt = inv(M_t)
        unprojected_update = np.squeeze(W_estimate) - np.dot(inv_Mt,g_Wt)
        if np.linalg.norm(unprojected_update) > self.S:
            if self.K == 1:
                W_estimate = self.S * unprojected_update / np.linalg.norm(unprojected_update)
            else:
                W_estimate = self.projection(unprojected_update, M_t)[:,None]
            self.W_t = W_estimate
        else:
            self.W_t = unprojected_update
        self.W_t = unprojected_update

    def Sigma(self, W, X):
        """
        calculate MNL probability 
        """
        z = np.matmul(X, W)
        sigma = np.exp(z)
        sigma = sigma / (sigma.sum(axis=0) + self.vzero)
        return sigma

    def proj_fun(self, W, un_projected, M):
        diff = W-un_projected
        fun = np.dot(diff, np.dot(M, diff))
        return fun

    def projection(self, unprojected, M):
        fun = lambda t: self.proj_fun(t, unprojected, M)
        constraints = []
        norm = lambda t: np.linalg.norm(t[self.d :self.d  + self.d])
        constraint = NonlinearConstraint(norm, 0, self.S)
        constraints.append(constraint)
        opt = minimize(fun, x0=np.zeros(self.d), method='SLSQP', constraints=constraints)
        return opt.x


#UCB-MNL
class ucb_mnl:
    def __init__(self, N, K, d, kappa = 0.5, alpha=None, lam = 1.0):
        """
        :param N: number of items
        :param d: dimension of the context vectors and unknown parameter w
        :param K: maximum assortment size
        :param X: set of contexts
        :param Y: set of choice feedbacks
        :param S: upper bound of L2 norm of each w
        :param kappa: degree of non-linearlity
        :param lam: regularization parameter
        """
        super(ucb_mnl, self).__init__()
        self.N = N
        self.K = K
        self.d = d
        self.X = np.zeros((K,d))[np.newaxis, ...]
        self.Y = np.zeros(K+1)[np.newaxis, ...]
        self.S = 1
        self.kappa = kappa
        self.lam = lam
        # init parameter
        self.theta = np.zeros(d)  # estimated parameter
        self.V = np.eye(d)*lam  # grammatrix
        self.mnl = RegularizedMNLRegression()  # MLE loss function
        self.alpha = alpha  # confidence radius
        
    def choose_S(self,t,x):  # x is N*d matrix
        """
        choose the optimistic assortment
        """
        if self.alpha is None:
            self.alpha = (1/(2*self.kappa))*np.sqrt(2*self.d*np.log(1+t/self.d)+2*np.log(t))
        means = np.dot(x,self.theta)
        xv = np.sqrt((np.matmul(x, inv(self.V)) * x).sum(axis = 1))
        u = means + self.alpha * xv
        self.S = np.argsort(u)[::-1][:self.K]
        self.X = np.concatenate((self.X, x[self.S,:][np.newaxis, ...]))
        self.V += np.matmul(x[self.S,:].T, x[self.S,:])
        return(self.S)

    def update_theta(self,Y,t):
        """
        update parameter
        """
        self.Y = np.concatenate((self.Y, Y[np.newaxis, ...]))
        if t==2:
            self.X = np.delete(self.X, (0), axis=0)
            self.Y = np.delete(self.Y, (0), axis=0)
        self.mnl.fit(self.X, self.Y, self.theta, self.lam)
        self.theta = self.mnl.w

# TS-MNL with Gaussian approximation
class ts_mnl:
    def __init__(self, N, K, d, kappa = 0.5, alpha=None, lam = 1.0):
        """
        :param N: number of items
        :param d: dimension of the context vectors and unknown parameter w
        :param K: maximum assortment size
        :param X: set of contexts
        :param Y: set of choice feedbacks
        :param S: upper bound of L2 norm of each w
        :param kappa: degree of non-linearlity
        :param lam: regularization parameter
        """
        super(ts_mnl, self).__init__()
        self.N=N
        self.K=K
        self.d=d
        self.X=np.zeros((K,d))[np.newaxis, ...]
        self.Y=np.zeros(K+1)[np.newaxis, ...]
        self.S = 1
        self.kappa = kappa
        self.lam = lam  

        # init parameter
        self.theta=np.zeros(d)  # estimated parameter
        self.V=np.eye(d)*lam  # grammatrix
        self.mnl=RegularizedMNLRegression()  # MLE loss function
        self.alpha=alpha  # confidence radius
        
        
    def choose_S(self,t,x):  # x is N*d matrix
        """
        choose the optimistic assortment
        """
        if self.alpha is None:
            self.alpha = (1/(2*self.kappa))*np.sqrt(2*self.d*np.log(1+t/self.d)+2*np.log(t))
        theta_tilde = np.random.multivariate_normal(self.theta, np.square(self.alpha)*inv(self.V))
        means = np.dot(x,theta_tilde)            
        self.S = np.argsort(means)[::-1][:self.K]
        self.X = np.concatenate((self.X, x[self.S,:][np.newaxis, ...]))
        self.V += np.matmul(x[self.S,:].T, x[self.S,:])
        return(self.S)

    def update_theta(self,Y,t):
        """
        update parameter
        """
        self.Y = np.concatenate((self.Y, Y[np.newaxis, ...]))
        if t==2:
            self.X = np.delete(self.X, (0), axis=0)
            self.Y = np.delete(self.Y, (0), axis=0)
        self.mnl.fit(self.X, self.Y, self.theta, self.lam)
        self.theta = self.mnl.w

class RegularizedMNLRegression:

    def compute_prob(self, theta, x):
        means = np.dot(x, theta)
        u = np.exp(means)
        u_ones = np.column_stack((u,np.ones(u.shape[0])))
        logSumExp = u_ones.sum(axis=1)
        prob = u_ones/logSumExp[:,None]
        return prob

    def cost_function(self, theta, x, y, lam):
        m = x.shape[0]
        prob = self.compute_prob(theta, x)
        return -(1/m)*np.sum( np.multiply(y, np.log(prob))) + (1/m)*lam*np.linalg.norm(theta)

    def gradient(self, theta, x, y, lam):
        m = x.shape[0]
        prob = self.compute_prob(theta, x)
        eps = (prob-y)[:,:-1]
        grad = (1/m)*np.tensordot(eps,x,axes=([1,0],[1,0])) + (1/m)*lam*theta
        return grad

    def fit(self, x, y, theta, lam):
        opt_weights = fmin_tnc(func=self.cost_function, x0=theta, fprime=self.gradient, args=(x, y, lam), disp=False)
        self.w = opt_weights[0]
        return self