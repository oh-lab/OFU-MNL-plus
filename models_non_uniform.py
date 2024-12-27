import numpy as np
from scipy.optimize import fmin_tnc
from numpy.linalg import inv
import random
import pulp
from scipy.optimize import minimize, NonlinearConstraint

# ofu_mnl_plus
class ofu_mnl_plus(object):
    def __init__(self, N, K, d, kappa, beta = None):  
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
        self.L = 1
        self.eta = (self.S + 1) + np.log(self.K+1)/2
        self.r_lambda = 84 * np.sqrt(2) * self.eta * self.d
        # init parameter
        self.t = 1
        self.W_t = np.zeros(self.d)[:, None]  # estimated parameter
        self.H = self.r_lambda * np.identity(self.d)  # hessian of loss matrix
        self.inv_H = 1/self.r_lambda * np.identity(self.d)  # inverse of H
        self.beta = beta  # confidence radius
        
    def choose_S(self, t, x, reward):
        """
        choose the optimistic assortment under uniform rewards
        """
        if self.beta is None:
            ## calculate beta
            self.beta = np.sqrt(2 *self.eta *( (3*np.log(1 + (self.K +1)*t ) +3)
            *(17/16*self.r_lambda + 2* np.sqrt(self.r_lambda)*np.log(2*np.sqrt(1+2*t)) + 16*(np.log(2*np.sqrt(1+2*t)))**2 )
            + 2 + np.sqrt(6)*7/6*self.eta*self.d*np.log(1 + (t+1)/(2*self.r_lambda))) + 4*self.r_lambda)   
        means = np.squeeze(np.dot(x,self.W_t))
        xv = np.sqrt((np.matmul(x, self.inv_H) * x).sum(axis = 1))
        z = means + self.beta/10 * xv

        # Define the problem
        lp_prob = pulp.LpProblem('Maximize_Revenue', pulp.LpMaximize)	
        clipped_z = np.clip(z,  0, 2)
        w_ti = np.exp(clipped_z)

        # Decision variables
        p_ti = pulp.LpVariable.dicts('p_ti', range(self.N), lowBound=0, upBound=1)
        p_t0 = pulp.LpVariable('p_t0', lowBound=0, upBound=1)

        # Objective Function
        lp_prob += pulp.lpSum([reward[i] * p_ti[i] for i in range(self.N)])

        # Constraints
        lp_prob += pulp.lpSum([p_ti[i] for i in range(self.N)]) + p_t0 == 1
        lp_prob += pulp.lpSum([p_ti[i] * (1.0 / w_ti[i]) for i in range(self.N)]) <= self.K * p_t0

        # The third constraint for each item
        for i in range(self.N):
            lp_prob += p_ti[i] * (1.0 / w_ti[i]) <= p_t0
            lp_prob += p_ti[i] * (1.0 / w_ti[i]) >= 0
        # Solve the problem
        lp_prob.solve(pulp.PULP_CBC_CMD(msg=False))
        # print(pulp.value(lp_prob.objective))

        chosen_items = list()
        for i, variable in enumerate(lp_prob.variables()):
            # print(f"{variable.name} = {variable.varValue}")
            if variable.varValue > 0 and variable.name != 'p_t0':
                chosen_items.append(int(variable.name[5:]))
                # print(f"{variable.name} = {variable.varValue}")

        self.assortment = np.array(chosen_items)
        self.chosen_vectors = x[self.assortment,:]

        return self.assortment 


    def update_state(self, y):
        """
        update state
        """
        X = self.chosen_vectors
        assert isinstance(X, np.ndarray), 'np.array required'
        len_y = len(y)
        self.update_estimator(X, y)
        probs = self.Sigma(self.W_t, X)
        if len_y == 2:
            gg_Wt_fst_term = np.sum(probs[:, None, None] * np.einsum('ki,kj->kij', X, X), axis=0) 
            gg_Wt_snd_term = np.sum(np.einsum('i,j->ij', probs, probs)[:, :, np.newaxis, np.newaxis] 
                            * np.einsum('ik,jl->ijkl', X, X), axis=(0, 1))            
        else:
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
        len_y = len(y)
        y = np.squeeze(y[:-1])
        W_estimate = self.W_t
        probs = np.squeeze(self.Sigma(self.W_t, X))
        if len_y == 2:
            g_Wt = np.squeeze((probs - y) * X)
            gg_Wt_fst_term = probs * np.outer(X,X)
            p_X = probs * X
            gg_Wt_snd_term = np.outer(p_X, p_X)
            gg_Wt = gg_Wt_fst_term - gg_Wt_snd_term
        else:
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
        sigma = sigma / (sigma.sum(axis=0) + 1)
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
    def __init__(self, N, K, d, kappa, alpha=None, lam = 1.0):
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
        
    def choose_S(self,t,x,reward):  # x is N*d matrix
        """
        choose the optimistic assortment
        """
        if self.alpha is None:
            self.alpha = (1/(2*self.kappa))*np.sqrt(2*self.d*np.log(1+t/self.d)+2*np.log(t))
        means = np.dot(x,self.theta)
        xv = np.sqrt((np.matmul(x, inv(self.V)) * x).sum(axis = 1))
        z = means + self.alpha/10 * xv

        # Define the problem
        lp_prob = pulp.LpProblem('Maximize_Revenue', pulp.LpMaximize)	
        clipped_z = np.clip(z,  0, 2)
        w_ti = np.exp(clipped_z)

        # Decision variables
        p_ti = pulp.LpVariable.dicts('p_ti', range(self.N), lowBound=0, upBound=1)
        p_t0 = pulp.LpVariable('p_t0', lowBound=0, upBound=1)

        # Objective Function
        lp_prob += pulp.lpSum([reward[i] * p_ti[i] for i in range(self.N)])

        # Constraints
        lp_prob += pulp.lpSum([p_ti[i] for i in range(self.N)]) + p_t0 == 1
        lp_prob += pulp.lpSum([p_ti[i] * (1.0 / w_ti[i]) for i in range(self.N)]) <= self.K * p_t0

        # The third constraint for each item
        for i in range(self.N):
            lp_prob += p_ti[i] * (1.0 / w_ti[i]) <= p_t0
            lp_prob += p_ti[i] * (1.0 / w_ti[i]) >= 0
        # Solve the problem
        lp_prob.solve(pulp.PULP_CBC_CMD(msg=False))

        chosen_items = list()
        for i, variable in enumerate(lp_prob.variables()):
            if variable.varValue > 0 and variable.name != 'p_t0':
                chosen_items.append(int(variable.name[5:]))

        self.S = np.array(chosen_items)

        chosen_vectors = x[self.S,:][np.newaxis, ...]
        if chosen_vectors.shape[1] < self.K:
            column_gap = int(self.K - chosen_vectors.shape[1])
            infty_array = np.full((1, column_gap, self.d), -np.inf)
            chosen_vectors_padding = np.concatenate((chosen_vectors, infty_array), axis=1)
        else:
            chosen_vectors_padding = chosen_vectors

        self.X = np.concatenate((self.X, chosen_vectors_padding))
        self.V += np.matmul(x[self.S,:].T, x[self.S,:])
        return(self.S)

    def update_theta(self,Y,t):
        """
        update parameter
        """
        Y=Y[np.newaxis, ...]
        if Y.shape[1] < self.K+1:
            column_gap = int(self.K+1 - Y.shape[1])
            infty_array = np.full((1, column_gap), -np.inf)
            Y_padding = np.concatenate((Y, infty_array), axis=1)
        else:
            Y_padding = Y

        self.Y = np.concatenate((self.Y, Y_padding))
        if t==2:
            self.X = np.delete(self.X, (0), axis=0)
            self.Y = np.delete(self.Y, (0), axis=0)
        # if t>self.T0:
        self.mnl.fit(self.X, self.Y, self.theta, self.lam)
        self.theta = self.mnl.w

# TS-MNL with Gaussian approximation
class ts_mnl:
    def __init__(self, N, K, d, kappa, alpha=None, lam = 1.0):
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
        
    def choose_S(self,t,x,reward):  # x is N*d matrix
        """
        choose the optimistic assortment
        """
        if self.alpha is None:
            self.alpha = (1/(2*self.kappa))*np.sqrt(2*self.d*np.log(1+t/self.d)+2*np.log(t))/10
        theta_tilde = np.random.multivariate_normal(self.theta, np.square(self.alpha)*inv(self.V))
        means = np.dot(x,theta_tilde)            

        # Define the problem
        lp_prob = pulp.LpProblem('Maximize_Revenue', pulp.LpMaximize)
        clipped_means = np.clip(means, 0, 2)	
        w_ti = np.exp(clipped_means)

        # Decision variables
        p_ti = pulp.LpVariable.dicts('p_ti', range(self.N), lowBound=0, upBound=1)
        p_t0 = pulp.LpVariable('p_t0', lowBound=0, upBound=1)

        # Objective Function
        lp_prob += pulp.lpSum([reward[i] * p_ti[i] for i in range(self.N)])

        # Constraints
        lp_prob += pulp.lpSum([p_ti[i] for i in range(self.N)]) + p_t0 == 1
        lp_prob += pulp.lpSum([p_ti[i] * (1.0 / w_ti[i]) for i in range(self.N)]) <= self.K * p_t0

        # The third constraint for each item
        for i in range(self.N):
            lp_prob += p_ti[i] * (1.0 / w_ti[i]) <= p_t0
            lp_prob += p_ti[i] * (1.0 / w_ti[i]) >= 0
        # Solve the problem
        lp_prob.solve(pulp.PULP_CBC_CMD(msg=False))

        chosen_items = list()
        for i, variable in enumerate(lp_prob.variables()):
            if variable.varValue > 0.00001 and variable.name != 'p_t0':
                chosen_items.append(int(variable.name[5:]))

        self.S = np.array(chosen_items)

        chosen_vectors = x[self.S,:][np.newaxis, ...]
        if chosen_vectors.shape[1] < self.K:
            column_gap = int(self.K - chosen_vectors.shape[1])
            infty_array = np.full((1, column_gap, self.d), -np.inf)
            chosen_vectors_padding = np.concatenate((chosen_vectors, infty_array), axis=1)
        else:
            chosen_vectors_padding = chosen_vectors
        self.X = np.concatenate((self.X, chosen_vectors_padding))
        self.V += np.matmul(x[self.S,:].T, x[self.S,:])
        return(self.S)

    def update_theta(self,Y,t):
        """
        update parameter
        """
        Y=Y[np.newaxis, ...]
        if Y.shape[1] < self.K+1:
            column_gap = int(self.K+1 - Y.shape[1])
            infty_array = np.full((1, column_gap), -np.inf)
            Y_padding = np.concatenate((Y, infty_array), axis=1)
        else:
            Y_padding = Y

        self.Y = np.concatenate((self.Y, Y_padding))
        if t==2:
            self.X = np.delete(self.X, (0), axis=0)
            self.Y = np.delete(self.Y, (0), axis=0)
        # if t>self.T0:
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