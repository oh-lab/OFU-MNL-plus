import numpy as np
import random
import time
import pulp
from models_non_uniform import ucb_mnl, ts_mnl, ofu_mnl_plus
import argparse

parser = argparse.ArgumentParser(description='mnl bandit for non-uniform rewards')

parser.add_argument('--N', type = int, default=100, help='number of base items')
parser.add_argument('--K', type = int, default=5, help='size of assortment')
parser.add_argument('--T', type = int, default=3000, help='horizon')
parser.add_argument('--d', type = int, default=5, help='feature dimension')
parser.add_argument('--dist', type = int, default=0, help='context distribution - 0:gaussian, 1:uniform, 2:elliptical')
parser.add_argument('--id', type = int, default=999, help='job ID')


class mnlEnv:
	def __init__(self, N, theta, K):
		super(mnlEnv, self).__init__()
		self.N = N
		self.theta = theta
		self.K = K
        
	def compute_rwd(self, means, S, reward):
		u = np.exp(means)
		chosen_rewards = reward[S]
		uSum = 1 + u.sum()
		prob = np.append(u, [1])/uSum
		rwd = np.multiply(u,chosen_rewards).sum()/uSum
		Y = np.random.multinomial(1, prob)
		return rwd, Y
    
	def get_opt_rwd(self, x, reward):
		# Define the problem
		lp_prob = pulp.LpProblem('Maximize_Revenue', pulp.LpMaximize)	
		w_ti = np.exp(np.dot(x,self.theta))

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

		#calculate optimal revenue
		optimal_assortment = np.array(chosen_items)
		opt_means = np.dot(x[optimal_assortment,:],self.theta)
		opt_revenue, Y = self.compute_rwd(opt_means, optimal_assortment, reward)
		
		return opt_revenue, optimal_assortment

def sample_spherical(N, k):
    vec = np.random.randn(k, N)
    vec /= np.linalg.norm(vec, axis=0)
    return vec

def sample_elliptical(N, d, k, mu):
    S = sample_spherical(N, k)
    A = np.random.rand(d,k)
    R = np.random.normal(size=N)
    return mu + A.dot(R*S)


def main():
	args = parser.parse_args()
	random.seed(args.id)

	N = args.N
	K = args.K
	d = args.d
	T = args.T
	dist = args.dist

	sigma_sq=1.
	rho_sq= 0
	W=(sigma_sq-rho_sq)*np.eye(N) + rho_sq*np.ones((N,N))
	kappa = np.exp(-1) / (1+K * np.exp(1))**2 

	theta=np.random.uniform(-1./np.sqrt(d),1./np.sqrt(d),d) 


	regret_savename = "./results/mnlBandit_N={}_K={}_d={}_dist={}_id={}_regret_nu_scaled.csv".format(N, K, d, args.dist, args.id)
	runtime_savename = "./results/mnlBandit_N={}_K={}_d={}_dist={}_id={}_runtime_nu_scaled.csv".format(N, K, d, args.dist, args.id)
	cum_runtime_savename = "./results/mnlBandit_N={}_K={}_d={}_dist={}_id={}_cum_runtime_nu_scaled.csv".format(N, K, d, args.dist, args.id)

	simul_n=1

	cumulated_regret = [[] for i in range(3)]

	for simul in range(simul_n):

		env=mnlEnv(N, theta, K)
		M1=ucb_mnl(N=N, K=K, d=d, kappa = kappa)
		M2=ts_mnl(N=N, K=K, d=d, kappa = kappa)
		M3=ofu_mnl_plus(N=N, K=K, d=d, kappa = kappa)

		RWD1=list()
		RWD2=list()
		RWD3=list()
		optRWD=list()

		TIME1=list()
		TIME2=list()
		TIME3=list()

		for t in range(T):
			reward=np.random.uniform(0,1,N)
			# reward=np.ones(N)

			if dist == 0:
				x=np.random.multivariate_normal(np.zeros(N),W,d).T
			elif dist == 1:
				x=(np.random.random((N, d)) * 2 - 1)/np.sqrt(d)
			elif dist == 2:
				x=sample_elliptical(N, d, int(d/2), 0).T
			x = np.clip(x, -1./np.sqrt(d), 1./np.sqrt(d))

			start_time = time.time()
			S1=M1.choose_S(t+1,x,reward)
			rwd1, Y1 = env.compute_rwd(np.dot(x[S1,:],theta), S1, reward)
			RWD1.append(rwd1)
			M1.update_theta(Y1,t+1)
			TIME1.append(time.time() - start_time)

			start_time = time.time()
			S2=M2.choose_S(t+1,x,reward)
			rwd2, Y2 = env.compute_rwd(np.dot(x[S2,:],theta), S2, reward)
			RWD2.append(rwd2)
			M2.update_theta(Y2,t+1)
			TIME2.append(time.time() - start_time)

			start_time = time.time()
			S3=M3.choose_S(t+1,x,reward)
			rwd3, Y3 = env.compute_rwd(np.dot(x[S3,:],theta), S3, reward)
			RWD3.append(rwd3)
			M3.update_state(Y3)
			TIME3.append(time.time() - start_time)

			opt_rwd, opt_S = env.get_opt_rwd(x, reward)
			print("Time: ", t, "Regret:", np.round(opt_rwd - rwd1,2), np.round(opt_rwd - rwd2,2), np.round(opt_rwd - rwd3,2))
			optRWD.append(opt_rwd)
	    
		cumulated_regret[0].append(np.cumsum(optRWD)-np.cumsum(RWD1))
		cumulated_regret[1].append(np.cumsum(optRWD)-np.cumsum(RWD2))
		cumulated_regret[2].append(np.cumsum(optRWD)-np.cumsum(RWD3))

	regret = np.vstack([
		np.asarray(cumulated_regret[0]),
		np.asarray(cumulated_regret[1]),
		np.asarray(cumulated_regret[2])
		])
	np.savetxt(regret_savename, regret, delimiter=",")

	runtime = np.vstack([
		np.asarray(TIME1),
		np.asarray(TIME2),
		np.asarray(TIME3)
		])
	np.savetxt(runtime_savename, runtime, delimiter=",")

	cum_runtime = np.vstack([
		np.cumsum(TIME1),
		np.cumsum(TIME2),
		np.cumsum(TIME3)
		])
	np.savetxt(cum_runtime_savename, cum_runtime, delimiter=",")

	
if __name__ == '__main__':
	main()
