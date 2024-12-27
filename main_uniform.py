import numpy as np
import random
import time
from models_uniform import ucb_mnl, ts_mnl, ofu_mnl_plus
import argparse

parser = argparse.ArgumentParser(description='mnl bandit for uniform rewards')

parser.add_argument('--N', type = int, default=100, help='number of base items')
parser.add_argument('--K', type = int, default=15, help='size of assortment')
parser.add_argument('--T', type = int, default=3000, help='horizon')
parser.add_argument('--d', type = int, default=5, help='feature dimension')
parser.add_argument('--dist', type = int, default=0, help='context distribution - 0:gaussian, 1:uniform, 2:elliptical')
parser.add_argument('--id', type = int, default=999, help='job ID')
parser.add_argument('--vzero', type = float, default=1.0, help='utility for the outisde option')


class mnlEnv:
	def __init__(self, theta, K, vzero):
		super(mnlEnv, self).__init__()
		self.theta = theta
		self.K = K
		self.vzero = vzero
        
	def compute_rwd(self, means):
		u = np.exp(means)
		uSum = self.vzero + u.sum()
		prob = np.append(u, [self.vzero])/uSum
		rwd = u.sum()/uSum
		Y = np.random.multinomial(1, prob)
		return rwd, Y
    
	def get_opt_rwd(self, x):
		opt_means = np.sort(np.dot(x,self.theta))[::-1][:self.K]
		rwd, Y = self.compute_rwd(opt_means)
		return rwd

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
	vzero = args.vzero
	dist = args.dist
	sigma_sq=1.
	rho_sq= 0
	W=(sigma_sq-rho_sq)*np.eye(N) + rho_sq*np.ones((N,N))
	kappa = np.exp(-1) / (vzero +K * np.exp(1))**2 

	theta=np.random.uniform(-1./np.sqrt(d),1./np.sqrt(d),d) 

	if args.vzero != 1:
		regret_savename = "./results/mnlBandit_N={}_K={}_d={}_dist={}_id={}_v0={}_regret.csv".format(N, K, d, args.dist, args.id, args.vzero)
		runtime_savename = "./results/mnlBandit_N={}_K={}_d={}_dist={}_id={}_v0={}_runtime.csv".format(N, K, d, args.dist, args.id, args.vzero)
		cum_runtime_savename = "./results/mnlBandit_N={}_K={}_d={}_dist={}_id={}_v0={}_cum_runtime.csv".format(N, K, d, args.dist, args.id, args.vzero)
	else:
		regret_savename = "./results/mnlBandit_N={}_K={}_d={}_dist={}_id={}_regret.csv".format(N, K, d, args.dist, args.id)
		runtime_savename = "./results/mnlBandit_N={}_K={}_d={}_dist={}_id={}_runtime.csv".format(N, K, d, args.dist, args.id)
		cum_runtime_savename = "./results/mnlBandit_N={}_K={}_d={}_dist={}_id={}_cum_runtime.csv".format(N, K, d, args.dist, args.id)		

	simul_n=1

	cumulated_regret = [[] for i in range(3)]

	for simul in range(simul_n):

		env=mnlEnv(theta, K, vzero)
		M1=ucb_mnl(N=N, K=K, d=d, kappa = kappa)
		M2=ts_mnl(N=N, K=K, d=d, kappa = kappa)
		M3=ofu_mnl_plus(N=N, K=K, d=d, kappa = kappa, vzero = vzero)

		RWD1=list()
		RWD2=list()
		RWD3=list()
		optRWD=list()

		TIME1=list()
		TIME2=list()
		TIME3=list()

		for t in range(T):
			if dist == 0:
				x=np.random.multivariate_normal(np.zeros(N),W,d).T
			elif dist == 1:
				x=(np.random.random((N, d)) * 2 - 1)/np.sqrt(d)
			elif dist == 2:
				x=sample_elliptical(N, d, int(d/2), 0).T
			x = np.clip(x, -1./np.sqrt(d), 1./np.sqrt(d))

			start_time = time.time()
			S1=M1.choose_S(t+1,x)
			rwd1, Y1 = env.compute_rwd(np.dot(x[S1,:],theta))
			RWD1.append(rwd1)
			M1.update_theta(Y1,t+1)
			TIME1.append(time.time() - start_time)

			start_time = time.time()
			S2=M2.choose_S(t+1,x)
			rwd2, Y2 = env.compute_rwd(np.dot(x[S2,:],theta))
			RWD2.append(rwd2)
			M2.update_theta(Y2,t+1)
			TIME2.append(time.time() - start_time)

			start_time = time.time()
			S3=M3.choose_S(t+1,x)
			rwd3, Y3 = env.compute_rwd(np.dot(x[S3,:],theta))
			RWD3.append(rwd3)
			M3.update_state(Y3)
			TIME3.append(time.time() - start_time)

			opt_rwd = env.get_opt_rwd(x)
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
