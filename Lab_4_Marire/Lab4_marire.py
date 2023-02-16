import pymc3 as pm
import numpy as np
lambda_ = 20  # parameter of Poisson distribution
theta = 0.5   # probability of making a purchase
alpha_0 = 5   # mean time spent in store when not shopping
alpha_1 = 6   # mean time spent in store when shopping
n = pm.Poisson('n', lambda_)    # number of customers entering the store
p = pm.Beta('p', alpha=1, beta=1)    # probability of making a purchase
y = pm.Binomial('y', n=n, p=p, shape=len(n))  # number of customers making a purchase
alpha = pm.Deterministic('alpha', alpha_0 + y*(alpha_1 - alpha_0))  # time spent in store
t = pm.Exponential('t', lam=1/alpha, shape=len(n))    # time spent in store for each customer
model = pm.Model()
# a)
with model:
    n = pm.Poisson('n', lambda_)
    p = pm.Beta('p', alpha=1, beta=1)
    y = pm.Binomial('y', n=n, p=p, shape=len(n))
    alpha = pm.Deterministic('alpha', alpha_0 + y*(alpha_1 - alpha_0))
    t = pm.Exponential('t', lam=1/alpha, shape=len(n))
    trace = pm.sample(1000, tune=1000)

# b)
theta_values = [0.2, 0.5]   # values of theta to consider
y_obs = np.array([10, 20, 30, 40, 50])   # observed number of customers making a purchase
lambda_ = 20
alpha_0 = 5
alpha_1 = 6
alpha_max = 50

with pm.Model() as model:
    alpha = pm.Uniform('alpha', lower=0, upper=alpha_max, shape=len(theta_values))
    p = pm.Beta('p', alpha=1, beta=1, shape=len(theta_values))
    n = pm.Poisson('n', lambda_, shape=len(y_obs))
    y = pm.Binomial('y', n=n, p=p, shape=(len(y_obs), len(theta_values)))
    t = pm.Exponential('t', lam=1/(alpha_0 + (alpha_1 - alpha_0)*y), shape=(len(y_obs), len(theta_values)))
def posterior_prob(alpha, t):
    return (1 - np.exp(-15/alpha))*(t > 15).mean()
with model:
    for i, theta in enumerate(theta_values):
        p.tag.test_value[i] = theta
    trace = pm.sample(1000, tune=1000, chains=2)
    pm.traceplot(trace)

posterior_probs = np.zeros((len(alpha), len(theta_values)))
for i, alpha_i in enumerate(alpha):
    for j, theta_j in enumerate(theta_values):
        posterior_probs[i, j] = posterior_prob(alpha_i, trace['t'][:, i, j])

print(posterior_probs)
