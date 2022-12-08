import pymc3 as pm
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import arviz as az


# subpunctul 1)
def model(beta):
    with pm.Model() as model_p:
        α = pm.Normal('α', mu=0, sd=1)
        β = pm.Normal('β', mu=0, sd=beta, shape=order)
        ε = pm.HalfNormal('ε', 5)
        μ = α + pm.math.dot(β, x_1s)
        y_pred = pm.Normal('y_pred', mu=μ, sd=ε, observed=y_1s)
        idata_p = pm.sample(2000, return_inferencedata=True)

    x_new = np.linspace(x_1s[0].min(), x_1s[0].max(), 100)
    α_p_post = idata_p.posterior['α'].mean(("chain", "draw")).values
    β_p_post = idata_p.posterior['β'].mean(("chain", "draw")).values
    idx = np.argsort(x_1s[0])
    y_p_post = α_p_post + np.dot(β_p_post, x_1s)
    plt.plot(x_1s[0][idx], y_p_post[idx], 'C2', label=f'model order {order}')
    plt.scatter(x_1s[0], y_1s, c='C0', marker='.')
    plt.legend()


az.style.use('arviz-darkgrid')
data = np.loadtxt('date.csv')
x_1 = data[:, 0]
y_1 = data[:, 1]

order = 5
x_1p = np.vstack([x_1 ** i for i in range(1, order + 1)])
x_1s = (x_1p - x_1p.mean(axis=1, keepdims=True)) / x_1p.std(axis=1, keepdims=True)
x_1p.std(axis=1, keepdims=True)
y_1s = (y_1 - y_1.mean()) / y_1.std()
plt.scatter(x_1s[0], y_1s)
plt.xlabel('x')
plt.ylabel('y')
# a)
model(10)
# b)
model(100)
model(np.array([10, 0.1, 0.1, 0.1, 0.1]))

# subpunctul 2)
# am crescut nr de date la 500 puncte luand in considerare intervalul datelor iniitale
x_2 = np.random.uniform(low=min(x_1), high=max(x_1), size=(500,))
y_2 = np.random.uniform(low=min(y_1), high=max(y_1), size=(500,))
x_2p = np.vstack([x_2 ** i for i in range(1, order + 1)])
x_2s = (x_2p - x_2p.mean(axis=1, keepdims=True)) / x_2p.std(axis=1, keepdims=True)
x_2p.std(axis=1, keepdims=True)
y_2s = (y_2 - y_2.mean()) / y_2.std()
plt.scatter(x_2s[0], y_2s)

# subpunctul 3)
order = 2
idata_p = pm.sample(2000, return_inferencedata=True)
waic_p = az.waic(idata_p, scale="deviance")
loo_p = az.loo(idata_p, scale="deviance")
plt.show()
