import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc3 as pm
import arviz as az
import math

if __name__ == '__main__':
    np.random.seed(1)
    df = pd.read_csv('Prices.csv')
    Price = df['Price'].values
    Speed = df['Speed'].values
    HardDrive = df['HardDrive'].values
    Ram = df['Ram'].values
    Premium = df['Premium'].values
    # ex1
    with pm.Model() as model:
        α_tmp = pm.Normal('α', mu=0, sd=1)
        beta1 = pm.Normal('beta1', mu=0, sd=1)
        beta2 = pm.Normal('beta2', mu=0, sd=1)
        sigma = pm.HalfNormal('sigma', sd=1)
        μ = pm.Deterministic('μ ', α_tmp + beta1 * Speed + beta2 * [math.log(i) for i in HardDrive])
        Price_obs = pm.Normal('price_obs', mu=μ, sigma=sigma, observed=Price)
        trace = pm.sample(5000, tune=1000)
        predictive = pm.sample_posterior_predictive(trace, var_names=["mu"], samples=5000)
        
    # ex2
    az.plot_posterior(
        {"beta1": trace['beta1'], "beta2": trace['beta2']}, hdi_prob=0.95)
    plt.savefig("95%beta1,2.png")

    # ex3
    with pm.Model() as model:
        α_tmp = pm.Normal('α', mu=0, sd=1)
        beta = pm.Normal('beta', mu=0, sd=1)
        sigma = pm.HalfNormal('sigma', sd=1)
        μ = pm.Deterministic('mu', α_tmp + beta)
        Price_obs = pm.Normal('price_obs', mu=μ, sigma=sigma, observed=Price)
        trace = pm.sample(5000, tune=1000)
    az.plot_posterior({"betaa": trace['beta']})
    # ex4
    az.plot_posterior({"pret de vanzare": trace['μ ']}, hdi_prob=0.9)
    plt.show()
