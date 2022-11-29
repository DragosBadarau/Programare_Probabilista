import matplotlib.pyplot as plt
import numpy as np
import pymc3 as pm
import pandas as pd
import arviz as az

if __name__ == '__main__':
    df=pd.read_csv("Admission.csv")
    admission = df['Admission'].values
    gre = df['GRE'].values
    gpa = df['GPA'].values
    y_1 = pd.Categorical(df['Admission']).codes
    x_n = ['GRE', 'GPA']
    first_axis_x1 = df[x_n].values
    with pm.Model() as model_1:
        alpha = pm.Normal('alpha', mu=0, sd=1)
        β = pm.Normal('β', mu=0, sd=2, shape=len(x_n))
        μ = alpha + pm.math.dot(first_axis_x1, β)
        sigma = pm.HalfNormal('sigma', sd=1)
        θ = pm.Deterministic('θ', 1 / (1 + pm.math.exp(-μ)))
        bd = pm.Deterministic('bd', -alpha/β[1] - β[0]/β[1] * first_axis_x1[:,0])
        bd1 = pm.Deterministic('bd1', (alpha / β) * -1)
        bd2 = pm.Deterministic('bd2', (alpha / β) * -1)
        admission_obs = pm.Normal('admission_obs', mu=μ, sd=sigma, observed=admission)
        yl = pm.Bernoulli('yl', p=θ, observed=y_1)
        trace = pm.sample(2000, tune=1000, return_inferencedata=True)
    sort_idx = np.argsort(first_axis_x1[:,0])
    posterior_0 = trace.posterior.stack(samples=("chain", "draw"))
    bd = trace.posterior['bd'].mean(("chain", "draw"))[sort_idx]
    theta = posterior_0['mu'].mean("samples")
    idx = np.argsort(gre)
    plt.plot(gre[idx], theta[idx], color='C2', lw=3)
    plt.scatter(first_axis_x1[:,0], first_axis_x1[:,1], c=[f'C{x}' for x in y_1])
    plt.plot(first_axis_x1[:,0][sort_idx], bd, color='k')
    az.plot_hdi(first_axis_x1[:,0], trace.posterior['bd'], color='k')
    az.plot_posterior({"result": trace['mu']},hdi_prob=0.9)
    plt.xlabel(x_n[0])
    plt.ylabel(x_n[1])
