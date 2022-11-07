import numpy as np
import csv
import arviz as az
import pymc3 as pm
import matplotlib.pyplot as plt
import statistics as stt

with open(r'data.csv', 'r') as file:
    data = csv.reader(file)
    ppvt = []
    educ_cat = []
    mom_age = []
    ok = 0
    for row in data:
        if ok:
            ok = 1
        elif ok == 1:
            ppvt.append(int(row[1]))
            educ_cat.append(int(row[2]))
            mom_age.append(int(row[3]))
        az.plot_posterior({'ppvt': ppvt, 'educ_cat': educ_cat, "mom_age": mom_age})
    avg_ppvt = sum(ppvt) / 400
    avg_mom_age = sum(mom_age) / 400
    avg_educ_cat = sum(educ_cat) / 400
    stdev_ppvt =stt.stdev(ppvt)
    stdev_educ_cat = stt.stdev(educ_cat)
    stdev_mom_age = stt.stdev(mom_age)
ppvt = np.array(ppvt)
educ_cat = np.array(educ_cat)
mom_age = np.array(mom_age)
mom_age.sort()
#1
plt.scatter(ppvt, mom_age)
plt.xlabel('ppvt')
plt.ylabel('mom_age')
plt.show()

#2
# bayesian regression line
with pm.Model() as model_g:
    alpha = pm.Normal("alpha", mu=avg_ppvt, sigma=stdev_ppvt)
    beta = pm.Normal("beta", mu=avg_mom_age, sigma=stdev_mom_age)
    sigma = pm.HalfNormal("sigma", sigma=stdev_ppvt)
    index = alpha + beta * mom_age
    mu = pm.Deterministic('μ', index)
    Y_obs = pm.Normal("Y_obs", mu=mu, sigma=sigma, observed=ppvt)
    idata_g = pm.sample(1000, tune=1000, return_inferencedata=True)
model_educ = pm.Model()
with model_educ:
    alpha = pm.Normal("alpha", mu=avg_ppvt, sigma=stdev_ppvt)
    beta = pm.Normal("beta", mu=avg_educ_cat, sigma=stdev_educ_cat)
    sigma = pm.HalfNormal("sigma", sigma=stdev_ppvt)
    ppc = pm.sample_posterior_predictive(idata_g, samples=100, model=model_educ)
    index = alpha + beta * model_educ
    plt.plot(model_educ, index, c='k',
            label=f'ppvt = {alpha:.2f} + {beta:.2f} * educ_cat')
    az.plot_hdi(mom_age, ppc['ppvt_pred'], hdi_prob=0.97, color='gray')
    plt.show()