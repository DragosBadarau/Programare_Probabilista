import matplotlib.pyplot as plt
import pandas as pd
import pymc3 as pm
import numpy as np
import statistics
df = pd.read_csv("auto-mpg.csv")
print(df)
ppvt = []
mpg = df["mpg"]
CP = df["horsepower"]
# a)

plt.scatter(mpg,CP)
plt.xlabel("CP")
plt.ylabel('mpg')
plt.title('Relatia de dependenta dintre CP si mpg')
plt.show()

# b)
stdev_mpg = statistics.stdev(mpg)
stdev_CP = statistics.stdev(CP)
model = pm.Model()
with model:
    α = pm.Normal('α', mu=0, sd=stdev_CP)
    β = pm.Normal('β', mu=0, sd=stdev_CP)
    eps = pm.HalfCauchy('eps', beta=10)
    mu = α + β * CP
    # Likelihood (sampling distribution) of observations
    y_obs = pm.Normal('y_obs', mu=mu, sd=eps, observed=mpg)
    idata_g = pm.sample(2000, tune=2000, return_inferencedata=True)
#c)
map_estimate = pm.find_MAP(model=model)
ppc = pm.sample_posterior_predictive(idata_g, samples=100, model=model)
