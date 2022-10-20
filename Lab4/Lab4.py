from numpy import random
import matplotlib.pyplot as plt
import seaborn as sns
import pymc3 as pm
import numpy as np
import pandas as pd

#a)

def FastFoodProbability(alfa):

    traffic = np.random.poisson(20, 1)
    sns.distplot(random.normal(loc=60, scale=30, size=traffic), hist=False, label='normal')#every hour
    #sns.distplot(random.poisson(lam=20, size=alfa),  hist=False, label='poisson')

    # plt.show()

    model = pm.Model()
    with model:
        trafic = np.random.poisson(20, 1)
        order_pay = pm.Normal('op', mu=1, sigma=0.5)
        prepare = pm.Exponential('pC', 1 / alfa)
        trace = pm.sample(trafic[0], chains=1)

        dictionary = {
                'order_pay': trace['op'].tolist(),
                'prepare': trace['p'].tolist()
        }
        df = pd.DataFrame(dictionary)

def clients(nr_clients):
    for client in range(nr_clients):
        alfa = random.randint(0, 60)
        FastFoodProbability(alfa)