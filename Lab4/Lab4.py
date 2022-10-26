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
        clients = pm.Poisson('C', mu=20)
        order_pay = pm.Normal('op', mu=1, sigma=0.5)
        prepare = pm.Exponential('p', 1 / alfa)
        trace = pm.sample(1000)
        time = pm.Normal('T', order_pay + prepare)

        dictionary = {
                'clients': trace['C'].tolist(),
                'order_pay': trace['op'].tolist(),
                'prepare': trace['p'].tolist(),
                'time': trace['T'].tolist()
        }
        df = pd.DataFrame(dictionary)
        print(df[(df['time'] <= 15)].shape[0] / df.shape[0])
        #timpul de servire mai mic de 15 minute
        total_time = 0
        number_of_clients = df.shape[0]
        for each_time in number_of_clients:
            total_time = total_time + df.at[each_time,'time']
        print(total_time/number_of_clients)