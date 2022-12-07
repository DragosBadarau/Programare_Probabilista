import numpy as np
import pymc3 as pm

if __name__ == 'main':
    register_count = 2
    number_of_tables = 5
    time_spent_cooking = time_spent_ordering = time_spent_eating = counter = counter_tables = 0
    cooking_station_count = 3
    trafic = np.random.poisson(20, 100)
    for i in range(len(trafic)):
        with pm.Model() as model:
            order = pm.Normal('O', sigma=0.5, mu=1)
            cook = pm.Exponential('C', 0.5)
            eat = pm.Normal('E', mu=10, sigma=2)
        bonus_dictionary = \
            {
            'order': trace['O'].tolist(),
            'cook': trace['C'].tolist(),
            'eat': trace['E'].tolist()
            }
        for key, value in bonus_dictionary:
            if key == 'order':
                time_spent_ordering += value
            if key == 'cook':
                time_spent_cooking += value
            if key == 'eat':
                time_spent_eating += value
        trace = pm.sample(trafic[i], chains=1, model=model)
        # 1)
        time_spent_cooking /= cooking_station_count
        time_spent_ordering /= register_count
        time_waiting = time_spent_ordering + time_spent_cooking
        if time_waiting <= 60:
            counter += 1
        # 2)
        time_spent_eating /= number_of_tables
        if time_spent_eating <= 60:
            counter_tables += 1
    print(counter / 100)
    print(counter_tables / 100)
