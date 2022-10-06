import numpy as np
from scipy import stats
from numpy import random
import matplotlib.pyplot as plt
import arviz as az
import seaborn as sns

"""
np.random.seed(1)

x = stats.norm.rvs(0, 1, size=10000) # Distributie normala cu media 0 si deviatie standard 1, 1000 samples
y = stats.uniform.rvs(-1, 2, size=10000) # Distributie uniforma intre -1 si 1, 1000 samples . Primul parametru fiind
# limita inferioara a intervalului, al doilea parametru fiind "marimea" intervalului, aka [-1,-1+2] = [-1,1]
z = x+y # Compunerea prin insumare a celor 2 distributii

az.plot_posterior({'x':x,'y':y,'z':z}) # Afisarea aproximarii densitatii probabilitatilor, mediei, intervalului etc.
# variabilelor x,y,z
plt.show()
"""
X1 = random.exponential(1/4,10000)
x2 = random.exponential(1/6,10000)


final_values = []

for i in range(0,10000):
    if random.uniform(0,1)>0.4:
        final_values.append(x2[i])
    else:
        final_values.append(X1[i])
mean_of_X1 = np.array(final_values).mean()
sd_of_X1 = np.std(final_values)
sns.set_style('whitegrid')
sns.kdeplot(np.array(final_values), bw=0.5)
#loc -> mean
#scale -> standard deviation