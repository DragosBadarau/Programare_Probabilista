import numpy as np
from scipy import stats

import matplotlib.pyplot as plt
import arviz as az

np.random.seed(0)

var1 = stats.norm.rvs(10, 100, size=1000)  # Distributie normala cu media 10 si deviatie standard 100, 1000 samples
var2 = stats.uniform.rvs(0, 100, size=1000)  # Distributie uniforma intre 0 si 100, 1000 samples . Primul parametru fiind limita inferioara a intervalului, al doilea parametru fiind diferenta dintre limita inferioara si cea superioara a intervalului.

az.plot_posterior({'var1': var1, 'var2': var2})
plt.show()
