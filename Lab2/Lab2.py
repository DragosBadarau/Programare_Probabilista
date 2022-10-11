import numpy as np
from scipy.stats import expon, gamma
from numpy import random
import statistics
import matplotlib.pyplot as plt
import arviz as az
import seaborn as sns


def ex1():
    final_values = []
    for i in range(0, 10000):
        if random.uniform(0, 1) > 0.4:
            final_values.append(expon.rvs(0, 0.16, 1)[0])
        else:
            final_values.append(expon.rvs(0, 0.25, 1)[0])
    sns.set_style('whitegrid')
    print(statistics.mean(final_values))
    print(statistics.stdev(final_values))
    sns.kdeplot(np.array(final_values), bw=0.5)


# ex1()

# loc -> mean
# scale -> standard deviation
def ex2():
    all_processes = []
    prob_3ms = 0
    for i in range(1000):
        r = random.uniform(0, 1)
        if r < 0.25:
            process = expon.rvs(scale=0.25, loc=0, size=1)[0] + gamma.rvs(4, 0, scale=0.33, size=1)
            all_processes.append(process)
        elif 0.25 <= r < 0.5:
            process = expon.rvs(scale=0.25, loc=0, size=1)[0] + gamma.rvs(4, 0, scale=0.50, size=1)
            all_processes.append(process)
        elif 0.5 <= r < 0.8:
            process = expon.rvs(scale=0.25, loc=0, size=1)[0] + gamma.rvs(5, 0, scale=0.50, size=1)
            all_processes.append(process)
        else:
            process = expon.rvs(scale=0.25, loc=0, size=1)[0] + gamma.rvs(5, 0, scale=0.33, size=1)
            all_processes.append(process)
        if process > 3:
            prob_3ms += 1
    az.plot_posterior({'Density distribution of the processes': all_processes})
    print(f'Probability that the process takes longer than 3 ms: {prob_3ms / 10}%')
    plt.show()


#ex2()


def ex3():
    bb = []
    ss = []
    sb = []
    bs = []
    experimentBB = experimentBS = experimentSB = experimentSS = 0
    for experiment in range(1, 100):
        for aruncari in range(1, 10):
            masluit = random.uniform(0, 1)
            nemasluit = random.uniform(0, 1)
            print(masluit, nemasluit)
            if masluit < 0.5 and nemasluit >= 0.3:
                experimentBB += 1
            elif masluit < 0.5 and nemasluit < 0.3:
                experimentBS += 1
            elif masluit >= 0.5 and nemasluit >= 0.3:
                experimentSB += 1
            else:
                experimentSS += 1
        bb.append(experimentBB)
        bs.append(experimentBS)
        sb.append(experimentSB)
        ss.append(experimentSS)
        experimentBB = 0
        experimentSS = 0
        experimentBS = 0
        experimentSB = 0
    az.plot_posterior({'ss': ss})
    az.plot_posterior({'sb': sb})
    az.plot_posterior({'bs': bs})
    az.plot_posterior({'bb': bb})
    plt.show()


#ex3()

