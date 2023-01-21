import scipy.stats as stats
import numpy as np
import pymc3 as pm
import matplotlib.pyplot as plt


def A():
    # Distribuțiile Poisson, uniforma, normala, skew normala
    # subpunct 1
    poisson = stats.poisson(mu=5)
    uniform = stats.uniform(loc=10, scale=20)
    normal = stats.norm(loc=3000, scale=10)
    skewnorm = stats.skewnorm(a=0, loc=70, scale=10)
    # a specifica asimetria
    # generam esantioane de dimensiune cate 1000
    oameni_cafenea = poisson.rvs(size=1000)
    greutate_caini = uniform.rvs(size=1000)
    greutate_elefanti = normal.rvs(size=1000)
    greutate_adulti = skewnorm.rvs(size=1000)
    # Vizualizam esantionul folosind histograma
    # subpunct 2
    plt.hist(oameni_cafenea, bins=20, label="Vizite cafenea")
    plt.hist(greutate_caini, bins=20, label="Greutate caini")
    plt.hist(greutate_elefanti, bins=20, label="Greutate elefanti")
    plt.hist(greutate_adulti, bins=20, label="Greutate adulti")
    plt.legend()
    plt.show()


def B():
    # subpunct 1
    with pm.Model() as model:
        p_theta = np.array([0.2, 0.5])
        θ = pm.Categorical("θ", p=p_theta)
        n = pm.Poisson("n", mu=10)
        y = pm.Binomial("y", n=n, p=θ, observed=[0, 5, 10])
        trace = pm.sample(1000)
        pm.plot_posterior(trace)


# subpunct 2
# Efectul lui Y : Numarul de clienti care cumpara produsul va influenta distributia a posteriori pentru n
# si θ in sensul in care, pentru Y mare distributia va tinde sa se concentreze asupra valorii reale pentru n si θ,
# ingustandu-se.
# Efectul lui θ : probabilitatea ca un client sa cumpere produsul influenteaza distributia a posteiori
# pentru n daca aceasta creste, fiindca se va concentra in jurul valorii reale n, ingustandu-se


def C():
    # esantion aleator random cu distributie binomiala
    sample = np.random.binomial(n=1, p=0.5, size=150)
    with pm.Model() as model:
        alpha = pm.Uniform('alpha', 0, 10)
        beta = pm.Uniform('beta', 0, 10)
        y = pm.BetaBinomial('y', alpha=alpha, beta=beta, n=1, observed=sample)
        trace_standard = pm.sample()
        # sampler-ul Metropolis
        trace_mh = pm.sample(step=pm.Metropolis())
        plt.figure(figsize=(10, 5))
        pm.traceplot(trace_standard)
        plt.show()
        plt.figure(figsize=(10, 5))
        pm.traceplot(trace_mh)
        plt.show()
        plt.figure(figsize=(10, 5))
        pm.plot_posterior(trace_standard)
        plt.show()
        plt.figure(figsize=(10, 5))
        pm.plot_posterior(trace_mh)
        plt.show()


if __name__ == '_main_':
    A()  # Ex1
    B()  # Ex2
    C()  # Ex3
