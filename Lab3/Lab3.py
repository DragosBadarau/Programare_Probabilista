from numpy import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc3 as pm
import arviz as az


def depozit():
    # Ex 1
    """
    Problema propriu-zisa cu if-uri
    """
    prob_cutremur = 0.0005
    prob_incendiu = 0.01  # daca cutremur atunci 0.03
    alarma_accidental = 0.0001  # daca cutremur atunci 0.02
    alarma_incendiu = 0.95  # daca daca cutremur si incendiu 0.98

    cutremur = random.uniform(0, 1)
    incendiu = random.uniform(0, 1)
    alarma = random.uniform(0, 1)

    are_loc_cutremur = False
    are_loc_incendiu = False
    are_loc_alarma = False
    if cutremur <= prob_cutremur:
        are_loc_cutremur = True
    if incendiu <= prob_incendiu or incendiu <= 0.03 and are_loc_cutremur:
        are_loc_incendiu = True
    if alarma <= alarma_accidental or alarma <= 0.02 and are_loc_cutremur or alarma <= alarma_incendiu and are_loc_incendiu or alarma < 0.98 and are_loc_cutremur and are_loc_incendiu:
        are_loc_alarma = True

    """
    
    Problema cu PyMC3 
    
    """

    model = pm.Model()

    with model:
        Cutremur = pm.Bernoulli("C", 0.0005)
        Incendiu = pm.Bernoulli("I", pm.Deterministic("PI", pm.math.switch(Cutremur, 0.03, 0.01)))
        AlarmaIncendiu = pm.Deterministic('AI', pm.math.switch(Incendiu, pm.math.switch(Cutremur, 0.98, 0.02),
                                                               pm.math.switch(Incendiu, 0.95, 0.0001)))
        Alarma = pm.Bernoulli("A", p=AlarmaIncendiu)
        trace = pm.sample(20000, chains=1)

        # declansare_accidental=pm.Deterministic('D_a', pm.math.switch(cutremur,0.02,0.0001))
        # declansare=pm.Deterministic('D', pm.math.switch(incendiu,pm.math.switch(cutremur,0.98,0.95)))

    dictionary = {
        'Alarma': trace['A'].tolist(),
        'Cutremur': trace['C'].tolist(),
        'Incendiu': trace['I'].tolist()
    }
    df = pd.DataFrame(dictionary)
    # Ex.2
    p_cutremur = df[(df['cutremur'] == 1)].shape[0] & df[(df['fireAlarm'] == 1)].shape[0] / \
                     df[(df['fireAlarm'] == 1)].shape[0]
    # Ex.3
    p_incendiu = df[(df['incendiu'] == 1)].shape[0] & df[(df['fireAlarm'] == 0)].shape[0] / \
                     df[(df['incendiu'] == 1)].shape[0]


