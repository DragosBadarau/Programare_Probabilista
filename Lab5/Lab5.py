from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD


def simplified_poker():
    # Defining network structure
    joc_carti = BayesianNetwork(
        [
            ("Card1", "Card2"),
            ("Card1", "Round1"),
            ("Card2", "Round2"),
            ("Round1", "Round2"),
            ("Round1", "Round3"),
            ("Round2", "Round3"),
            ("Card1", "Round3")
        ]
    )
    # Defining the parameters using CPT
    CPD_C1 = TabularCPD(variable='Card1', variable_card=5, values=[[0.2], [0.2], [0.2], [0.2], [0.2]])
    print(CPD_C1)
    CPD_C2 = TabularCPD(variable='Card2', variable_card=5,
                        values=[[0, 0.25, 0.25, 0.25, 0.25], [0.25, 0, 0.25, 0.25, 0.25], [0.25, 0.25, 0, 0.25, 0.25],
                                [0.25, 0.25, 0.25, 0, 0.25], [0.25, 0.25, 0.25, 0.25, 0]], evidence=['Card1'],
                        evidence_card=[5])
    print(CPD_C2)
    CPD_R1 = TabularCPD(variable='Round1', variable_card=2, values=[[1, 0.75, 0.5, 0.25, 0], [0, 0.25, 0.5, 0.75, 1]],
                        evidence=['Card1'], evidence_card=[
            5])  # Probability to stay and to play, from by 0.25 because there are 5 diff values
    print(CPD_R1)
    CPD_R2 = TabularCPD(variable='Round2', variable_card=3,
                        values=[[0, 0.2, 0.4, 0.6, 0.9, 0, 0, 0, 0, 0],
                                [1, 0.75, 0.55, 0.35, 0.05, 1, 0.75, 0.4, 0.1, 0],
                                [0, 0.05, 0.05, 0.05, 0.05, 0, 0.25, 0.6, 0.9, 1]],
                        evidence=['Card2', 'Round1'], evidence_card=[5,
                                                                     3])  # 3 liste de posibilitati pentru jucatorul 2 : asteapta, pariaza, iese;
    # primele 5 valori din 10 sunt pentru runda 1 asteapta, urmatoarele 5 sunt pentru runda 1 pariaza
    print(CPD_R2)
    CPD_R3 = TabularCPD(variable='Round3', variable_card=2, values=[[1, 0.7, 0.5, 0.2, 0], [0, 0.3, 0.5, 0.8, 1]],
                        evidence=['Card1', 'Round2', 'Round1'],
                        evidence_card=[5, 3])
    # jucatorul 2 pariaza iar jucatorul 1 a asteptat =>
    # prima lista : jucatorul 1 pariaza
    # a2 a lista : jucatorul 1 iese

    print(CPD_R3)

    joc_carti.add_cpds(CPD_C1, CPD_C2, CPD_R1, CPD_R2, CPD_R3)
    joc_carti.check_model()


print(simplified_poker())
