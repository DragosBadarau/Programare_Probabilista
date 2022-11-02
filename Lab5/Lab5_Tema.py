from pgmpy.models import BayesianNetwork
from pgmpy.inference import VariableElimination
from pgmpy.factors.discrete import TabularCPD

if __name__ == '__main__':
    # Defining network structure
    joc_carti = BayesianNetwork(
        [
            ("Card1", "Card2"),
            ("Card1", "Round1"),
            ("Card2", "Round2"),
            ("Round1", "Round2"),
            ("Round2", "Round3"),
            ("Card1", "Round3")
        ]
    )
    # Defining the parameters using CPT
    CPD_C1 = TabularCPD(variable='Card1', variable_card=5, values=[[0.2], [0.2], [0.2], [0.2], [0.2]], state_names={
        'Card1': ['AsF', 'RegeF', 'RegeI', 'ReginaF', 'ReginaI']
    })
    CPD_C2 = TabularCPD(variable='Card2', variable_card=5,
                        values=[[0, 0.25, 0.25, 0.25, 0.25], [0.25, 0, 0.25, 0.25, 0.25], [0.25, 0.25, 0, 0.25, 0.25],
                                [0.25, 0.25, 0.25, 0, 0.25], [0.25, 0.25, 0.25, 0.25, 0]], evidence=['Card1'],
                        evidence_card=[5], state_names={
            'Card2': ['AsF', 'RegeF', 'RegeI', 'ReginaF', 'ReginaI'],
            'Card1': ['AsF', 'RegeF', 'RegeI', 'ReginaF', 'ReginaI']
        })
    CPD_R1 = TabularCPD(variable='Round1', variable_card=2, values=[[1, 0.75, 0.5, 0.25, 0],  # pariaza
                                                                    [0, 0.25, 0.5, 0.75, 1]],  # asteapta
                        evidence=['Card1'], evidence_card=[5], state_names={
            'D1': ['Pariaza', 'Asteapta'],
            'CJ1': ['AsF', 'RegeF', 'RegeI', 'ReginaF', 'ReginaI']
        })  # Probability to stay and to play, from by 0.25 because there are 5 diff values
    CPD_R2 = TabularCPD(variable='Round2', variable_card=3,
                        values=[[0, 0.2, 0.4, 0.6, 0.9, 0, 0, 0, 0, 0],  # asteapta
                                [1, 0.75, 0.55, 0.35, 0.05, 1, 0.75, 0.4, 0.1, 0],  # pariaza
                                [0, 0.05, 0.05, 0.05, 0.05, 0, 0.25, 0.6, 0.9, 1]],  # iese
                        evidence=['Card2', 'Round1'], evidence_card=[5,
                                                                     2], state_names={
            'D2': ['Asteapta', 'Pariaza', 'Afara'],
            'CJ2': ['AsF', 'RegeF', 'RegeI', 'ReginaF', 'ReginaI'],
            'D1': ['Asteapta', 'Pariaza']
        })  # 3 liste de posibilitati pentru jucatorul 2 : asteapta, pariaza, iese;
    # primele 5 valori din 10 sunt pentru runda 1 asteapta, urmatoarele 5 sunt pentru runda 1 pariaza
    CPD_R3 = TabularCPD(variable='Round3', variable_card=2,
                        values=[[1, 1, 1, 0.75, 0.75, 0.75, 0.5, 0.5, 0.5, 0.25, 0.25, 0.25, 0.0, 0.0, 0.0],  # pariaza
                                [0.0, 0.0, 0.0, 0.25, 0.25, 0.25, 0.5, 0.5, 0.5, 0.75, 0.75, 0.75, 1, 1, 1]],  # iese,
                        evidence=['Card1', 'Round2'],
                        evidence_card=[5, 3], state_names={
            'D3': ['Pariaza', 'Afara'],
            'CJ1': ['AsF', 'RegeF', 'RegeI', 'ReginaF', 'ReginaI'],
            'D2': ['Pariaza', 'Asteapta', 'Afara']
        })
    # jucatorul 2 pariaza iar jucatorul 1 a asteptat =>
    # prima lista : jucatorul 1 pariaza
    # a2 a lista : jucatorul 1 iese
    joc_carti.add_cpds(CPD_C1, CPD_C2, CPD_R1, CPD_R2, CPD_R3)
    # TEMA
    rez = VariableElimination(joc_carti)
    print("Subpunctul (a):\n", rez.query(['Round1'], evidence={'Card1': 'RegeF'}))
    print("Subpunctul (b):\n", rez.query(['Round2'], evidence={'Card2': 'RegeI', 'Round1': 'Pariaza'}))
