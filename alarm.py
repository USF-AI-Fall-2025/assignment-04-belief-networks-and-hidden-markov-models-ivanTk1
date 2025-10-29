from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.inference import VariableElimination

def build_alarm_model():
    alarm_model = DiscreteBayesianNetwork(
        [
            ("Burglary", "Alarm"),
            ("Earthquake", "Alarm"),
            ("Alarm", "JohnCalls"),
            ("Alarm", "MaryCalls"),
        ]
    )

    # Defining the parameters using CPT
    from pgmpy.factors.discrete import TabularCPD

    cpd_burglary = TabularCPD(
        variable="Burglary", variable_card=2, values=[[0.999], [0.001]],
        state_names={"Burglary":['no','yes']},
    )
    cpd_earthquake = TabularCPD(
        variable="Earthquake", variable_card=2, values=[[0.998], [0.002]],
        state_names={"Earthquake":["no","yes"]},
    )
    cpd_alarm = TabularCPD(
        variable="Alarm",
        variable_card=2,
        values=[[0.999, 0.71, 0.06, 0.05], [0.001, 0.29, 0.94, 0.95]],
        evidence=["Burglary", "Earthquake"],
        evidence_card=[2, 2],
        state_names={"Burglary":['no','yes'], "Earthquake":['no','yes'], 'Alarm':['no','yes']},
    )
    cpd_johncalls = TabularCPD(
        variable="JohnCalls",
        variable_card=2,
        values=[[0.95, 0.1], [0.05, 0.9]],
        evidence=["Alarm"],
        evidence_card=[2],
        state_names={"Alarm":['no','yes'], "JohnCalls":['no', 'yes']},
    )
    cpd_marycalls = TabularCPD(
        variable="MaryCalls",
        variable_card=2,
        values=[[0.99, 0.3], [0.01, 0.7]],
        evidence=["Alarm"],
        evidence_card=[2],
        state_names={'Alarm':['no','yes'], 'MaryCalls':['no', 'yes']},
    )

    # Associating the parameters with the model structure
    alarm_model.add_cpds(
        cpd_burglary, cpd_earthquake, cpd_alarm, cpd_johncalls, cpd_marycalls)
    return alarm_model

def main():
    model = build_alarm_model()
    alarm_infer = VariableElimination(model)

    q = alarm_infer.query(variables=["JohnCalls"], evidence={"Earthquake": "yes"})
    print(q)
    print("\n")

    print("P(MaryCalls | JohnCalls = yes)")
    q1 = alarm_infer.query(variables=["MaryCalls"], evidence={"JohnCalls": "yes"})
    print(q1)
    print("\n")

    print("P(JohnCalls, MaryCalls | Alarm = yes)")
    q2 = alarm_infer.query(variables=["JohnCalls", "MaryCalls"], evidence={"Alarm": "yes"})
    print(q2)
    print("\n")

    print("P(Alarm | MaryCalls = yes")
    q3 = alarm_infer.query(variables=["Alarm"], evidence={"MaryCalls": "yes"})
    print(q3)
    print("\n")


if __name__ == "__main__":
    main()
