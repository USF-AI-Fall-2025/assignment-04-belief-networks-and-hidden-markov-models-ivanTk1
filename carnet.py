from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.inference import VariableElimination
from pgmpy.factors.discrete import TabularCPD

car_model = DiscreteBayesianNetwork(
    [
        ("Battery", "Radio"),
        ("Battery", "Ignition"),
        ("Ignition","Starts"),
        ("Gas","Starts"),
        ("KeyPresent", "Starts"),
        ("Starts","Moves"),
])

# Defining the parameters using CPT


cpd_battery = TabularCPD(
    variable="Battery", variable_card=2, values=[[0.70], [0.30]],
    state_names={"Battery":['Works',"Doesn't work"]},
)

cpd_gas = TabularCPD(
    variable="Gas", variable_card=2, values=[[0.40], [0.60]],
    state_names={"Gas":['Full',"Empty"]},
)

cpd_key = TabularCPD(
    variable="KeyPresent",
    variable_card=2,
    values=[[0.7], [0.3]],
    state_names={"KeyPresent": ['yes', 'no']},
)


cpd_radio = TabularCPD(
    variable=  "Radio", variable_card=2,
    values=[[0.75, 0.01],[0.25, 0.99]],
    evidence=["Battery"],
    evidence_card=[2],
    state_names={"Radio": ["turns on", "Doesn't turn on"],
                 "Battery": ['Works',"Doesn't work"]}
)

cpd_ignition = TabularCPD(
    variable=  "Ignition", variable_card=2,
    values=[[0.75, 0.01],[0.25, 0.99]],
    evidence=["Battery"],
    evidence_card=[2],
    state_names={"Ignition": ["Works", "Doesn't work"],
                 "Battery": ['Works',"Doesn't work"]}
)

cpd_starts = TabularCPD(
    variable="Starts",
    variable_card=2,
    values=[
        [0.99, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01],  # P(Starts = yes)
        [0.01, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99],  # P(Starts = no)
    ],
    evidence=["Ignition", "Gas", "KeyPresent"],
    evidence_card=[2, 2, 2],
    state_names={
        "Starts": ['yes', 'no'],
        "Ignition": ["Works", "Doesn't work"],
        "Gas": ['Full', "Empty"],
        "KeyPresent": ['yes', 'no'],
    },
)


cpd_moves = TabularCPD(
    variable="Moves", variable_card=2,
    values=[[0.8, 0.01],[0.2, 0.99]],
    evidence=["Starts"],
    evidence_card=[2],
    state_names={"Moves": ["yes", "no"],
                 "Starts": ['yes', 'no'] }
)



# Associating the parameters with the model structure
car_model.add_cpds( cpd_battery, cpd_gas, cpd_key, cpd_radio, cpd_ignition, cpd_starts, cpd_moves
)

car_infer = VariableElimination(car_model)

print(car_infer.query(variables=["Moves"],evidence={"Radio":"turns on", "Starts":"yes"}))

def run_queries():
    print("\n")

    #1
    print("Given that the car will not move, what is the probability that the battery is not working?")
    print(car_infer.query(variables=["Battery"], evidence={"Moves": "no"}))
    print("\n")

    #2
    print("Given that the radio is not working, what is the probability that the car will not start?")
    print(car_infer.query(variables=["Starts"], evidence={"Radio": "Doesn't turn on"}))
    print("\n")

    #3a
    print("if the battery is working:")
    print("Probability that the radio works:")
    print(car_infer.query(variables=["Radio"], evidence={"Battery": "Works"}))
    print("\n")

    #3b
    print("does the probability of the radio working change if we discover that the car has gas in it?     ")
    print(car_infer.query(variables=["Radio"], evidence={"Battery": "Works", "Gas": "Full"}))
    print("\n")

    #4a
    print("when the car does NOT move:")
    print("Probability that the ignition works")
    print(car_infer.query(variables=["Ignition"], evidence={"Moves": "no"}))
    print("\n")

    #4b
    print("When the car does NOT move AND the gas tank is empty:")
    print("Probability that the ignition works")
    print(car_infer.query(variables=["Ignition"], evidence={"Moves": "no", "Gas": "Empty"}))
    print("\n")

    #5
    print("What is the probability that the car starts if the radio works and it has gas in it?")
    print(car_infer.query(variables=["Starts"], evidence={"Radio": "turns on", "Gas": "Full"}))


if __name__ == "__main__":
    run_queries()

