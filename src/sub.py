# Subproblem for a given scenario
# Given a fixed first-stage solution x, solve the dual LP for scenario s.
# This is used to generate Benders cuts.
from pyomo.environ import *


def build_subproblem_for_scenario(data, scenario, facility_open) -> ConcreteModel:
    """
    Build the LP for scenario s, given a fixed sub_facility_open (dict-like {i:0/1}).
    Returns a Pyomo model ready to be solved.
    """
    model = ConcreteModel(name=f"FacilityLocation-BendersSubProblem-{scenario}")
    # Create a 'dual' and 'dunbdd` suffix components on the model so the solver plugin will know which suffixes to collect
    model.dual = Suffix(direction=Suffix.IMPORT, datatype=Suffix.FLOAT)
    model.dunbdd = Suffix(direction=Suffix.IMPORT_EXPORT, datatype=Suffix.FLOAT)

    # Indexing sets
    model.FACILITIES = Set(initialize=data["FACILITIES"])
    model.CUSTOMERS = Set(initialize=data["CUSTOMERS"])

    # Parameters
    # Master and Suproblem Parameters
    model.facility_capacity = Param(
        model.FACILITIES, initialize=data["facility_capacity"], within=NonNegativeReals
    )
    model.customer_demand = Param(
        model.CUSTOMERS,
        initialize={
            j: data["customer_demand"].loc[j, scenario] for j in data["CUSTOMERS"]
        },
        within=NonNegativeReals,
    )
    # Suproblem-only parameters
    model.variable_cost = Param(
        model.FACILITIES,
        model.CUSTOMERS,
        initialize=data["variable_cost"]["distance"].to_dict(),
        within=NonNegativeReals,
    )
    # Benders' parameters
    model.facility_open = Param(
        model.FACILITIES,
        initialize={i: float(facility_open[i]) for i in data["FACILITIES"]},
    )
    model.production_coeff = Param(
        model.FACILITIES,
        model.CUSTOMERS,
        initialize=data["production_coeff"]["production_coeff"]
        .xs(scenario, level="SCENARIOS")
        .to_dict(),
        within=NonNegativeReals,
    )

    # Variables
    model.production = Var(model.FACILITIES, model.CUSTOMERS, within=NonNegativeReals)

    # Subproblem objective
    def operating_cost_rule(m):
        return sum(
            m.variable_cost[i, j] * m.production[i, j]
            for i in m.FACILITIES
            for j in m.CUSTOMERS
        )

    model.objective = Objective(rule=operating_cost_rule, sense=minimize)

    # Constraints
    def satisfying_customer_demand_rule(m, j):
        return (
            sum(m.production_coeff[i, j] * m.production[i, j] for i in m.FACILITIES)
            >= m.customer_demand[j]
        )

    model.satisfying_customer_demand = Constraint(
        model.CUSTOMERS, rule=satisfying_customer_demand_rule
    )

    def facility_capacity_limits_rule(m, i):
        return (
            sum(m.production_coeff[i, j] * m.production[i, j] for j in m.CUSTOMERS)
            <= m.facility_capacity[i] * m.facility_open[i]
        )

    model.facility_capacity_limits = Constraint(
        model.FACILITIES, rule=facility_capacity_limits_rule
    )

    return model
