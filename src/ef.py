import pandas as pd
from typing import Dict, Union
from pyomo.environ import *

NumDict = Dict[str, Union[int, float]]
DataDict = Dict[str, Union[NumDict, pd.DataFrame]]


def floc_model(data: DataDict) -> ConcreteModel:
    # Define the model
    model = ConcreteModel()

    # Indexing sets
    model.FACILITIES = Set(initialize=data["FACILITIES"])
    model.CUSTOMERS = Set(initialize=data["CUSTOMERS"])
    model.SCENARIOS = Set(initialize=data["SCENARIOS"])

    # Variables
    model.facility_open = Var(model.FACILITIES, within=Binary)
    model.production = Var(
        model.FACILITIES, model.CUSTOMERS, model.SCENARIOS, within=NonNegativeReals
    )

    # Parameters
    model.fixed_cost = Param(
        model.FACILITIES, initialize=data["fixed_cost"], within=NonNegativeReals
    )
    model.variable_cost = Param(
        model.FACILITIES,
        model.CUSTOMERS,
        initialize=data["variable_cost"]["distance"].to_dict(),
        within=NonNegativeReals,
    )
    model.customer_demand = Param(
        model.CUSTOMERS,
        model.SCENARIOS,
        initialize=data["customer_demand"].stack().to_dict(),
        within=NonNegativeReals,
    )
    model.facility_capacity = Param(
        model.FACILITIES, initialize=data["facility_capacity"], within=NonNegativeReals
    )
    model.production_coeff = Param(
        model.FACILITIES,
        model.CUSTOMERS,
        model.SCENARIOS,
        initialize=data["production_coeff"]["production_coeff"].to_dict(),
        within=NonNegativeReals,
    )
    model.prob = Param(
        model.SCENARIOS, initialize=data["prob"], within=NonNegativeReals
    )

    # Objective
    def total_cost_rule(model):
        fixed_cost = sum(
            model.fixed_cost[i] * model.facility_open[i] for i in model.FACILITIES
        )
        variable_cost = sum(
            model.prob[s] * model.variable_cost[i, j] * model.production[i, j, s]
            for s in model.SCENARIOS
            for i in model.FACILITIES
            for j in model.CUSTOMERS
        )
        return fixed_cost + variable_cost

    model.TotalCost = Objective(rule=total_cost_rule, sense=minimize)

    # Constraints
    def satisfying_customer_demand_rule(model, s, j):
        return (
            sum(
                model.production_coeff[i, j, s] * model.production[i, j, s]
                for i in model.FACILITIES
            )
            >= model.customer_demand[j, s]
        )

    model.satisfying_customer_demand = Constraint(
        model.SCENARIOS, model.CUSTOMERS, rule=satisfying_customer_demand_rule
    )

    def facility_capacity_limits_rule(model, s, i):
        return (
            sum(
                model.production_coeff[i, j, s] * model.production[i, j, s]
                for j in model.CUSTOMERS
            )
            <= model.facility_capacity[i] * model.facility_open[i]
        )

    model.facility_capacity_limits = Constraint(
        model.SCENARIOS, model.FACILITIES, rule=facility_capacity_limits_rule
    )

    def sufficient_production_capacity_rule(model):
        return sum(
            model.facility_capacity[i] * model.facility_open[i]
            for i in model.FACILITIES
        ) >= max(
            sum(model.customer_demand[j, s] for j in model.CUSTOMERS)
            for s in model.SCENARIOS
        )

    model.sufficient_production_capacity = Constraint(
        rule=sufficient_production_capacity_rule
    )

    return model
