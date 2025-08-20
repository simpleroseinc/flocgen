# Master problem for Benders decomposition of two-stage stochastic facility location
# Benders feasibility cuts and optimality cuts are added iteratively
from pyomo.environ import *
from utils import *


def build_master(data, capacity_rule: CapacityRule) -> ConcreteModel:
    """
    Take a data dictionary and build the Benders master problem for the two-stage stochastic facility location problem.
    If use_capacity_sufficiency is True, adds a feasibility "guard" constraint.
    Returns a Pyomo model ready to be solved.
    """
    model = ConcreteModel(name="FacilityLocation-BendersMaster")

    # Indexing sets
    model.FACILITIES = Set(initialize=data["FACILITIES"])
    model.CUSTOMERS = Set(initialize=data["CUSTOMERS"])
    model.SCENARIOS = Set(initialize=data["SCENARIOS"])

    # Master-only Parameters
    model.fixed_cost = Param(model.FACILITIES, initialize=data["fixed_cost"])
    model.prob = Param(model.SCENARIOS, initialize=data["prob"])
    # Master and Subproblem Parameters
    model.facility_capacity = Param(
        model.FACILITIES, initialize=data["facility_capacity"]
    )
    model.customer_demand = Param(
        model.CUSTOMERS,
        model.SCENARIOS,
        initialize=data["customer_demand"].stack().to_dict(),
        within=NonNegativeReals,
    )

    # Master Problem Variables
    # First Stage Decisions
    model.facility_open = Var(model.FACILITIES, within=Binary)
    # Subproblem Objective Bound needed for Benders decomposition
    model.sub_variable_cost = Var(model.SCENARIOS, within=NonNegativeReals)

    # Objective
    # Total Cost Objective: fixed + variable costs
    def total_cost_rule(m):
        fixed = sum(m.fixed_cost[i] * m.facility_open[i] for i in m.FACILITIES)
        variable = sum(m.prob[s] * m.sub_variable_cost[s] for s in m.SCENARIOS)
        return fixed + variable

    model.objective = Objective(rule=total_cost_rule, sense=minimize)

    # Constraints
    # Sufficient production capacity constraint
    def sufficient_production_capacity_rule(m):
        # Determine the capacity threshold for sufficient production capacity (max means demand will be satisfied under all circumstances)
        capacity_threshold = calculate_capacity_threshold(m, capacity_rule)
        return (
            sum(m.facility_capacity[i] * m.facility_open[i] for i in m.FACILITIES)
            >= capacity_threshold
        )

    model.sufficient_production_capacity = Constraint(
        rule=sufficient_production_capacity_rule
    )

    # Create container for holding Benders cuts during the algorithm
    model.BendersCuts = ConstraintList()

    return model
