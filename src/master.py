# Master problem for Benders decomposition of two-stage stochastic facility location
# Extensive-form has first-stage open/close decisions and second-stage recourse costs
# This master problem has first-stage decisions and Benders (recourse) variables theta_s
# Benders cuts are added iteratively during the algorithm
from pyomo.environ import *


def build_master(data, use_capacity_sufficiency=True):
    """
    Build the Benders master problem.
    If use_capacity_sufficiency is True, adds a feasibility "guard" constraint.
    Returns a Pyomo model ready to be solved.
    """
    model = ConcreteModel(name="FacilityLocation-BendersMaster")

    model.FACILITIES = Set(initialize=data["FACILITIES"])
    model.CUSTOMERS = Set(initialize=data["CUSTOMERS"])
    model.SCENARIOS = Set(initialize=data["SCENARIOS"])

    # Parameters
    model.fixed_cost = Param(model.FACILITIES, initialize=data["fixed_cost"])
    # Expected value weights
    model.prob = Param(model.SCENARIOS, initialize=data["prob"])
    # For the optional feasibility guard:
    model.facility_capacity = Param(
        model.FACILITIES, initialize=data["facility_capacity"]
    )
    model.customer_demand = Param(
        model.CUSTOMERS,
        model.SCENARIOS,
        initialize=data["customer_demand"].stack().to_dict(),
    )
    # First-stage: open/close
    model.x = Var(model.FACILITIES, within=Binary)
    # Benders (recourse) variables (multi-cut version)
    model.theta = Var(model.SCENARIOS, within=NonNegativeReals)

    # Objective: fixed + expected recourse
    def obj_rule(m):
        fixed = sum(m.fixed_cost[i] * m.x[i] for i in m.FACILITIES)
        rec = sum(m.prob[s] * m.theta[s] for s in m.SCENARIOS)
        return fixed + rec

    model.Obj = Objective(rule=obj_rule, sense=minimize)

    # Optional feasibility “guard”: total capacity of open facs ≥ worst-case total demand
    if use_capacity_sufficiency:
        worst_total = max(
            sum(value(model.customer_demand[j, s]) for j in model.CUSTOMERS)
            for s in model.SCENARIOS
        )

        def cap_guard(m):
            return (
                sum(m.facility_capacity[i] * m.x[i] for i in m.FACILITIES)
                >= worst_total
            )

        model.CapacityGuard = Constraint(rule=cap_guard)

    # Benders cuts added here during the algorithm
    model.BendersCuts = ConstraintList()

    return model
