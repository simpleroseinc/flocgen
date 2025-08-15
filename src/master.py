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
    m = ConcreteModel(name="BendersMaster")

    # Sets (same naming as your extensive-form)
    m.F = Set(initialize=data["FACILITIES"])
    m.C = Set(initialize=data["CUSTOMERS"])
    m.S = Set(initialize=data["SCENARIOS"])

    # Parameters
    m.fixed_cost = Param(m.F, initialize=data["fixed_cost"])
    # Expected value weights
    m.prob = Param(m.S, initialize=data["prob"])
    # For the optional feasibility guard:
    m.facility_capacity = Param(m.F, initialize=data["facility_capacity"])
    m.customer_demand = Param(
        m.C, m.S, initialize=data["customer_demand"].stack().to_dict()
    )
    # First-stage: open/close
    m.x = Var(m.F, within=Binary)
    # Benders (recourse) variables (multi-cut version)
    m.theta = Var(m.S, within=NonNegativeReals)

    # Objective: fixed + expected recourse
    def obj_rule(m):
        fixed = sum(m.fixed_cost[i] * m.x[i] for i in m.F)
        rec = sum(m.prob[s] * m.theta[s] for s in m.S)
        return fixed + rec

    m.Obj = Objective(rule=obj_rule, sense=minimize)

    # Optional feasibility “guard”: total capacity of open facs ≥ worst-case total demand
    if use_capacity_sufficiency:
        worst_total = max(sum(value(m.customer_demand[j, s]) for j in m.C) for s in m.S)

        def cap_guard(m):
            return sum(m.facility_capacity[i] * m.x[i] for i in m.F) >= worst_total

        m.CapacityGuard = Constraint(rule=cap_guard)

    # Benders cuts added here during the algorithm
    m.BendersCuts = ConstraintList()

    return m
