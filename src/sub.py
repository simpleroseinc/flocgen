# Subproblem (dual) for a given scenario
# Given a fixed first-stage solution x, solve the dual LP for scenario s.
# This is used to generate Benders cuts.
from pyomo.environ import *

def build_dual_subproblem_for_scenario(data, s, x_values):
    """
    Build the dual LP for scenario s, given a fixed x (dict-like {i:0/1}).
    Returns a Pyomo model ready to be solved.
    """
    m = ConcreteModel(name=f"Dual_s={s}")

    m.F = Set(initialize=data["FACILITIES"])
    m.C = Set(initialize=data["CUSTOMERS"])

    # Params
    m.cap = Param(m.F, initialize=data["facility_capacity"])
    m.demand = Param(m.C, initialize={j: data["customer_demand"].loc[j, s] for j in data["CUSTOMERS"]})
    # variable_cost is a DataFrame with column 'distance' → dict with (i,j) keys in your EF
    vc_dict = data["variable_cost"]["distance"].to_dict()
    m.c = Param(m.F, m.C, initialize=vc_dict)

    # production_coeff is DataFrame with column 'production_coeff' keyed by (i,j,s)
    a_dict = data["production_coeff"]["production_coeff"].to_dict()
    def a_init(m, i, j):
        return a_dict[(i, j, s)]
    m.a = Param(m.F, m.C, initialize=a_init)

    # Fixed first-stage x
    m.xbar = Param(m.F, initialize={i: float(x_values[i]) for i in data["FACILITIES"]})

    # Dual variables
    m.pi = Var(m.C, within=NonNegativeReals)     # for demand ≥ constraints
    m.mu = Var(m.F, within=NonPositiveReals)     # for capacity ≤ constraints

    # Dual constraints: a_{ij}(pi_j + mu_i) ≤ c_{ij}
    def dual_con_rule(m, i, j):
        return m.a[i, j]*(m.pi[j] + m.mu[i]) <= m.c[i, j]
    m.DualCon = Constraint(m.F, m.C, rule=dual_con_rule)

    # Dual objective
    def dual_obj(m):
        term_demand = sum(m.demand[j]*m.pi[j] for j in m.C)
        term_cap    = sum(m.cap[i]*m.xbar[i]*m.mu[i] for i in m.F)
        return term_demand + term_cap
    m.Obj = Objective(rule=dual_obj, sense=maximize)

    return m
