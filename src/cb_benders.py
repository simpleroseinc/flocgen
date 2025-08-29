# cb_benders.py
from gurobipy import GRB
from pyomo.environ import Var
from pyomo.contrib import appsi
from master import *
from sub import *


def cb_benders_solve(
    data,
    capacity_rule=CapacityRule.MAX,
    relax=False,
    solver="gurobi",
    max_time=100,
    tol=1e-6,
    verbose=True,
) -> ConcreteModel:
    """
    Multi-cut Benders for the two-stage Stochastic Facility Location problem.
    Returns master_model.
    """
    master = build_master(data=data, capacity_rule=capacity_rule)
    # Relax integrality
    if relax:
        TransformationFactory("core.relax_integer_vars").apply_to(master)
    # Set up Master Solver
    if solver == "gurobi":
        ms = appsi.solvers.Gurobi()
    else:
        raise RuntimeError(f"solver '{solver}' not implemented.")
    ms.config.stream_solver = bool(verbose)  # Verbose output
    ms.set_instance(master)
    ms.gurobi_options['PreCrush'] = 1 # Required so user added constraints can be applied to presolved model
    ms.gurobi_options['LazyConstraints'] = 1 # Enable lazy constraints so we can add Benders cuts
    ms.gurobi_options['TimeLimit'] = max_time
    # Get number of physical cores and set threads to half that
    master_threads = max(1,get_physical_cores()//2)
    sub_threads = max(1,get_physical_cores()//2)
    #ms.gurobi_options['Threads'] =

    # Define callback function that will be called by the solver for each incumbent
    def benders_callback(mod, sol, where):
        if where != GRB.Callback.MIPSOL:
            return

        # Get Master incumbent solution (facility_open and sub_variable_cost)
        sol.cbGetSolution(vars=list(mod.component_data_objects(Var, active=True, descend_into=True)))
        facility_open = {i: value(mod.facility_open[i]) for i in mod.FACILITIES}
        sub_variable_cost = {s: value(mod.sub_variable_cost[s]) for s in mod.SCENARIOS}

        # Initialize the subproblem solver
        ss = SolverFactory(f"{solver}_persistent")

        # helpers to pull multipliers with your constraint names
        def dual_on(con):      # valid if optimal
            return float(ss.get_linear_constraint_attr(con, 'Pi'))
        def ray_on(con):       # valid if infeasible
            return float(ss.get_linear_constraint_attr(con, 'FarkasDual'))

        for s in mod.SCENARIOS:
            # Build subproblem for scenario s with fixed facility_open from master solution
            sub = build_subproblem_for_scenario(data, s, facility_open)
            # Set options and solve
            ss.set_instance(sub)
            ss.set_gurobi_param('OutputFlag', bool(verbose))  # Verbose output
            ss.set_gurobi_param('InfUnbdInfo', 1)  # To get unbounded ray information for the dual
            ss.set_gurobi_param('Threads', 1)  # Allows us to farm out each subproblem to a different core if desired
            ss.set_gurobi_param('Method', 1)  # Use dual simplex so we can get Farkas Rays (i.e. direction of unboundedness for the dual) for infeasible primal subproblems
            sub_result = ss.solve(sub)
            # Get solve results
            termination = sub_result.solver.termination_condition
            termination_name = str(termination)

            if termination == TerminationCondition.infeasible:
                lhs = (
                        sum(ray_on(sub.satisfying_customer_demand[i]) * value(sub.customer_demand[i])
                            for i in sub.CUSTOMERS)
                        +
                        sum(ray_on(sub.facility_capacity_limits[i]) * value(sub.facility_capacity[i]) * mod.facility_open[i]
                            for i in sub.FACILITIES)
                )
                con = mod.BendersCuts.add(lhs >= 0)
                sol.cbLazy(con)
            elif termination == TerminationCondition.optimal:
                operating_cost = value(sub.objective)  # subproblem optimal value
                if operating_cost > sub_variable_cost[s] + tol:
                    lhs = (
                            sum(dual_on(sub.satisfying_customer_demand[i]) * value(sub.customer_demand[i])
                                for i in sub.CUSTOMERS)
                            +
                            sum(dual_on(sub.facility_capacity_limits[i]) * value(sub.facility_capacity[i]) *
                                mod.facility_open[i]
                                for i in sub.FACILITIES)
                    )
                    con = mod.BendersCuts.add(lhs <= mod.sub_variable_cost[s])
                    sol.cbLazy(con)
            else:
                raise Exception(
                    f"Solution for scenario {s} is neither optimal nor infeasible: {termination_name}"
                )

    ms.set_callback(benders_callback)
    ms.solve(master)
    return master
