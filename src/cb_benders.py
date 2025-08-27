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
    #ms = SolverFactory(f"appsi_{solver}")
    ms = appsi.solvers.Gurobi()
    ms.config.stream_solver = True
    ms.set_instance(master)
    ms.gurobi_options['PreCrush'] = 1
    ms.gurobi_options['LazyConstraints'] = 1
    ms.gurobi_options['TimeLimit'] = max_time

    # Define callback function that will be called by the solver for each incumbent
    def benders_callback(mod, sol, where):
        if where != GRB.Callback.MIPSOL:
            return
        # Initialize the subproblem solver
        ss = SolverFactory(solver, solver_io="nl")
        # ss = SolverFactory(f"appsi_{solver}")
        sol.cbGetSolution(vars=list(mod.component_data_objects(Var)))

        facility_open = {i: value(mod.facility_open[i]) for i in mod.FACILITIES}
        sub_variable_cost = {s: value(mod.sub_variable_cost[s]) for s in mod.SCENARIOS}

        for s in mod.SCENARIOS:
            # Build subproblem for scenario s with fixed facility_open from master solution
            sub = build_subproblem_for_scenario(data, s, facility_open)
            # Set options and solve
            ss_options = {
                "tech:outlev": int(verbose),
                "pre:solve": 0,
                "cvt:pre:all": 0,
                "alg:method": 1,
                "alg:rays": 2,
            }  # Turn off presolve in both the solver and MP driver use dual simplex as the solve method and turn on verbose output
            sub_result = ss.solve(sub, tee=verbose, options=ss_options)
            # Get solve results
            termination = sub_result.solver.termination_condition
            termination_name = str(termination)
            if termination == TerminationCondition.infeasible:
                con = mod.BendersCuts.add(
                    0
                    <= sum(  # Greater than 0 i.e. <= here b/c we haven't transformed our customer demand rule to <= constraints so we have to flip the sign
                        (
                            value(sub.dunbdd[sub.satisfying_customer_demand[i]])
                            if sub.satisfying_customer_demand[i] in sub.dunbdd
                            else 0
                        )
                        * value(sub.customer_demand[i])
                        for i in mod.CUSTOMERS
                    )
                    + sum(
                        (
                            value(sub.dunbdd[sub.facility_capacity_limits[i]])
                            if sub.facility_capacity_limits[i] in sub.dunbdd
                            else 0
                        )
                        * value(sub.facility_capacity[i])
                        * mod.facility_open[i]
                        for i in mod.FACILITIES
                    )
                )
                sol.cbLazy(con)
            elif termination == TerminationCondition.optimal:
                operating_cost = value(sub.objective)  # subproblem optimal value
                if operating_cost > sub_variable_cost[s] + tol:
                    con = mod.BendersCuts.add(
                        mod.sub_variable_cost[s]
                        >= sum(
                            value(sub.dual[sub.satisfying_customer_demand[i]])
                            * value(sub.customer_demand[i])
                            for i in mod.CUSTOMERS
                        )
                        + sum(
                            value(sub.dual[sub.facility_capacity_limits[i]])
                            * value(sub.facility_capacity[i])
                            * mod.facility_open[i]
                            for i in mod.FACILITIES
                        )
                    )
                    sol.cbLazy(con)
            else:
                raise Exception(
                    f"Solution for scenario {s} is neither optimal nor infeasible: {termination_name}"
                )

    ms.set_callback(benders_callback)
    ms.solve(master)
    return master
