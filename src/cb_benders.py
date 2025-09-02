# cb_benders.py
from gurobipy import GRB
from pyomo.environ import Var
from pyomo.contrib import appsi
from master import *
from sub import *


def cb_benders_solve(
    data,
    capacity_rule=CapacityRule.MAX,
    max_time=100,
    relax=False,
    solver="gurobi",
    threads=1,
    tol=1e-6,
    verbose=True,
) -> ConcreteModel:
    """
    Multi-cut Benders for the two-stage Stochastic Facility Location problem.
    Returns master_model.
    """
    num_incumbent = 0  # Global counter for number of incumbents

    # Define callback function that will be called by the solver for each incumbent
    def benders_callback(mod, sol, where):
        # If no incumbent solution, do nothing
        if where != GRB.Callback.MIPSOL:
            return

        # Increment counter for every incumbent solution
        nonlocal num_incumbent
        num_incumbent += 1
        # Get master problem statistics
        master_cons = mod.nconstraints()
        master_vars = mod.nvariables()
        # Get master incumbent solution (facility_open and sub_variable_cost)
        sol.cbGetSolution(
            vars=list(mod.component_data_objects(Var, active=True, descend_into=True))
        )
        facility_open = {i: value(mod.facility_open[i]) for i in mod.FACILITIES}
        sub_variable_cost = {s: value(mod.sub_variable_cost[s]) for s in mod.SCENARIOS}

        # Initialize the subproblem solver
        ss = get_solver(solver)
        if not hasattr(ss, "get_linear_constraint_attr"):
            raise RuntimeError(
                f"Solver '{solver}' does not support dual or ray extraction needed for Benders decomposition."
            )

        # Helpers to pull dual and dual ray information
        # (The names 'Pi' and 'FarkasDual' are specific to Gurobi)
        def dual_on(con):  # valid if optimal
            try:
                return float(ss.get_linear_constraint_attr(con, "Pi"))
            except AttributeError:
                return 0.0

        def ray_on(con):  # valid if infeasible
            try:
                return float(ss.get_linear_constraint_attr(con, "FarkasDual"))
            except AttributeError:
                return 0.0

        # Reset expected operating cost for this incumbent
        expected_operating_cost = 0.0
        for s in mod.SCENARIOS:
            # Build subproblem for scenario s with fixed facility_open from master solution
            sub = build_subproblem_for_scenario(data, s, facility_open)
            # Get subproblem statistics
            sub_cons = sub.nconstraints()
            sub_vars = sub.nvariables()
            # Set options and solve
            options = None
            if solver == "gurobi":
                options = {
                    "InfUnbdInfo": 1,  # To get unbounded ray information for the dual of the primal problem
                    "Method": 1,  # Use dual simplex so we can get Farkas Rays (i.e. direction of unboundedness for the dual) for infeasible primal subproblems
                }
            sub_result = solve_model(
                sub, ss, options=options, solver_threads=threads, verbose=verbose
            )
            # Get solve results
            termination = sub_result.solver.termination_condition
            termination_name = str(termination)

            if termination == TerminationCondition.infeasible:
                lhs = sum(
                    ray_on(sub.satisfying_customer_demand[i])
                    * value(sub.customer_demand[i])
                    for i in sub.CUSTOMERS
                ) + sum(
                    ray_on(sub.facility_capacity_limits[i])
                    * value(sub.facility_capacity[i])
                    * mod.facility_open[i]
                    for i in sub.FACILITIES
                )
                con = mod.BendersCuts.add(lhs >= 0 + tol)
                sol.cbLazy(con)
            elif termination == TerminationCondition.optimal:
                operating_cost = value(sub.objective)  # subproblem optimal value
                expected_operating_cost += value(mod.prob[s]) * operating_cost
                if operating_cost > sub_variable_cost[s] + tol:
                    lhs = sum(
                        dual_on(sub.satisfying_customer_demand[i])
                        * value(sub.customer_demand[i])
                        for i in sub.CUSTOMERS
                    ) + sum(
                        dual_on(sub.facility_capacity_limits[i])
                        * value(sub.facility_capacity[i])
                        * mod.facility_open[i]
                        for i in sub.FACILITIES
                    )
                    con = mod.BendersCuts.add(lhs <= mod.sub_variable_cost[s] - tol)
                    sol.cbLazy(con)
            else:
                raise Exception(
                    f"Solution for scenario {s} is neither optimal nor infeasible: {termination_name}"
                )

        # Update upper bound
        fixed = sum(value(mod.fixed_cost[i]) * facility_open[i] for i in mod.FACILITIES)
        upper_bound = fixed + expected_operating_cost
        print(
            f"{iteration} Statistics:\n\tMaster: {master_cons} cons, {master_vars} vars\n\tSubproblem: {sub_cons} cons, {sub_vars} vars, {len(list(mod.SCENARIOS))} scenarios\n\tViolations: {feas_violation} feas, {opt_violation} opt\n\tBounds: {lower_bound:.2f} <= {upper_bound:.2f}",
            flush=True,
        )

    # Check arguments
    tol = abs(tol)
    if max_time < 1:
        raise ValueError("max_time must be at least 1.")

    # Build master problem
    master = build_master(data=data, capacity_rule=capacity_rule)
    # If required by user relax integrality
    if relax:
        TransformationFactory("core.relax_integer_vars").apply_to(master)
    # Set up master solver
    ms = get_solver(solver, callback=True)
    if not hasattr(ms, "set_callback"):
        raise RuntimeError(f"solver '{solver}' does not accept callbacks.")
    if solver == "gurobi":
        options = {
            "PreCrush": 1,  # Required so user added constraints can be applied to presolved model
            "LazyConstraints": 1,  # Enable lazy constraints so we can add Benders cuts
            "TimeLimit": max_time,
        }
    _ = solve_model(
        master,
        ms,
        callback=benders_callback,
        options=options,
        solver_threads=threads,
        verbose=verbose,
    )
    return master
