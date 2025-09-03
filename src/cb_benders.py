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

        # Get master problem statistics
        master_cons, master_vars = mod.nconstraints(), mod.nvariables()
        # Get master incumbent solution (facility_open and sub_variable_cost)
        sol.cbGetSolution(
            vars=list(mod.component_data_objects(Var, active=True, descend_into=True))
        )
        facility_open = {i: value(mod.facility_open[i]) for i in mod.FACILITIES}
        sub_variable_cost = {s: value(mod.sub_variable_cost[s]) for s in mod.SCENARIOS}

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

        # Per-incumbent stats
        nonlocal num_incumbent
        num_incumbent += 1
        feas_violation = opt_violation = 0
        sub_cons = sub_vars = 0
        expected_operating_cost = 0.0
        for s in mod.SCENARIOS:
            # Get subproblem and solver for scenario s
            sub = sub_problems[s]
            ss = sub_solvers[s]
            # Get subproblem statistics
            sub_cons, sub_vars = sub.nconstraints(), sub.nvariables()
            # Update subproblem with fixed facility_open from master solution
            set_sub_probelm_rhs(sub, ss, facility_open)
            # Solve subproblem
            sub_result = ss.solve(sub)
            # Get solve results
            termination = sub_result.solver.termination_condition
            termination_name = str(termination)

            if termination == TerminationCondition.infeasible:
                feas_violation += 1
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
                    opt_violation += 1
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
        lower_bound = get_objective_value(mod)
        print(
            f"{num_incumbent} Statistics:\n\tMaster: {master_cons} cons, {master_vars} vars\n\tSubproblem: {sub_cons} cons, {sub_vars} vars, {len(list(mod.SCENARIOS))} scenarios\n\tViolations: {feas_violation} feas, {opt_violation} opt\n\tBounds: {lower_bound:.2f} <= {upper_bound:.2f}",
            file=sys.__stdout__,
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
    master_options = None
    if solver == "gurobi":
        master_options = {
            "PreCrush": 1,  # Required so user added constraints can be applied to presolved model
            "LazyConstraints": 1,  # Enable lazy constraints so we can add Benders cuts
            "TimeLimit": max_time,
        }
    # Prebuild one sub-problem model and one persistent solver per scenario
    sub_problems = {}
    sub_solvers = {}

    # Set subproblem solver options
    sub_options = None
    if solver == "gurobi":
        sub_options = {
            "InfUnbdInfo": 1,  # To get unbounded ray information for the dual of the primal problem
            "Method": 1,  # Use dual simplex so we can get Farkas Rays (i.e. direction of unboundedness for the dual) for infeasible primal subproblems
        }
    for s in master.SCENARIOS:
        sub = build_subproblem_for_scenario(
            data, s, {i: 0.0 for i in data["FACILITIES"]}
        )
        sub_problems[s] = sub
        ss = get_solver(solver)
        if not hasattr(ss, "get_linear_constraint_attr"):
            raise RuntimeError(
                f"Solver '{solver}' does not support dual or ray extraction needed for Benders decomposition."
            )
        sub_solvers[s] = ss
        setup_benders_sub_solver(
            sub_problems[s],
            sub_solvers[s],
            options=sub_options,
            solver_threads=threads,
            verbose=verbose,
        )

    _ = solve_model(
        master,
        ms,
        callback=benders_callback,
        options=master_options,
        solver_threads=threads,
        verbose=verbose,
    )
    return master
