# benders.py
from master import *
from sub import *
from pyomo.opt import *


def benders_solve(
    data,
    capacity_rule: CapacityRule = CapacityRule.MAX,
    max_iters: int = 100,
    relax: bool = False,
    solver: str = "gurobi",
    threads: int = 1,
    tol: float = 1e-6,
    verbose: bool = False,
) -> ConcreteModel:
    """
    Multi-cut Benders for the two-stage Stochastic Facility Location problem.
    Returns master_model.
    """
    # Check arguments
    tol = abs(tol)
    if max_iters < 1:
        raise ValueError("max_iters must be at least 1.")

    # Create master problem
    master = build_master(data=data, capacity_rule=capacity_rule)
    # If required by user relax integrality
    if relax:
        TransformationFactory("core.relax_integer_vars").apply_to(master)

    # Set up solvers
    ms = get_solver(solver)  # Master solver
    ss = get_solver(solver)  # Subproblem solver
    if not hasattr(ss, "get_linear_constraint_attr"):
        raise RuntimeError(
            f"Solver '{solver}' does not support dual or ray extraction needed for Benders decomposition."
        )

    # Solve each scenario and add cuts
    # Assume feasibility or optimality is violated so we enter the loop
    iteration = 0
    violated = True
    expected_operating_cost = 0.0
    upper_bound = float("inf")
    while violated and iteration < max_iters:
        # Increment iteration counter
        iteration += 1
        # Reset violated flag and counters
        violated = False
        feas_violation = opt_violation = 0
        # Solve master
        master_result = solve_model(master, ms, verbose=verbose)
        # Get Master solution (objective, facility_open and sub_variable_cost)
        termination = master_result.solver.termination_condition
        if (
            termination == TerminationCondition.infeasible
            or termination == TerminationCondition.unbounded
        ):
            raise Exception(f"Master problem {termination} at iteration {iteration}")
        lower_bound = get_objective_value(master)  # Master objective value
        facility_open = {i: value(master.facility_open[i]) for i in master.FACILITIES}
        sub_variable_cost = {
            s: value(master.sub_variable_cost[s]) for s in master.SCENARIOS
        }
        # Get master problem statistics
        master_cons = master.nconstraints()
        master_vars = master.nvariables()

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

        # Reset expected operating cost for this iteration
        expected_operating_cost = 0.0
        for s in master.SCENARIOS:
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
                violated = True
                feas_violation += 1
                master.BendersCuts.add(
                    sum(
                        ray_on(sub.satisfying_customer_demand[i])
                        * value(sub.customer_demand[i])
                        for i in sub.CUSTOMERS
                    )
                    + sum(
                        ray_on(sub.facility_capacity_limits[i])
                        * value(sub.facility_capacity[i])
                        * master.facility_open[i]
                        for i in sub.FACILITIES
                    )
                    >= 0 + tol
                    # Greater than 0 (i.e. >=) here b/c we haven't transformed our customer demand rule to <= constraints so we have to flip the sign
                )
            elif termination == TerminationCondition.optimal:
                operating_cost = value(sub.objective)  # subproblem optimal value
                expected_operating_cost += value(master.prob[s]) * operating_cost
                if operating_cost > sub_variable_cost[s] + tol:
                    opt_violation += 1
                    violated = True
                    master.BendersCuts.add(
                        sum(
                            dual_on(sub.satisfying_customer_demand[i])
                            * value(sub.customer_demand[i])
                            for i in sub.CUSTOMERS
                        )
                        + sum(
                            dual_on(sub.facility_capacity_limits[i])
                            * value(sub.facility_capacity[i])
                            * master.facility_open[i]
                            for i in sub.FACILITIES
                        )
                        <= master.sub_variable_cost[s] - tol
                    )
            else:
                raise Exception(
                    f"Solution for scenario {s} is neither optimal nor infeasible: {termination_name}"
                )
        # Update upper bound
        fixed = sum(
            value(master.fixed_cost[i]) * facility_open[i] for i in master.FACILITIES
        )
        upper_bound = fixed + expected_operating_cost
        print(
            f"{iteration} Statistics:\n\tMaster: {master_cons} cons, {master_vars} vars\n\tSubproblem: {sub_cons} cons, {sub_vars} vars, {len(list(master.SCENARIOS))} scenarios\n\tViolations: {feas_violation} feas, {opt_violation} opt\n\tBounds: {lower_bound:.2f} <= {upper_bound:.2f}"
        )
        if not violated:
            # Sanity check: master objective value should be equal to fixed cost plus expected subproblem cost
            sanity_check_benders_solution(master, expected_operating_cost, tol=tol)
            print(f"No violations! Problem converged!")
            break
    if violated:
        print(f"Iteration limit reached ({max_iters} iterations) exiting.")
    return master
