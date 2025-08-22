# benders.py
from master import *
from sub import *
from pyomo.opt import *


def benders_solve(
    data,
    capacity_rule=CapacityRule.MAX,
    relax=False,
    solver="gurobi",
    max_iters=100,
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
    ms = SolverFactory(solver, solver_io="nl")
    ss = SolverFactory(solver, solver_io="nl")

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
        master_result = ms.solve(master, tee=verbose)
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

        # Reset expected operating cost for this iteration
        expected_operating_cost = 0.0
        for s in master.SCENARIOS:
            # Build subproblem for scenario s with fixed facility_open from master solution
            sub = build_subproblem_for_scenario(data, s, facility_open)
            # Get subproblem statistics
            sub_cons = sub.nconstraints()
            sub_vars = sub.nvariables()
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
                violated = True
                feas_violation += 1
                master.BendersCuts.add(
                    0
                    <= sum(  # Greater than 0 i.e. <= here b/c we haven't transformed our customer demand rule to <= constraints so we have to flip the sign
                        (
                            value(sub.dunbdd[sub.satisfying_customer_demand[i]])
                            if sub.satisfying_customer_demand[i] in sub.dunbdd
                            else 0
                        )
                        * value(sub.customer_demand[i])
                        for i in master.CUSTOMERS
                    )
                    + sum(
                        (
                            value(sub.dunbdd[sub.facility_capacity_limits[i]])
                            if sub.facility_capacity_limits[i] in sub.dunbdd
                            else 0
                        )
                        * value(sub.facility_capacity[i])
                        * master.facility_open[i]
                        for i in master.FACILITIES
                    )
                )
            elif termination == TerminationCondition.optimal:
                operating_cost = value(sub.objective)  # subproblem optimal value
                expected_operating_cost += value(master.prob[s]) * operating_cost
                if operating_cost > sub_variable_cost[s] + tol:
                    opt_violation += 1
                    violated = True
                    master.BendersCuts.add(
                        master.sub_variable_cost[s]
                        >= sum(
                            value(sub.dual[sub.satisfying_customer_demand[i]])
                            * value(sub.customer_demand[i])
                            for i in master.CUSTOMERS
                        )
                        + sum(
                            value(sub.dual[sub.facility_capacity_limits[i]])
                            * value(sub.facility_capacity[i])
                            * master.facility_open[i]
                            for i in master.FACILITIES
                        )
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
            print(f"No violations! Problem converged!")
            break
    # Sanity check: master objective value should be equal to fixed cost plus expected subproblem cost
    sanity_check_benders_solution(master, expected_operating_cost, tol=tol)
    return master
