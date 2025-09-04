# benders.py
import multiprocessing as mp
from pyomo.opt import *
from time import sleep
from master import *
from sub import *
from sub_solver import *
from utils import *


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
    Master problem is resolved from scratch each iteration.
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
    # Set up master solver
    ms = get_solver(solver)

    # Set up for parallel subproblem solves
    num_scenarios = len(list(master.SCENARIOS))
    num_cores = get_physical_cores()
    solver_threads = num_cores if threads == 0 else threads
    if num_cores < solver_threads * (1 + num_scenarios):
        print(
            f"WARNING: You're oversubscribing your CPU!\n"
            f"\tAvailable physical cores: {num_cores}\n"
            f"\tRequested: {solver_threads * (num_scenarios + 1)}."
        )
        sleep(5)  # Give user a chance to see the warning
    ctx = mp.get_context("spawn")
    sub_options = None  # Options will differ based on solver
    if solver == "gurobi":
        sub_options = {
            "InfUnbdInfo": 1,  # To get unbounded ray information for the dual of the primal problem
            "Method": 1,
            # Use dual simplex so we can get Farkas Rays (i.e. direction of unboundedness for the dual) for infeasible primal subproblems
        }

    # Solve each scenario subproblem in parallel
    # Apply cuts to master based on subproblem results
    violated = True  # Assume feasibility or optimality is violated so we enter the loop
    iteration = 0
    with ctx.Pool(
        processes=num_scenarios,
        initializer=sub_solver_init,
        initargs=(data, solver, sub_options, threads, verbose),
        maxtasksperchild=max_iters,
    ) as pool:

        while violated and iteration < max_iters:
            # Increment iteration counter
            iteration += 1
            # Reset violated flag and counters
            violated = False
            feas_violation = opt_violation = 0
            # Solve master
            master_result = solve_model(
                master, ms, solver_threads=threads, verbose=verbose
            )
            # Get Master solution (facility_open and sub_variable_cost)
            termination = master_result.solver.termination_condition
            if (
                termination == TerminationCondition.infeasible
                or termination == TerminationCondition.unbounded
            ):
                raise Exception(
                    f"Master problem {termination} at iteration {iteration}"
                )
            facility_open = {
                i: value(master.facility_open[i]) for i in master.FACILITIES
            }
            sub_variable_cost = {
                s: value(master.sub_variable_cost[s]) for s in master.SCENARIOS
            }
            # Get master problem statistics
            master_cons = master.nconstraints()
            master_vars = master.nvariables()

            # Solve subproblems in parallel
            jobs = [(s, facility_open) for s in master.SCENARIOS]
            # starmap is a barrier, returns after all sub-solves complete
            results = pool.starmap(solve_sub, jobs)

            # Reset stats
            expected_operating_cost = 0.0
            sub_cons = sub_vars = 0

            # Get results and add cuts
            for (
                kind,
                termination_name,
                s,
                stats,
                const,
                coeff,
                operating_cost,
            ) in results:
                sub_cons += stats["sub_cons"]
                sub_vars += stats["sub_vars"]

                if kind == "feas":
                    violated = True
                    feas_violation += 1
                    lhs = const + sum(
                        coeff[i] * master.facility_open[i] for i in master.FACILITIES
                    )
                    master.BendersCuts.add(
                        lhs
                        >= 0 + tol
                        # Greater than 0 (i.e. >=) here b/c we haven't transformed our customer demand rule to <= constraints so we have to flip the sign
                    )
                elif kind == "opt":
                    expected_operating_cost += value(master.prob[s]) * operating_cost
                    if operating_cost > sub_variable_cost[s] + tol:
                        violated = True
                        opt_violation += 1
                        lhs = const + sum(
                            coeff[i] * master.facility_open[i]
                            for i in master.FACILITIES
                        )
                        master.BendersCuts.add(lhs <= master.sub_variable_cost[s] - tol)
                else:
                    if verbose:
                        print(
                            f"Solver returned '{termination_name}' for scenario {s}, no cut added.",
                            file=sys.__stdout__,
                            flush=True,
                        )

            # Update upper bound
            fixed = sum(
                value(master.fixed_cost[i]) * facility_open[i]
                for i in master.FACILITIES
            )
            upper_bound = fixed + expected_operating_cost
            lower_bound = get_objective_value(master)  # Master objective value
            print(
                f"{iteration} Statistics:\n"
                f"\tMaster: {master_cons} cons, {master_vars} vars\n"
                f"\tSubproblem: {sub_cons // num_scenarios} cons, {sub_vars // num_scenarios} vars, {num_scenarios} scenarios\n"
                f"\tViolations: {feas_violation} feas, {opt_violation} opt\n"
                f"\tBounds: {lower_bound:.2f} <= {upper_bound:.2f}",
                file=sys.__stdout__,
                flush=True,
            )
            if not violated:
                # Sanity check: master objective value should be equal to fixed cost plus expected subproblem cost
                sanity_check_benders_solution(master, expected_operating_cost, tol=tol)
                print(f"No violations! Problem converged!")
                break
    if violated:
        print(f"Iteration limit reached ({max_iters} iterations) exiting.")
    return master
