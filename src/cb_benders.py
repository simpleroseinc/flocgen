# cb_benders.py
import sys
from time import sleep
import multiprocessing as mp
from gurobipy import GRB
from pyomo.environ import Var
from pyomo.contrib import appsi
from pyomo.opt import TerminationCondition
from master import *
from sub import *
from sub_solver import *


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

    # Set up for parallel subproblem solves
    num_incumbent = 0  # Global counter for number of incumbents
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
    sub_options = None
    if solver == "gurobi":
        sub_options = {
            "InfUnbdInfo": 1,  # To get unbounded ray information for the dual of the primal problem
            "Method": 1,  # Use dual simplex so we can get Farkas Rays (i.e. direction of unboundedness for the dual) for infeasible primal subproblems
        }
    with ctx.Pool(
        processes=num_scenarios,
        initializer=sub_solver_init,
        initargs=(data, solver, sub_options, threads, verbose),
        maxtasksperchild=int(max_time),
    ) as pool:

        # Define callback function that will be called by the solver for each incumbent
        def benders_callback(mod, sol, where):
            """
            Gurobi specific callback to add Benders cuts for each incumbent solution.
            """
            # If no incumbent solution, do nothing (Gurobi specific)
            if where != GRB.Callback.MIPSOL:
                return

            # Get master problem statistics
            master_cons, master_vars = mod.nconstraints(), mod.nvariables()
            # Get master incumbent solution (facility_open and sub_variable_cost)
            sol.cbGetSolution(
                vars=list(
                    mod.component_data_objects(Var, active=True, descend_into=True)
                )
            )
            facility_open = {i: value(mod.facility_open[i]) for i in mod.FACILITIES}
            sub_variable_cost = {
                s: value(mod.sub_variable_cost[s]) for s in mod.SCENARIOS
            }

            # Setup for stats collection
            nonlocal num_incumbent
            num_incumbent += 1
            feas_violation = opt_violation = 0
            sub_cons = sub_vars = 0
            expected_operating_cost = 0.0

            # Solve subproblems in parallel
            jobs = [(s, facility_open) for s in mod.SCENARIOS]
            # starmap is a barrier, returns after all sub-solves complete
            results = pool.starmap(solve_sub, jobs)

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
                    feas_violation += 1
                    lhs = const + sum(
                        coeff[i] * master.facility_open[i] for i in master.FACILITIES
                    )
                    con = master.BendersCuts.add(
                        lhs
                        >= 0 + tol
                        # Greater than 0 (i.e. >=) here b/c we haven't transformed our customer demand rule to <= constraints so we have to flip the sign
                    )
                    sol.cbLazy(con)
                elif kind == "opt":
                    expected_operating_cost += value(master.prob[s]) * operating_cost
                    if operating_cost > sub_variable_cost[s] + tol:
                        opt_violation += 1
                        lhs = const + sum(
                            coeff[i] * master.facility_open[i]
                            for i in master.FACILITIES
                        )
                        con = master.BendersCuts.add(
                            lhs <= master.sub_variable_cost[s] - tol
                        )
                        sol.cbLazy(con)
                else:
                    if verbose:
                        print(
                            f"Solver returned '{termination_name}' for scenario {s}, no cut added.",
                            file=sys.__stdout__,
                            flush=True,
                        )

            # Update upper bound
            fixed = sum(
                value(mod.fixed_cost[i]) * facility_open[i] for i in mod.FACILITIES
            )
            upper_bound = fixed + expected_operating_cost
            lower_bound = get_objective_value(mod)
            print(
                f"{num_incumbent} Statistics:\n"
                f"\tMaster: {master_cons} cons, {master_vars} vars\n"
                f"\tSubproblem: {sub_cons // num_scenarios} cons, {sub_vars // num_scenarios} vars, {num_scenarios} scenarios\n"
                f"\tViolations: {feas_violation} feas, {opt_violation} opt\n"
                f"\tBounds: {lower_bound:.2f} <= {upper_bound:.2f}",
                file=sys.__stdout__,
                flush=True,
            )
            return

        _ = solve_model(
            master,
            ms,
            callback=benders_callback,
            options=master_options,
            solver_threads=threads,
            verbose=verbose,
        )

    return master
