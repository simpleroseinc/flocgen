# utils.py
import os
import sys
from time import sleep
import psutil
import argparse
import importlib.util
from enum import Enum
from gen_data import *
from pyomo.environ import Objective, ConcreteModel, value
from pyomo.opt import SolverFactory, SolverResults
from rosepy.pyomo.pyomo_interface import PyomoInterface


# Utility functions
def is_package_installed(package_name):
    """
    Check if a package is installed (used for solvers).
    """
    return importlib.util.find_spec(package_name) is not None


def validate_state_code(state_code: str) -> str:
    """
    Validate the state code provided by the user.
    """
    if state_code.upper() not in VALID_STATE_CODES:
        raise argparse.ArgumentTypeError(
            f"Invalid state code: {state_code}. Must be one of {', '.join(VALID_STATE_CODES)}."
        )
    return state_code.upper()


def positive_int(val) -> int:
    """
    Validate that the value provided by the user is a positive integer.
    """
    ivalue = int(val)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError(f"{val} is an invalid positive int value")
    return ivalue


def non_negative_int(val) -> int:
    """
    Validate that the value provided by the user is a non-negative integer.
    """
    ivalue = int(val)
    if ivalue < 0:
        raise argparse.ArgumentTypeError(f"{val} is an invalid non-negative int value")
    return ivalue


def positive_float(val) -> float:
    """
    Validate that the value provided by the user is a positive float.
    """
    fvalue = float(val)
    if fvalue <= 0:
        raise argparse.ArgumentTypeError(f"{val} is an invalid positive float value")
    return fvalue


def get_physical_cores() -> int | None:
    """
    Get the number of physical CPU cores available on the machine.
    """
    try:
        return psutil.cpu_count(logical=False)
    except Exception as e:
        print(f"Error retrieving physical cores: {e}")
        return None


def get_solver(solver_name: str, callback: bool = False) -> SolverFactory:
    """
    Get the solver factory for the specified solver name.
    """
    if not is_package_installed(f"{solver_name}py"):
        raise RuntimeError(f"Solver '{solver_name}' is not installed.")
    if solver_name == "gurobi":
        if callback:
            solver = SolverFactory(f"appsi_{solver_name}")
        else:
            solver = SolverFactory(f"{solver_name}_persistent")
    elif solver_name == "highs":
        solver = SolverFactory(f"appsi_{solver_name}")
    elif solver_name == "rose":
        solver = SolverFactory(f"{solver_name}")
    else:
        raise RuntimeError(
            f"Invalid solver '{solver_name}' please choose gurobi, highs or rose."
        )
    return solver


def setup_benders_sub_solver(
    model: ConcreteModel,
    solver: SolverFactory,
    callback: callable = None,
    options: dict = None,
    solver_threads: int = 0,
    verbose: bool = False,
) -> Dict:
    """
    Prep the solver for Benders using the specified solver.
    Returns the updated options dictionary.
    """
    if not isinstance(model, ConcreteModel):
        raise ValueError("The model must be a ConcreteModel instance.")
    if not hasattr(solver, "solve") or not callable(getattr(solver, "solve")):
        raise ValueError("The solver object must have callable attribute 'solve'.")
    # Try to determine the solver interface name
    solver_iface_name = (
        getattr(solver.__class__, "__module__", "")
        or getattr(solver, "__name__", "")
        or ""
    )
    if hasattr(
        solver, "set_instance"
    ):  # For some solver interfaces you have to set the instance first before setting callbacks or options
        solver.set_instance(model)
    if hasattr(solver, "set_callback") and callback:
        solver.set_callback(callback)
    if hasattr(solver, "set_gurobi_param"):
        solver.set_gurobi_param("OutputFlag", verbose)
        solver.set_gurobi_param("Threads", solver_threads)
        if options:
            options["OutputFlag"] = verbose
            options["Threads"] = solver_threads
            for key, val in options.items():
                solver.set_gurobi_param(key, val)
        else:
            options = {"OutputFlag": verbose, "Threads": solver_threads}
        return options
    elif hasattr(solver, "gurobi_options"):
        solver.config.stream_solver = verbose
        solver.gurobi_options["OutputFlag"] = verbose
        solver.gurobi_options["Threads"] = solver_threads
        if options:
            options["OutputFlag"] = verbose
            options["Threads"] = solver_threads
            for key, val in options.items():
                solver.gurobi_options[key] = val
        else:
            options = {"OutputFlag": verbose, "Threads": solver_threads}
        return options
    elif hasattr(solver, "highs_options"):
        solver.config.stream_solver = verbose
        solver.highs_options["output_flag"] = verbose
        solver.highs_options["threads"] = solver_threads
        if options:
            options["output_flag"] = verbose
            options["threads"] = solver_threads
            for key, val in options.items():
                solver.highs_options[key] = val
        else:
            options = {"output_flag": verbose, "threads": solver_threads}
        return options
    elif "rose" in solver_iface_name:
        if options:
            options["rank_burls"] = solver_threads
            options["solver_engine"] = "rose_experimental_with_default_presolve"
        else:
            options = {
                "rank_burls": solver_threads,
                "solver_engine": "rose_experimental_with_default_presolve",
            }
        return options
    else:
        raise RuntimeError(f"Solver interface error for: '{solver_iface_name}'.")


def solve_model(
    model: ConcreteModel,
    solver: SolverFactory,
    callback: callable = None,
    options: dict = None,
    solver_threads: int = 0,
    verbose: bool = False,
) -> SolverResults:
    """
    Solve the Pyomo model using the specified solver and return the results.
    """
    if not isinstance(model, ConcreteModel):
        raise ValueError("The model must be a ConcreteModel instance.")
    if not hasattr(solver, "solve") or not callable(getattr(solver, "solve")):
        raise ValueError("The solver object must have callable attribute 'solve'.")
    # Try to determine the solver interface name
    solver_iface_name = (
        getattr(solver.__class__, "__module__", "")
        or getattr(solver, "__name__", "")
        or ""
    )
    if hasattr(
        solver, "set_instance"
    ):  # For some solver interfaces you have to set the instance first before setting callbacks or options
        solver.set_instance(model)
    if hasattr(solver, "set_callback") and callback:
        solver.set_callback(callback)
    if hasattr(solver, "set_gurobi_param"):
        solver.set_gurobi_param("OutputFlag", bool(verbose))
        solver.set_gurobi_param("Threads", solver_threads)
        if options:
            for key, val in options.items():
                solver.set_gurobi_param(key, val)
        return solver.solve(model)
    elif hasattr(solver, "gurobi_options"):
        solver.gurobi_options["OutputFlag"] = verbose
        solver.gurobi_options["Threads"] = solver_threads
        if options:
            for key, val in options.items():
                solver.gurobi_options[key] = val
        solver.config.stream_solver = verbose
        return solver.solve(model)
    elif hasattr(solver, "highs_options"):
        solver.config.stream_solver = verbose
        solver.highs_options["output_flag"] = verbose
        solver.highs_options["threads"] = verbose
        if options:
            for key, val in options.items():
                solver.highs_options[key] = val
        return solver.solve(model)
    elif "rose" in solver_iface_name:
        if options:
            options["rank_burls"] = solver_threads
            options["solver_engine"] = "rose_experimental_with_default_presolve"
        else:
            options = {
                "rank_burls": solver_threads,
                "solver_engine": "rose_experimental_with_default_presolve",
            }
        return solver.solve(model, options=options)
    else:
        raise RuntimeError(f"Solver interface error for: '{solver_iface_name}'.")


def get_objective_value(model: ConcreteModel) -> float:
    """
    Get the value of the objective function from the Pyomo model.
    """
    # Find the active objective(s) in the model
    objs = [obj for obj in model.component_objects(Objective, active=True)]
    if not objs:
        raise RuntimeError("No active objective found in the model.")
    if len(objs) > 1:
        print("Warning: Multiple active objectives found, returning the first one.")
    obj = objs[0]
    return value(obj)


class CapacityRule(Enum):
    """
    Enumeration of capacity rules for the required (sufficient) production capacity constraint in the Facility Location problem.
    MIN: Ensure there is enough capacity to meet minimum demand (not robust).
    MAX: Ensure there is enough capacity to meet demand under any circumstance (robust solution).
    AVERAGE: Ensure there is enough capacity to meet average demand (not robust).
    EXPECTED: Ensure there is enough capacity to meet expected demand (not robust).
    """

    MIN = "min"
    MAX = "max"
    AVERAGE = "average"
    EXPECTED = "expected"


def calculate_capacity_threshold(
    model: ConcreteModel, capacity_rule: CapacityRule
) -> float:
    """
    Calculate the capacity threshold for the required (sufficient) production capacity constraint based on the specified capacity rule.
    MIN: Ensure there is enough capacity to meet minimum demand (not robust).
    MAX: Ensure there is enough capacity to meet demand under any circumstance (robust solution).
    AVERAGE: Ensure there is enough capacity to meet average demand (not robust).
    EXPECTED: Ensure there is enough capacity to meet expected demand (not robust).
    """
    if capacity_rule == CapacityRule.MIN:
        return min(
            sum(value(model.customer_demand[j, s]) for j in model.CUSTOMERS)
            for s in model.SCENARIOS
        )
    elif capacity_rule == CapacityRule.MAX:
        return max(
            sum(value(model.customer_demand[j, s]) for j in model.CUSTOMERS)
            for s in model.SCENARIOS
        )
    elif capacity_rule == CapacityRule.AVERAGE:
        return sum(
            sum(value(model.customer_demand[j, s]) for j in model.CUSTOMERS)
            for s in model.SCENARIOS
        ) / len(model.SCENARIOS)
    elif capacity_rule == CapacityRule.EXPECTED:
        return sum(
            sum(
                value(model.customer_demand[j, s]) * value(model.prob[s])
                for j in model.CUSTOMERS
            )
            for s in model.SCENARIOS
        )
    else:
        raise ValueError(f"Unsupported capacity rule: {capacity_rule}")


def sanity_check_benders_solution(
    master: ConcreteModel, expected_operating_cost: float, tol=1e-6
):
    """
    Sanity check for the Benders decomposition solution.
    Compares the objective value of the master problem with the fixed cost + expected operating cost coming from the subproblems.
    """
    obj = get_objective_value(master)
    fixed = sum(
        value(master.fixed_cost[i]) * value(master.facility_open[i])
        for i in master.FACILITIES
    )
    rhs = fixed + expected_operating_cost
    gap = abs(obj - rhs) / abs(obj) if abs(obj) >= 1.0 else abs(obj - rhs)
    if gap > tol:
        raise RuntimeError(
            f"Decomposition failed: gap={gap} > tol={tol}, master objective:{obj}"
        )
