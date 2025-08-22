# utils.py
import argparse
from enum import Enum
from gen_data import *
from pyomo.environ import Objective, ConcreteModel, value


# Utility functions
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


def positive_float(val) -> float:
    """
    Validate that the value provided by the user is a positive float.
    """
    fvalue = float(val)
    if fvalue <= 0:
        raise argparse.ArgumentTypeError(f"{val} is an invalid positive float value")
    return fvalue


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
        raise RuntimeError(f"Decomposition failed: gap={gap} > tol={tol}")
