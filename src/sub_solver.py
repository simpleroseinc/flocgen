# sub_solver.py
from typing import Any, Dict, Tuple
from pyomo.environ import value
from pyomo.opt import TerminationCondition
from sub import build_subproblem_for_scenario, set_sub_probelm_rhs
from utils import get_solver, setup_benders_sub_solver, solve_model

# Process-local cache & configuration
sub_solver_ctx: Dict[str, Any] = {
    "data": None,  # full 'data' dict used to create subproblems
    "solver_name": None,  # e.g., "gurobi"
    "options": None,  # e.g., {"InfUnbdInfo": 1, "Method": 1}
    "verbose": False,
    "threads": 1,
    "cache": {},  # scenario -> (sub_model, persistent_solver)
}


def sub_solver_init(
    data: Dict[str, Any],
    solver_name: str,
    options: Dict[str, Any] | None,
    threads: int = 1,
    verbose: bool = False,
) -> None:
    """
    Initializer that runs once in each worker process.
    We store shared inputs and create a per-process cache for (sub, solver).
    """
    sub_solver_ctx["data"] = data
    sub_solver_ctx["solver_name"] = solver_name
    sub_solver_ctx["options"] = options
    sub_solver_ctx["verbose"] = verbose
    sub_solver_ctx["threads"] = 1
    sub_solver_ctx["cache"] = {}  # fresh cache per subprocess


def solve_sub(
    scenario: Any, facility_open: Dict[Any, float]
) -> Tuple[str, str, Any, Dict[str, int], float | None, Dict[Any, float], float | None]:
    """
    Solve ONE scenario subproblem for the given incumbent 'facility_open'.

    Returns a tuple describing the cut:
      - ("feas", s, const, {i: coeff_i}, None)              -> feasibility cut: const + Σ coeff_i * x_i >= 0
      - ("opt",  s, const, {i: coeff_i}, operating_cost)    -> optimality cut: θ_s >= const + Σ coeff_i * x_i
      - ("other", s, status_string, {}, None)               -> anything else (e.g., time limit), you decide policy
    """
    data = sub_solver_ctx["data"]
    solver_name = sub_solver_ctx["solver_name"]
    options = sub_solver_ctx["options"]
    verbose = sub_solver_ctx["verbose"]
    threads = sub_solver_ctx["threads"]
    cache: Dict[Any, tuple] = sub_solver_ctx["cache"]

    # Build once per process per scenario; then reuse and just update RHS
    if scenario not in cache:
        sub = build_subproblem_for_scenario(data, scenario, facility_open)
        ss = get_solver(solver_name)  # persistent for Gurobi
        setup_benders_sub_solver(
            model=sub,
            solver=ss,
            options=options,
            solver_threads=threads,
            verbose=verbose,
        )
        cache[scenario] = (sub, ss)
    else:
        sub, ss = cache[scenario]
        # Your function to refresh RHS under the new incumbent
        set_sub_probelm_rhs(sub, ss, facility_open)
    # Collect statistics
    stats = {"sub_cons": sub.nconstraints(), "sub_vars": sub.nvariables()}
    # Solve (persistent path)
    sub_result = ss.solve(sub)
    termination = sub_result.solver.termination_condition

    # Check if solver supports dual/ray extraction
    if not hasattr(ss, "get_linear_constraint_attr"):
        raise RuntimeError(
            f"Solver '{solver_name}' does not support dual or ray extraction needed for Benders decomposition."
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

    if termination == TerminationCondition.infeasible:
        # Coefficents for feasibility cut
        feas_const = sum(
            ray_on(sub.satisfying_customer_demand[j]) * value(sub.customer_demand[j])
            for j in sub.CUSTOMERS
        )
        feas_coeffs = {
            i: ray_on(sub.facility_capacity_limits[i]) * value(sub.facility_capacity[i])
            for i in sub.FACILITIES
        }
        return (
            "feas",
            str(termination),
            scenario,
            stats,
            feas_const,
            feas_coeffs,
            None,
        )

    if termination == TerminationCondition.optimal:
        # Coefficients for optimality cut
        opt_const = sum(
            dual_on(sub.satisfying_customer_demand[j]) * value(sub.customer_demand[j])
            for j in sub.CUSTOMERS
        )
        opt_coeffs = {
            i: dual_on(sub.facility_capacity_limits[i])
            * value(sub.facility_capacity[i])
            for i in sub.FACILITIES
        }
        operating_cost = value(sub.objective)
        return (
            "opt",
            str(termination),
            scenario,
            stats,
            opt_const,
            opt_coeffs,
            operating_cost,
        )

    # e.g., time limit, numeric issues, etc.
    return ("other", str(termination), scenario, stats, None, {}, None)
