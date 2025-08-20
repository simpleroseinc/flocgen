# benders.py
import sys
from sys import executable

from master import *
from sub import *
from pyomo.opt import *


def benders_solve(
    data,
    master_solver="gurobi",
    sub_solver="gurobi",
    max_iters=50,
    tol=1e-6,
    time_limit=None,
    log=True,
):
    """
    Multi-cut Benders for the two-stage Stochastic Facility Location problem.
    Returns (master_model, history) where history logs bounds per iteration.
    """
    master = build_master(data=data, capacity_rule=CapacityRule.AVERAGE)
    ms = SolverFactory(master_solver, solver_io="nl")
    print(f"Using master solver with interface {master_solver}")
    ss = SolverFactory(sub_solver, solver_io="nl")
    options = {"pre:solve": 0, "cvt:pre:all": 0, "alg:method": 1}
    print(f"Using sub solver with interface {sub_solver}")
    if time_limit:
        try:
            ms.options["timelimit"] = time_limit
        except Exception:
            pass

    history = []

    bestUB = float("inf")  # incumbent objective
    bestLB = -float("inf")  # Benders lower bound (master objective)

    for it in range(1, max_iters + 1):
        # ---- solve master
        resM = ms.solve(master, tee=log)
        LB = value(master.objective)  # master gives an upper bound
        bestLB = max(bestLB, LB)

        # Read facility_open and current theta
        facility_open = {
            i: round(value(master.facility_open[i])) for i in master.FACILITIES
        }
        sub_variable_cost = {
            s: value(master.sub_variable_cost[s]) for s in master.SCENARIOS
        }

        # ---- solve each scenario and add cuts
        expected_variable_cost = 0.0
        violated = False

        for s in master.SCENARIOS:
            sub = build_subproblem_for_scenario(data, s, facility_open)
            if time_limit:
                try:
                    ss.options["timelimit"] = time_limit
                except Exception:
                    pass
            sub_result = ss.solve(sub, tee=False, options=options)
            termination = sub_result.solver.termination_condition
            termination_name = str(termination)

            operating_cost = value(sub.objective)  # subproblem optimal (dual) value
            expected_variable_cost += value(master.prob[s]) * operating_cost

            print(
                f"\n\n\nSOLVER TERMINATION STATUS {termination_name} FOR SCENARIO {s}\n\n\n"
            )
            if termination == TerminationCondition.infeasible:
                violated = True
                print(f"Infeasible solution for scenario {s}")
                for j in sub.CUSTOMERS:
                    row = sub.satisfying_customer_demand[j]
                    val = sub.dunbdd[row] if row in sub.dunbdd else 0
                    print(f"j:{j}")
                    print(f"Dual rays: {val}")
            elif termination == TerminationCondition.optimal:
                print(
                    f"Optimal solution for scenario {s} with cost {operating_cost:.6g}"
                )
                if operating_cost > value(master.sub_variable_cost[s]) + tol:
                    violated = True
                    print(
                        f"Subproblem cost {operating_cost:.6g} exceeds master bound {value(master.sub_variable_cost[s]):.6g} for scenario {s}"
                    )
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

        if not violated:
            print(
                "\n\n\nNo violated cuts found in this iteration, checking for convergence."
            )
            break
        ms.solve(master, tee=log)
    sys.exit(0)


"""
            # Duals → Benders cut: theta[s] ≥ Σ_j d_js * pi_j  + Σ_i cap_i * mu_i * x_i
            # Build the RHS expression using the *dual* solution we just found
            rhs = sum(
                value(sub.customer_demand[j]) * value(sub.pi[j]) for j in sub.CUSTOMERS
            ) + sum(
                value(sub.facility_capacity[i]) * facility_open[i] * value(sub.mu[i])
                for i in sub.FACILITIES
            )

            # numeric guard: rhs can be tiny negative due to numerical noise
            rhs_eps = max(rhs, 0.0)

            # If cut is violated (theta[s] < rhs - tol), add it
            if value(master.sub_variable_cost[s]) < rhs_eps - tol:
                master.BendersCuts.add(
                    master.sub_variable_cost[s]
                    >= sum(
                        float(sub.customer_demand[j]) * value(sub.pi[j])
                        for j in sub.CUSTOMERS
                    )
                    + sum(
                        float(sub.facility_capacity[i])
                        * master.facility_open[i]
                        * value(sub.mu[i])
                        for i in sub.FACILITIES
                    )
                )
                violated = True

        # Update UB using the current facility_open with exact subproblem value
        # (fixed cost + expected recourse computed from subproblems)
        fixed_cost = sum(
            value(master.fixed_cost[i]) * facility_open[i] for i in master.FACILITIES
        )
        UB = fixed_cost + expected_variable_cost
        bestUB = min(bestUB, UB)

        history.append(
            {
                "iter": it,
                "LB": bestLB,
                "UB": bestUB,
                "gap": (bestUB - bestLB) / max(1.0, abs(bestUB)),
            }
        )

        if log:
            print(
                f"[Iter {it}] LB={bestLB:.6g}  UB={bestUB:.6g}  "
                f"gap={(bestUB-bestLB):.3e}  rel={history[-1]['gap']:.3e}"
            )

        # Convergence if no scenario added a violated cut AND master’s thetas match subproblem values
        if (not violated) and abs(bestUB - bestLB) <= tol * max(1.0, abs(bestUB)):
            if log:
                print("Benders converged.")
            break
    return master, history
"""
