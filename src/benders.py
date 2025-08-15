# benders.py
from master import *
from sub import *

def benders_solve(data,
                  master_solver="gurobi", sub_solver="gurobi",
                  max_iters=50, tol=1e-6, time_limit=None, log=True):
    """
    Multi-cut Benders for the FLoC model.

    Returns (master_model, history) where history logs bounds per iteration.
    """
    M = build_master(data, use_capacity_sufficiency=True)
    ms = SolverFactory(master_solver)
    ss = SolverFactory(sub_solver)
    if time_limit:
        try:
            ms.options["timelimit"] = time_limit
        except Exception:
            pass

    history = []

    bestUB = float("inf")   # incumbent objective
    bestLB = -float("inf")  # Benders lower bound (master objective)

    for it in range(1, max_iters+1):
        # ---- solve master
        resM = ms.solve(M, tee=log)
        LB = value(M.Obj)           # master gives a lower bound
        bestLB = max(bestLB, LB)

        # Read x and current theta
        xbar = {i: round(value(M.x[i])) for i in M.F}
        thetabar = {s: value(M.theta[s]) for s in M.S}

        # ---- solve each scenario dual and add cuts
        expected_recourse = 0.0
        violated = False

        for s in M.S:
            D = build_dual_subproblem_for_scenario(data, s, xbar)
            if time_limit:
                try:
                    ss.options["timelimit"] = time_limit
                except Exception:
                    pass
            resS = ss.solve(D, tee=False)

            q_s = value(D.Obj)              # subproblem optimal (dual) value
            expected_recourse += value(M.prob[s]) * q_s

            # Duals → Benders cut: theta[s] ≥ Σ_j d_js * pi_j  + Σ_i cap_i * mu_i * x_i
            # Build the RHS expression using the *dual* solution we just found
            rhs = sum(value(D.demand[j])*value(D.pi[j]) for j in D.C) \
                + sum(value(D.cap[i])*xbar[i]*value(D.mu[i]) for i in D.F)

            # numeric guard: rhs can be tiny negative due to numerical noise
            rhs_eps = max(rhs, 0.0)

            # If cut is violated (theta[s] < rhs - tol), add it
            if value(M.theta[s]) < rhs_eps - tol:
                M.BendersCuts.add(M.theta[s] >=
                                  sum(float(D.demand[j]) * value(D.pi[j]) for j in D.C) +
                                  sum(float(D.cap[i]) * M.x[i] * value(D.mu[i]) for i in D.F))
                violated = True

        # Update UB using the current xbar with exact subproblem value
        # (fixed cost + expected recourse computed from subproblems)
        fixed_cost = sum(value(M.fixed_cost[i]) * xbar[i] for i in M.F)
        UB = fixed_cost + expected_recourse
        bestUB = min(bestUB, UB)

        history.append({"iter": it, "LB": bestLB, "UB": bestUB,
                        "gap": (bestUB - bestLB)/max(1.0, abs(bestUB))})

        if log:
            print(f"[Iter {it}] LB={bestLB:.6g}  UB={bestUB:.6g}  "
                  f"gap={(bestUB-bestLB):.3e}  rel={history[-1]['gap']:.3e}")

        # Convergence if no scenario added a violated cut AND master’s thetas match subproblem values
        if (not violated) and abs(bestUB - bestLB) <= tol * max(1.0, abs(bestUB)):
            if log:
                print("Benders converged.")
            break

    return M, history
