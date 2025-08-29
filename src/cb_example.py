from gurobipy import GRB
import pyomo.environ as pe
from pyomo.core.expr.taylor_series import taylor_series_expansion
from pyomo.contrib import appsi
m = pe.ConcreteModel()
m.x = pe.Var(bounds=(0, 4))
m.y = pe.Var(within=pe.Integers, bounds=(0, None))
m.obj = pe.Objective(expr=2*m.x + m.y)
m.cons = pe.ConstraintList()  # for the cutting planes
def _add_cut(xval):
    # a function to generate the cut
    m.x.value = xval
    return m.cons.add(m.y >= taylor_series_expansion((m.x - 2)**2))
_c = _add_cut(0)  # start with 2 cuts at the bounds of x
_c = _add_cut(4)  # this is an arbitrary choice
opt = appsi.solvers.Gurobi()
opt.config.stream_solver = True
opt.set_instance(m)
opt.gurobi_options['PreCrush'] = 1
opt.gurobi_options['LazyConstraints'] = 1
def my_callback(cb_m, cb_opt, cb_where):
    if cb_where == GRB.Callback.MIPSOL:
        print(f"Oh we're calling the callback from {cb_where} Mommy!!!")
        cb_opt.cbGetSolution(vars=[cb_m.x, cb_m.y])
        if cb_m.y.value < (cb_m.x.value - 2)**2 - 1e-6:
            cb_opt.cbLazy(_add_cut(cb_m.x.value))
opt.set_callback(my_callback)
res = opt.solve(m)