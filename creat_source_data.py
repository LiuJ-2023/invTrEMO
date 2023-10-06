import numpy as np
import matplotlib.pyplot as plt
import dtlz
from Utils import Inv_Transform
import pickle
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.optimize import minimize
from pymoo.util.ref_dirs import get_reference_directions

class OPT(Problem):
    def __init__(self,input_func,n_var,n_obj,xl=0,xu=1):
        super().__init__(n_var=n_var,n_obj=n_obj,xl=xl,xu=xu)
        self.input_func = input_func

    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = self.input_func(x)

# Generate high similarity source
PT = dtlz.DTLZ2(obj_num=3,n_var=6, delta1 = 0.9, delta2=0.05)
w_all = get_reference_directions("energy", PT.obj_num, 100, seed=1)
obj_problem = OPT(PT,n_var=PT.dim,n_obj=PT.obj_num)
algorithm = NSGA3(
                pop_size = 100,
                ref_dirs = w_all
                )
res = minimize(obj_problem,
        algorithm,
        ('n_gen', 500))
Pareto_X = res.X
Pareto_F = res.F
Pareto_W, Pareto_X = Inv_Transform(Pareto_X, Pareto_F)
source_models = [Pareto_W,Pareto_X]
pickle.dump(source_models, open("source_data_dtlz2_hs.p", "wb"))

# Generate medium similarity source
PT = dtlz.DTLZ2(obj_num=3,n_var=6, delta1 = 0.7, delta2=0.25)
w_all = get_reference_directions("energy", PT.obj_num, 100, seed=1)
obj_problem = OPT(PT,n_var=PT.dim,n_obj=PT.obj_num)
algorithm = NSGA3(
                pop_size = 100,
                ref_dirs = w_all
                )
res = minimize(obj_problem,
        algorithm,
        ('n_gen', 500))
Pareto_X = res.X
Pareto_F = res.F
Pareto_W, Pareto_X = Inv_Transform(Pareto_X, Pareto_F)
source_models = [Pareto_W,Pareto_X]
pickle.dump(source_models, open("source_data_dtlz2_ms.p", "wb"))

# Generate low similarity source
PT = dtlz.DTLZ2(obj_num=3,n_var=6, delta1 = 0.3, delta2=0.4)
w_all = get_reference_directions("energy", PT.obj_num, 100, seed=1)
obj_problem = OPT(PT,n_var=PT.dim,n_obj=PT.obj_num)
algorithm = NSGA3(
                pop_size = 100,
                ref_dirs = w_all
                )
res = minimize(obj_problem,
        algorithm,
        ('n_gen', 500))
Pareto_X = res.X
Pareto_F = res.F
Pareto_W, Pareto_X = Inv_Transform(Pareto_X, Pareto_F)
source_models = [Pareto_W,Pareto_X]
pickle.dump(source_models, open("source_data_dtlz2_ls.p", "wb"))