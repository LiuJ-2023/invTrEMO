import numpy as np
import matplotlib.pyplot as plt
import copy
from datetime import datetime
from pymoo.visualization.scatter import Scatter
import pickle
from pymoo.visualization.scatter import Scatter
from pymoo.visualization.scatter import Scatter
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.moead import MOEAD
from pymoo.optimize import minimize
from pymoo.util.ref_dirs import get_reference_directions
import Crash
from Utils import Inv_Transform

class OPT(Problem):
    def __init__(self,input_func,n_var,n_obj,xl=0,xu=1):
        super().__init__(n_var=n_var,n_obj=n_obj,xl=xl,xu=xu)
        self.input_func = input_func

    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = self.input_func(x)

test_problem_source1 = Crash.Crash_Source_Case2()
obj_problem = OPT(test_problem_source1,n_var=test_problem_source1.dim,n_obj=test_problem_source1.obj_num)
algorithm = NSGA2(
                pop_size = 200,
                )
res = minimize(obj_problem,
        algorithm,
        ('n_gen', 200))
Pareto_X = res.X
Pareto_F = res.F
Scatter().add(Pareto_F).show()  
Pareto_W, Pareto_X = Inv_Transform(Pareto_X, Pareto_F)
source_models = []
source_models.append([Pareto_X[:,1:],Pareto_W])
pickle.dump(source_models, open("Source_Case2.p", "wb"))
