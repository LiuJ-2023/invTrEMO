Source code of "Bayesian Inverse Transfer in Evolutionary Multiobjective Optimization"

@article{10.1145/3674152,
author = {Liu, Jiao and Gupta, Abhishek and Ong, Yew-Soon},
title = {Bayesian Inverse Transfer in Evolutionary Multiobjective Optimization},
year = {2024},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3674152},
doi = {10.1145/3674152},
abstract = {Transfer optimization enables data-efficient optimization of a target task by leveraging experiential priors from related source tasks. This is especially useful in multiobjective optimization settings where a set of trade-off solutions is sought under tight evaluation budgets. In this paper, we introduce a novel concept of inverse transfer in multiobjective optimization. Inverse transfer stands out by employing Bayesian inverse Gaussian process models to map performance vectors in the objective space to population search distributions in task-specific decision space, facilitating knowledge transfer through objective space unification. Building upon this idea, we introduce the first Inverse Transfer Evolutionary Multiobjective Optimizer (invTrEMO). A key highlight of invTrEMO is its ability to harness the common objective functions prevalent in many application areas, even when decision spaces do not precisely align between tasks. This allows invTrEMO to uniquely and effectively utilize information from heterogeneous source tasks as well. Furthermore, invTrEMO yields high-precision inverse models as a significant byproduct, enabling the generation of tailored solutions on-demand based on user preferences. Empirical studies on multi- and many-objective benchmark problems, as well as a practical case study, showcase the faster convergence rate and modelling accuracy of the invTrEMO relative to state-of-the-art evolutionary and Bayesian optimization algorithms. The source code of the invTrEMO is made available at .},
note = {Just Accepted},
journal = {ACM Trans. Evol. Learn. Optim.},
month = jun,
keywords = {Inverse transfer, multiobjective optimization, evolutionary algorithms, Gaussian processes}
}
