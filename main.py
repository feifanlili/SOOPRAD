# ++++++++++ LOCAL IMPORTS ++++++++++++++
from test_2dfunctions import ObjectiveFunction
from scipy_optimizer import SciPyOptimizer
from deap_optimizer import GA_Optimizer

# ++++++++++ PACKAGE IMPORTS ++++++++++++
import numpy as np


def main():
    sphere_func = ObjectiveFunction("sphere")
    sine_squared_func = ObjectiveFunction("sine_squared")
    absolute_sum_func = ObjectiveFunction("absolute_sum")
    quadratic_func = ObjectiveFunction("quadratic")
    ackley_func = ObjectiveFunction("ackley")
    levy_func = ObjectiveFunction("levy")
    eggholder_func = ObjectiveFunction("eggholder")
    ############################################################################
    # User Input:
    ############################################################################
    # 1. Define objective function
    objective_function = eggholder_func.f
    # 2. Define bounds
    bounds = [(-20, 20), (-20, 20)]
    bounds_egg = [(-512, 512), (-512, 512)]
    # 3. Define optimizer parameters
    params = {"method": "shgo", "n": 100, "iters": 1, "sampling_method": "simplicial"}
    params_de = {
        "method": "differential_evolution",
        "strategy": "best1bin",
        "maxiter": 1000,
        "popsize": 15,
        "tol": 0.01,
        "mutation": (0.5, 1),
        "recombination": 0.7,
    }
    # optimizer = SciPyOptimizer(objective_function, bounds_egg, params_de)
    optimizer = GA_Optimizer(
        objective_function,
        bounds_egg,
    )
    # 4. Run optimization
    result = optimizer.optimize()
    # 5. Visualize result
    optimizer.result_visualization()


if __name__ == "__main__":
    main()
