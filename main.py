import numpy as np
############################################################################
from optimizers.scipy_optimizer import SciPyOptimizer
from optimizers.deap_optimizer import GA_Optimizer
from optimizers.utils.test_2dfunctions import ObjectiveFunction

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
    bounds_egg = [(-512, 512), (-512, 512)]
    # 3. Define optimizer parameters
    optimizer = GA_Optimizer(objective_function, bounds_egg, log_population=False,log_summary=True)
    # 4. Run optimization 
    result = optimizer.optimize()

    
if __name__ == "__main__":
    main()
