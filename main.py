import numpy as np
############################################################################
from optimizers.scipy_optimizer import SciPyOptimizer
from optimizers.deap_optimizer import GA_Optimizer
from optimizers.utils.benckmark_functions import ObjectiveFunction
from optimizers.utils.visualizer import OptimizerVisualizer

def main():
    # === 2D Objective Functions ===
    sphere_func = ObjectiveFunction("sphere", dimension=2)
    sine_squared_func = ObjectiveFunction("sine_squared", dimension=2)
    absolute_sum_func = ObjectiveFunction("absolute_sum", dimension=2)
    quadratic_func = ObjectiveFunction("quadratic", dimension=2)
    ackley_func = ObjectiveFunction("ackley", dimension=2)
    levy_func = ObjectiveFunction("levy", dimension=2)
    eggholder_func = ObjectiveFunction("eggholder", dimension=2)

    # === 1D Objective Functions ===
    sphere_1d_func = ObjectiveFunction("sphere_1d", dimension=1)
    abs_1d_func = ObjectiveFunction("abs_1d", dimension=1)
    sin_1d_func = ObjectiveFunction("sin_1d", dimension=1)
    quadratic_1d_func = ObjectiveFunction("quadratic_1d", dimension=1)
    step_1d_func = ObjectiveFunction("step_1d", dimension=1)
    noisy_quad_1d_func = ObjectiveFunction("noisy_quad_1d", dimension=1)
    ############################################################################
    obj = sine_squared_func
    ############################################################################
    # User Input: 
    ############################################################################
    # 1. Define objective function
    objective_function = obj.f
    # 2. Define bounds
    bounds = obj.bounds
    # 3. Define optimizer parameters
    optimizer = GA_Optimizer(objective_function, bounds, log_population=True,log_summary=True)
    # 4. Run optimization 
    optimizer.optimize()

    # Create visualizer
    vis = OptimizerVisualizer(
        population_log_path="logs/run_population.json",
        objective_func=objective_function,
        bounds=bounds
    )

    # # Run the animation
    vis.replay(pause_time=0.1)

    
if __name__ == "__main__":
    main()
