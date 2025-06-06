"""
main.py

Main entry point for running optimization and visualizing its progress.

This script:
1. Selects an objective function
2. Initializes an optimizer
3. Runs the optimization
4. Visualizes the population evolution

Optimizers supported: DEAP (GA, ES), SciPy
"""
import numpy as np
############################################################################
from optimizers.scipy_optimizer import SciPyOptimizer
from optimizers.deap_optimizer import GA_Optimizer
from optimizers.utils.benckmark_functions import ObjectiveFunction
from optimizers.utils.visualizer import OptimizerVisualizer
from optimizers.deap_optimizer import ES_Optimizer

def main():
    # -------------------------------------------------------------------------
    # Define Objective Functions (from benchmark)
    # -------------------------------------------------------------------------
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
    # -------------------------------------------------------------------------
    # Select Objective Function
    # -------------------------------------------------------------------------
    selected_obj = quadratic_func
    # -------------------------------------------------------------------------
    # Extract core optimization components
    # -------------------------------------------------------------------------
    objective_function = selected_obj.f
    bounds = selected_obj.bounds
    # -------------------------------------------------------------------------
    # Create Optimizer Instance
    # -------------------------------------------------------------------------
    optimizer = GA_Optimizer(
        objective_function,
        bounds=bounds,
        log_population=True,
        log_summary=True
    )
    # optimizer.list_registered_operators()
    # optimizer.reset_mate('cxOnePoint')
    # -------------------------------------------------------------------------
    # Run Optimization
    # -------------------------------------------------------------------------
    optimizer.optimize()
    # # -------------------------------------------------------------------------
    # # Post-Processing: Visualization
    # # -------------------------------------------------------------------------
    visualizer = OptimizerVisualizer(
        population_log_path="logs/run_population.json",
        objective_func=objective_function,
        bounds=bounds
    )
    visualizer.replay(pause_time=0.2)

    
if __name__ == "__main__":
    main()