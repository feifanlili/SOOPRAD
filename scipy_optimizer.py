import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

class SciPyOptimizer:
    DEFAULT_PARAMS = {
        "shgo": {
            "method": "shgo",
            "n": 100,
            "iters": 1,
            "sampling_method": "simplicial"
        },
        "dual_annealing": {
            "method": "dual_annealing",
            "maxiter": 1000,
            "initial_temp": 5230.0,
            "visit": 2.62,
            "accept": -5.0,
            "maxfun": 10000000
        },
        "differential_evolution": {
            "method": "differential_evolution",
            "strategy": "best1bin",
            "maxiter": 1000,
            "popsize": 15,
            "tol": 0.01,
            "mutation": (0.5, 1),
            "recombination": 0.7
        }
    }

    def __init__(self, objective_func, bounds, params=None):
        """
        SciPy-based global optimizer with flexible parameter settings.
        
        Parameters:
        - objective_func: function to optimize.
        - bounds: list of tuples [(x_min, x_max), (y_min, y_max)].
        - params: dictionary with `method` and optimizer-specific parameters.
                  If `params` is None, it defaults to 'shgo'.
        """
        self.objective_func = objective_func
        self.bounds = bounds

        # Use default settings if params is None
        if params is None:
            self.params = self.DEFAULT_PARAMS["shgo"]  # Default to SHGO
        else:
            # Checks if "method" exists in the dictionary, if "method" exists, it returns its value.
            # If "method" does not exist, it returns "shgo" as the default value.
            method = params.get("method", "shgo")  # Default to SHGO if not specified
            self.params = {**self.DEFAULT_PARAMS.get(method, {}), **params}  # Merge defaults
        
        self.method = self.params["method"]
        self.results = None  # Store optimization results

    def optimize(self):
        """
        Runs the selected optimization method with user-defined parameters.
        
        Returns:
        - Optimization result object.
        """
        # Collection of the scipy optimizer
        optimizers = {
            "shgo": optimize.shgo,
            "dual_annealing": optimize.dual_annealing,
            "differential_evolution": optimize.differential_evolution
        }
        # Check input
        if self.method not in optimizers:
            raise ValueError(f"Method '{self.method}' not supported. Choose from {list(optimizers.keys())}.")
        # Define scipy optimizer instance based on user input
        optimizer = optimizers[self.method]
        
        # Remove "method" from params before passing, leaving the optimizer setup parameter for 
        # the scipy optimizer initiation
        params_to_use = {k: v for k, v in self.params.items() if k != "method"}
        self.results = optimizer(self.objective_func, self.bounds, **params_to_use)
        
        return self.results

    def result_visualization(self):
        """
        Visualizes the optimization results using 3D surface and contour plots.
        Just for testing, only suitable for the 2D optimization problem.
        """
        if self.results is None:
            raise ValueError("Run optimize() first before visualizing results.")
        
        # Generate grid for visualization
        x = np.arange(self.bounds[0][0]-1, self.bounds[0][1]+1)
        y = np.arange(self.bounds[1][0]-1, self.bounds[1][1]+1)
        xgrid, ygrid = np.meshgrid(x, y)
        xy = np.stack([xgrid, ygrid])
        z_grid = self.objective_func(xy)

        # Matplotlib style
        plt.style.use('bmh')
        plt.rcParams.update({"axes.facecolor": "white", "axes.grid": True})

        # Create subplots
        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
        ax1 = fig.add_subplot(1, 2, 1, projection="3d")
        ax2 = fig.add_subplot(1, 2, 2)

        # Surface plot
        ax1.plot_surface(xgrid, ygrid, z_grid, cmap="coolwarm")
        
        # Contour plot
        ax2.contour(xgrid, ygrid, z_grid, cmap="coolwarm")

        # Scatter the optimization result
        best_x, best_y = self.results.x
        best_z = self.objective_func(self.results.x)
        ax1.scatter(best_x, best_y, best_z, color="red", label=f"Best ({self.method})", marker='x', s=100)
        ax2.scatter(best_x, best_y, color="red", label=f"Best ({self.method})", marker='x', s=100)

        # Titles and labels
        ax1.set_title("Objective Function Surface")
        ax1.set_xlabel("X")
        ax1.set_ylabel("Y")
        ax1.set_zlabel("Z")
        ax2.set_title("Objective Function Contour")
        ax2.set_xlabel("X")
        ax2.set_ylabel("Y")

        # Legends
        ax1.legend()
        ax2.legend()
        plt.show()

# Example Usage
if __name__ == "__main__":
    #################################################################
    # 1. Example: eggholder, 2d problem, suitable for visualization
    #################################################################
    def first_example():
        # Define the objective function (Eggholder function)
        def eggholder(x):
            return (-(x[1] + 47) * np.sin(np.sqrt(abs(x[0]/2 + (x[1] + 47))))
                    - x[0] * np.sin(np.sqrt(abs(x[0] - (x[1] + 47)))))

        # Define bounds
        bounds = [(-512, 512), (-512, 512)]

        # Run optimizer with default SHGO settings
        optimizer = SciPyOptimizer(eggholder, bounds)
        result = optimizer.optimize()
        print("\n2D Eggholder Function Optimization")
        print("Best solution (SHGO):", result.x)
        print("Minimum value (SHGO):", result.fun)
        optimizer.result_visualization()

        # Run optimizer with custom Differential Evolution settings
        de_params = {
            "method": "differential_evolution",
            "strategy": "best1bin",
            "maxiter": 500,
            "popsize": 20,
            "mutation": (0.5, 1),
            "recombination": 0.7
        }

        optimizer = SciPyOptimizer(eggholder, bounds, params=de_params)
        result = optimizer.optimize()
        print("\n2D Eggholder Function Optimization")
        print("Best solution (DE):", result.x)
        print("Minimum value (DE):", result.fun)
        optimizer.result_visualization()
    #################################################################

    #################################################################
    # 2. Example: rosen, 4d problem, not available for visualization
    #################################################################
    def second_example():
        # 4D Rosenbrock function
        def rosenbrock(x):
            return optimize.rosen(x)  # Uses SciPy's built-in Rosenbrock function

        # 4D bounds
        bounds_4d = [(-2, 2)] * 4

        # Optimization parameters for Differential Evolution
        params_4d = {
            "method": "differential_evolution",
            "maxiter": 2000,
            "popsize": 20,
            "tol": 0.01
        }

        # Run optimizer
        optimizer_4d = SciPyOptimizer(rosenbrock, bounds_4d, params=params_4d)
        result_4d = optimizer_4d.optimize()

        # Print result
        print("\n4D Rosenbrock Function Optimization")
        print(f"Best solution: {result_4d.x}")
        print(f"Minimum value: {result_4d.fun}")

    first_example()