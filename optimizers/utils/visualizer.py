import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

class OptimizerVisualizer:
    """
    Visualizer for plotting the evolution of populations over generations.

    Supports both 1D and 2D problems.
    """

    def __init__(self, population_log_path, objective_func, bounds):
        """
        Initializes the visualizer.

        Args:
            population_log_path (str or Path): Path to the JSON log of population history.
            objective_func (callable): The objective function used in optimization.
            bounds (List[Tuple[float, float]]): Variable bounds (used to plot the domain).
        """
        self.population_log_path = Path(population_log_path)
        self.objective_func = objective_func
        self.bounds = bounds
        self.history = self._load_log()

        # Infer problem dimensionality from first individual's phenotype
        first_individual = self.history[0]["individuals"][0]
        self.dim = len(first_individual)

    def _load_log(self):
        """Load and parse JSON log of population history."""
        with open(self.population_log_path, "r") as f:
            return json.load(f)

    def replay(self, pause_time=0.3):
        """
        Replay the evolution process as an animation.

        Args:
            pause_time (float): Pause duration between frames (in seconds).
        """
        plt.style.use('bmh')
        plt.rcParams.update({"axes.facecolor": "white", "axes.grid": True})

        if self.dim == 1:
            self._replay_1d(pause_time)
        elif self.dim == 2:
            self._replay_2d(pause_time)
        else:
            raise NotImplementedError("Only 1D and 2D visualizations are supported.")

    def _replay_1d(self, pause_time):
        x = np.linspace(*self.bounds[0], 500)
        y = np.array([self.objective_func([xi]) for xi in x])

        # Matplotlib style
        plt.style.use('bmh')
        plt.rcParams.update({"axes.facecolor": "white", "axes.grid": True})

        fig, ax = plt.subplots(figsize=(10, 4))

        for gen_data in self.history:
            gen = gen_data["generation"]
            x_vals = [ind[0] for ind in gen_data["individuals"]]
            y_vals = gen_data["fitnesses"]

            ax.clear()
            ax.plot(x, y, label="Objective", color="skyblue")
            ax.scatter(x_vals, y_vals, color="red", label=f"Generation {gen}")
            ax.set_title(f"1D Evolution - Generation {gen}")
            ax.set_xlabel("X")
            ax.set_ylabel("Fitness")
            ax.legend()
            plt.pause(pause_time)

        plt.show()

    def _replay_2d(self, pause_time):
        x = np.linspace(*self.bounds[0], 100)
        y = np.linspace(*self.bounds[1], 100)
        xgrid, ygrid = np.meshgrid(x, y)
        xy = np.stack([xgrid, ygrid])
        zgrid = self.objective_func(xy)

        # Matplotlib style
        plt.style.use('bmh')
        plt.rcParams.update({"axes.facecolor": "white", "axes.grid": True})

        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
        ax1 = fig.add_subplot(1, 2, 1, projection="3d")
        ax2 = fig.add_subplot(1, 2, 2)

        for gen_data in self.history:
            gen = gen_data["generation"]
            pop = np.array(gen_data["individuals"])
            fits = np.array(gen_data["fitnesses"])
            x_vals, y_vals = pop[:, 0], pop[:, 1]

            ax1.clear()
            ax2.clear()

            ax1.plot_surface(xgrid, ygrid, zgrid, cmap="coolwarm", alpha=0.7)
            ax1.scatter(x_vals, y_vals, fits, c="tab:red", s=50)
            ax1.set_title(f"3D Surface - Gen {gen}")
            ax1.set_xlabel("X")
            ax1.set_ylabel("Y")
            ax1.set_zlabel("Fitness")

            ax2.contour(xgrid, ygrid, zgrid, cmap="coolwarm")
            ax2.scatter(x_vals, y_vals, c="tab:red", s=50)
            ax2.set_title(f"2D Contour - Gen {gen}")
            ax2.set_xlabel("X")
            ax2.set_ylabel("Y")

            plt.pause(pause_time)

        plt.show()
