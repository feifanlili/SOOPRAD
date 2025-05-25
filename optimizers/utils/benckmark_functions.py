import numpy as np
import matplotlib.pyplot as plt

# --- 2D Benchmark Functions ---

def sphere_function(x):
    return x[0] ** 2.0 + x[1] ** 2.0

def sine_squared_function(x):
    return np.sin(x[0]) ** 2.0 + np.sin(x[1]) ** 2.0 + 0.005 * (x[0] ** 2.0 + x[1] ** 2.0)

def absolute_sum_function(x):
    return 50.0 * abs(x[0] + x[1]) + x[0] ** 2.0

def quadratic_function(x):
    return x[0] ** 2.0 + 50.0 * x[1] ** 2.0

def ackley_function(x):
    x0, x1 = x[0], x[1]
    a, b, c = 20.0, 0.2, 2 * np.pi

    sum_sq = 0.5 * (x0 ** 2 + x1 ** 2)
    cos_sum = 0.5 * (np.cos(c * x0) + np.cos(c * x1))

    return -a * np.exp(-b * np.sqrt(sum_sq)) - np.exp(cos_sum) + a + np.e

def levy_function(x):
    wx = (x[0] + 3.0) / 4.0
    wy = (x[1] + 3.0) / 4.0
    return (
        np.sin(np.pi * wx) ** 2
        + (wy - 1) ** 2 * (1 + np.sin(2 * np.pi * wy) ** 2)
        + (wx - 1) ** 2 * (1 + 10 * np.sin(np.pi * wx) ** 2)
    )

def eggholder_function(x):
    return (-(x[1] + 47) * np.sin(np.sqrt(abs(x[0] / 2 + (x[1] + 47))))
            - x[0] * np.sin(np.sqrt(abs(x[0] - (x[1] + 47)))))

# --- 1D Benchmark Functions ---

def sphere_function_1d(x):
    return x[0] ** 2.0

def abs_function_1d(x):
    return abs(x[0])

def sin_function_1d(x):
    return np.sin(x[0]) + 0.1 * x[0]

def quadratic_function_1d(x):
    return 3.0 * (x[0] - 2) ** 2.0 + 5

def step_function_1d(x):
    return np.floor(x[0]) + 5

def noisy_quadratic_1d(x):
    return x[0] ** 2 + np.random.normal(0, 0.1)

# --- Objective Function Dictionary ---

OBJECTIVE_FUNCTIONS = {
    # 2D
    "sphere": sphere_function,
    "sine_squared": sine_squared_function,
    "absolute_sum": absolute_sum_function,
    "quadratic": quadratic_function,
    "ackley": ackley_function,
    "levy": levy_function,
    "eggholder": eggholder_function,

    # 1D
    "sphere_1d": sphere_function_1d,
    "abs_1d": abs_function_1d,
    "sin_1d": sin_function_1d,
    "quadratic_1d": quadratic_function_1d,
    "step_1d": step_function_1d,
    "noisy_quad_1d": noisy_quadratic_1d,
}

# --- Default Bounds ---

DEFAULT_BOUNDS = {
    # 2D
    "sphere": [(-5, 5), (-5, 5)],
    "sine_squared": [(-10, 10), (-10, 10)],
    "absolute_sum": [(-5, 5), (-5, 5)],
    "quadratic": [(-5, 5), (-5, 5)],
    "ackley": [(-5, 5), (-5, 5)],
    "levy": [(-10, 10), (-10, 10)],
    "eggholder": [(-512, 512), (-512, 512)],

    # 1D
    "sphere_1d": [(-5, 5)],
    "abs_1d": [(-5, 5)],
    "sin_1d": [(-10, 10)],
    "quadratic_1d": [(-10, 10)],
    "step_1d": [(-10, 10)],
    "noisy_quad_1d": [(-5, 5)],
}

# --- ObjectiveFunction Class ---

class ObjectiveFunction:
    def __init__(self, function_name: str, dimension: int = 2):
        self.function_name = function_name
        self.dimension = dimension
        self.function = OBJECTIVE_FUNCTIONS.get(function_name)
        if self.function is None:
            raise ValueError(f"Function '{function_name}' not found.")
        self.bounds = DEFAULT_BOUNDS.get(function_name)
        if self.bounds is None:
            raise ValueError(f"Default bounds not defined for '{function_name}'.")

    def f(self, x):
        return self.function(x)

    def visualization(self, delta=0.1, showPlot=True):
        plt.style.use('bmh')
        plt.rcParams.update({
            "axes.facecolor": "white",
            "axes.grid": True
        })

        if self.dimension == 1:
            x = np.arange(self.bounds[0][0], self.bounds[0][1], delta)
            y = np.array([self.f([xi]) for xi in x])

            plt.figure(figsize=(8, 4))
            plt.plot(x, y, label=self.function_name)
            plt.xlabel("X")
            plt.ylabel("Fitness")
            plt.title(f"{self.function_name} (1D)")
            plt.legend()
            if showPlot:
                plt.show()

        elif self.dimension == 2:
            x = np.arange(self.bounds[0][0], self.bounds[0][1], delta)
            y = np.arange(self.bounds[1][0], self.bounds[1][1], delta)
            X, Y = np.meshgrid(x, y)
            Z = np.array([[self.f([X[i, j], Y[i, j]]) for j in range(X.shape[1])] for i in range(X.shape[0])])

            fig = plt.figure(figsize=(12, 5))
            fig.suptitle(f"{self.function_name} (2D)", fontsize=14)
            ax1 = fig.add_subplot(1, 2, 1, projection="3d")
            ax2 = fig.add_subplot(1, 2, 2)
            ax1.plot_surface(X, Y, Z, cmap="coolwarm")
            ax2.contour(X, Y, Z, cmap="coolwarm")
            ax1.set_title("Surface"); ax2.set_title("Contour")
            if showPlot:
                plt.show()

        else:
            print("Visualization not supported for dimension > 2.")

# Example usage
if __name__ == "__main__":

    # 1D test
    obj1d = ObjectiveFunction("quadratic_1d", dimension=1)
    obj1d.visualization()

    # 2D test
    obj2d = ObjectiveFunction("ackley", dimension=2)
    obj2d.visualization(delta=0.25)