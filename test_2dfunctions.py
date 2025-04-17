# ++++++++++ LOCAL IMPORTS ++++++++++++++
# ++++++++++ PACKAGE IMPORTS ++++++++++++
import numpy as np
import matplotlib.pyplot as plt

# Define individual objective functions


def sphere_function(x):
    return x[0] ** 2.0 + x[1] ** 2.0


def sine_squared_function(x):
    return (
        np.sin(x[0]) ** 2.0 + np.sin(x[1]) ** 2.0 + 0.005 * (x[0] ** 2.0 + x[1] ** 2.0)
    )


def absolute_sum_function(x):
    return 50.0 * abs(x[0] + x[1]) + x[0] ** 2.0


def quadratic_function(x):
    return x[0] ** 2.0 + 50.0 * x[1] ** 2.0


def ackley_function(x):
    X = np.asarray(x)
    a = 20.0
    b = 0.2
    c = 2 * np.pi
    s = (
        -a * np.exp(-b * np.sqrt(1.0 / 2 * np.sum(np.square(X), 0)))
        - np.exp(1.0 / 2 * np.sum(np.cos(c * X), 0))
        + a
        + np.exp(1)
    )
    return s


def levy_function(x):
    wx = (x[0] + 3.0) / 4.0
    wy = (x[1] + 3.0) / 4.0
    return (
        np.sin(np.pi * wx) ** 2
        + (wy - 1) ** 2 * (1 + np.sin(2 * np.pi * wy) ** 2)
        + (wx - 1) ** 2 * (1 + 10 * np.sin(np.pi * wx) ** 2)
    )


def eggholder_function(x):
    return -(x[1] + 47) * np.sin(np.sqrt(abs(x[0] / 2 + (x[1] + 47)))) - x[0] * np.sin(
        np.sqrt(abs(x[0] - (x[1] + 47)))
    )


# Mapping function names to functions
OBJECTIVE_FUNCTIONS = {
    "sphere": sphere_function,
    "sine_squared": sine_squared_function,
    "absolute_sum": absolute_sum_function,
    "quadratic": quadratic_function,
    "ackley": ackley_function,
    "levy": levy_function,
    "eggholder": eggholder_function,
}


class ObjectiveFunction:
    def __init__(self, function_name, dimension=2):
        self.dimension = dimension
        self.function_name = function_name
        self.function = OBJECTIVE_FUNCTIONS.get(function_name, None)
        if self.function is None:
            print("Not valid function name!")

    def f(self, x):
        if self.function:
            return self.function(x)
        return 0  # Return a default value if function is invalid

    def visualization(self, bounds: list, delta=0.5, showPlot=True):
        # Generate sample data
        x = np.arange(bounds[0][0], bounds[0][1], delta)
        y = np.arange(bounds[1][0], bounds[1][1], delta)
        X, Y = np.meshgrid(x, y)
        Z = np.array(
            [
                [self.f([X[i, j], Y[i, j]]) for j in range(X.shape[1])]
                for i in range(X.shape[0])
            ]
        )

        # Create plots
        plt.style.use("bmh")
        plt.rcParams.update({"axes.facecolor": "white", "axes.grid": True})
        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
        fig.suptitle(f"{self.function_name} function", fontsize=14)
        ax1 = fig.add_subplot(1, 2, 1, projection="3d")
        ax2 = fig.add_subplot(1, 2, 2)
        ax1.plot_surface(X, Y, Z, cmap="coolwarm")
        ax2.contour(X, Y, Z, cmap="coolwarm")
        if showPlot:
            plt.show()
