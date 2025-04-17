# ++++++++++ LOCAL IMPORTS ++++++++++++

# ++++++++++ PACKAGE IMPORTS ++++++++++
from abc import ABCMeta, abstractmethod


class Optimizer(ABCMeta):
    def __init__(self, objective_func, bounds):
        self.objective_func = objective_func
        self.bounds = bounds
        self.results = None  # Store optimization results

    @abstractmethod
    def optimize(self):
        pass

    def visualize(self):
        # Maybe here the result_visualization function of scipy_optimizer can be moved to
        pass
