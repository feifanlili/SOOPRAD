import random
import numpy as np
import matplotlib.pyplot as plt
from deap import base, creator, tools

class GA_Optimizer:
    DEFAULT_PARAMS = {
        "N_bits": 10,          # Number of bits for binary encoding
        "N_pop": 40,           # Population size
        "maxiter": 100,       # Max number of generations
        "crossoverPB": 0.5,    # Crossover probability
        "mutationPB": 0.2,     # Mutation probability
        "stagnation_limit": 50,# Stop if no improvement for X generations
        "epsilon": 1e-6        # Convergence threshold
    }

    def __init__(self, objective_func, bounds, params=None):
        # Store the objective function and parameter bounds
        self.objective_func = objective_func
        self.bounds = bounds
        self.num_variables = len(bounds)

        # Merge default parameters with user-defined ones
        self.params = {**self.DEFAULT_PARAMS, **(params or {})}
        vars(self).update(self.params)

        # Initialize DEAP framework
        self._setup_deap()

    def _setup_deap(self):
        """Initializes DEAP components (Individual, Population, Operators)."""
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMin)

        self.toolbox = base.Toolbox()
        self.toolbox.register("gene_generator", random.randint, 0, 1)
        self.toolbox.register("individual", tools.initRepeat, creator.Individual, 
                              self.toolbox.gene_generator, self.N_bits * self.num_variables)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)

        self.toolbox.register("evaluate", self._evaluate)
        self.toolbox.register("mate", tools.cxTwoPoint)
        self.toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
        self.toolbox.register("select", tools.selTournament, tournsize=3)

    def _binary_to_physical(self, binary_individual):
        """Converts a binary individual to physical values based on bounds."""
        physical_values = []
        for i in range(self.num_variables):
            binary_part = binary_individual[i * self.N_bits: (i + 1) * self.N_bits]
            decimal_value = int("".join(map(str, binary_part)), 2)
            var_min, var_max = self.bounds[i]
            physical_value = var_min + (decimal_value / (2**self.N_bits - 1)) * (var_max - var_min)
            physical_values.append(physical_value)
        return physical_values

    def _evaluate(self, individual):
        """Evaluation function for individuals."""
        x = self._binary_to_physical(individual)
        return self.objective_func(x),

    def initialize_population(self):
        """Creates the initial population."""
        self.pop = self.toolbox.population(n=self.N_pop)
        fitnesses = list(map(self.toolbox.evaluate, self.pop))
        for ind, fit in zip(self.pop, fitnesses):
            ind.fitness.values = fit

    def evolve(self, plot_results=True):
        """Runs the genetic algorithm to optimize the objective function."""
        self.initialize_population()
        best_fitness = -float("inf")
        stagnation_count = 0
        ##########################################################################
        ##########################################################################
        # Generate grid for visualization (only if plotting is enabled)
        if plot_results:
            x = np.arange(self.bounds[0][0]-1, self.bounds[0][1]+1)
            y = np.arange(self.bounds[1][0]-1, self.bounds[1][1]+1)
            xgrid, ygrid = np.meshgrid(x, y)
            xy = np.stack([xgrid, ygrid])
            zgrid = self.objective_func(xy)

            # Matplotlib style
            plt.style.use('bmh')
            plt.rcParams.update({"axes.facecolor": "white", "axes.grid": True})

            # Create subplots
            fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
            ax1 = fig.add_subplot(1, 2, 1, projection="3d")
            ax2 = fig.add_subplot(1, 2, 2)

        # Surface plot
        # ax1.plot_surface(xgrid, ygrid, zgrid, cmap="coolwarm")
        
        # # Contour plot
        # ax2.contour(xgrid, ygrid, zgrid, cmap="coolwarm")
        ##########################################################################
        ##########################################################################

        for g in range(self.maxiter):
            print(f"-- Generation {g} --")
            ##########################################################################
            ##########################################################################
            if plot_results:
                ax1.clear()
                ax1.plot_surface(xgrid, ygrid, zgrid, cmap="coolwarm")  # Re-plot the objective function surface
                ax1.set_title("Objective Function Surface")
                ax1.set_xlabel("X")
                ax1.set_ylabel("Y")
                ax1.set_zlabel("Z")

                ax2.clear()
                ax2.contour(xgrid, ygrid, zgrid, cmap="coolwarm")  # Re-plot the contour of the objective function
                ax2.set_title("Objective Function Contour")
                ax2.set_xlabel("X")
                ax2.set_ylabel("Y")

                pop_phy = np.array([self._binary_to_physical(ind) for ind in self.pop])
                # print(pop_phy)
                x_vals = pop_phy[:, 0]  # First column
                y_vals = pop_phy[:, 1]  # Second column
                z_vals = np.array([ind.fitness.values[0] for ind in self.pop])
                ax1.scatter(x_vals,y_vals,z_vals, label= "Generation %i" %g, s=100)
                ax2.scatter(x_vals,y_vals, label= "Generation %i" %g, s=100)
                ax1.legend()
                ax2.legend()
                
                plt.pause(0.2)
            # ax1.cla()
            # ax2.cla()
            ##########################################################################
            ##########################################################################
            # Selection and cloning
            offspring = list(map(self.toolbox.clone, self.toolbox.select(self.pop, len(self.pop))))

            # Crossover
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < self.crossoverPB:
                    self.toolbox.mate(child1, child2)
                    del child1.fitness.values, child2.fitness.values

            # Mutation
            for mutant in offspring:
                if random.random() < self.mutationPB:
                    self.toolbox.mutate(mutant)
                    del mutant.fitness.values

            # Evaluate new individuals
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(self.toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            # Replace old population
            self.pop[:] = offspring

            # Track best and worst fitness
            fitness_values = [ind.fitness.values[0] for ind in self.pop]
            current_best = max(fitness_values)
            print(f"  Best fitness: {current_best}, Worst fitness: {min(fitness_values)}")

            # Check for convergence
            if current_best <= best_fitness:
                stagnation_count += 1
            else:
                stagnation_count = 0

            if stagnation_count >= self.stagnation_limit or \
               abs(current_best - min(fitness_values)) < self.epsilon:
                print(f"Stopping early at generation {g} due to stagnation or convergence.")
                break

            best_fitness = current_best

        best_ind = tools.selBest(self.pop, 1)[0]
        best_ind_phy = self._binary_to_physical(best_ind)
        print(f"Best individual: {best_ind_phy}, Fitness: {best_ind.fitness.values}")
        
        if plot_results:
            ax1.clear()
            ax1.plot_surface(xgrid, ygrid, zgrid, cmap="coolwarm")  
            ax1.set_title("Objective Function Surface")
            ax1.set_xlabel("X")
            ax1.set_ylabel("Y")
            ax1.set_zlabel("Z")

            ax2.clear()
            ax2.contour(xgrid, ygrid, zgrid, cmap="coolwarm")  
            ax2.set_title("Objective Function Contour")
            ax2.set_xlabel("X")
            ax2.set_ylabel("Y")

            ax1.scatter(best_ind_phy[0], best_ind_phy[1], best_ind.fitness.values[0], label='Best Individual')
            ax2.scatter(best_ind_phy[0], best_ind_phy[1], label='Best Individual', marker='x', color='r')

            ax1.legend()
            ax2.legend()
            plt.show()

# Example usage
if __name__ == "__main__":
    def eggholder(x):
        return (-(x[1] + 47) * np.sin(np.sqrt(abs(x[0]/2 + (x[1] + 47))))) - x[0] * np.sin(np.sqrt(abs(x[0] - (x[1] + 47))))

    bounds = [(-512, 512), (-512, 512)]
    ga = GA_Optimizer(objective_func=eggholder, bounds=bounds)
    ga.evolve()
