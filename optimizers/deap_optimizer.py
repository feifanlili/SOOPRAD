import random
import numpy as np
import matplotlib.pyplot as plt
from deap import base, creator, tools
import json
###########################################################################
from optimizers.utils.test_2dfunctions import ObjectiveFunction
from optimizers.utils import formatting
from optimizers.utils.logger import OptimizerLogger

class GA_Optimizer:     
    """
    A Genetic Algorithm optimizer using binary encoding, based on the DEAP framework.
    
    Attributes:
        objective_func (callable): The objective function to minimize.
        bounds (list of tuples): Variable bounds as (min, max) pairs.
        params (dict): Algorithm parameters.
    """        
    DEFAULT_PARAMS = {
        "N_bits": 10,          # Number of bits for binary encoding
        "N_pop": 40,           # Population size
        "maxiter": 1000,       # Max number of generations
        "crossoverPB": 0.5,    # Crossover probability
        "mutationPB": 0.2,     # Mutation probability
        "stagnation_limit": 50,# Stop if no improvement for X generations
        "epsilon": 1e-6        # Convergence threshold
    }

    def __init__(self, objective_func, bounds, params=None):
        """
        Initializes the GA optimizer.
        
        Args:
            objective_func (callable): Objective function to be minimized.
            bounds (list): List of (min, max) tuples for each variable.
            params (dict, optional): Custom GA parameters to override defaults.
        """
        # Store the objective function and parameter bounds
        self.objective_func = objective_func
        self.bounds = bounds
        self.num_variables = len(bounds)

        # Merge default parameters with user-defined ones
        self.params = {**self.DEFAULT_PARAMS, **(params or {})}
        vars(self).update(self.params)

        # Initialize DEAP framework
        self._setup_deap()

        # Optimization history record
        self.history_plot = []  # Holds data for later plotting, recording physical
        self.history_log = [] # Holds data for logging, recording 
        self.logger = OptimizerLogger()


    def _setup_deap(self):
        """Initializes DEAP components (Individual, Population, Operators)."""
        # The creator module in DEAP is a convenient metaclass-based utility that allows users to define custom classes (usually for individuals and fitness values) in a single line of code.
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMin)

        self.toolbox = base.Toolbox()
        #########################################################################################
        # Individual & Population Toolbox Registeration 
        #########################################################################################
        # ------------------------------------
        # Step 1: Gene (a single binary value)
        # ------------------------------------
        self.toolbox.register("gene_generator", random.randint, 0, 1)
        # -------------------------------------------------------------------
        # Step 2: Individual (a list of genes, wrapped in creator.Individual)
        # -------------------------------------------------------------------
        self.toolbox.register("individual", tools.initRepeat,
                            creator.Individual,
                            self.toolbox.gene_generator,
                            self.N_bits * self.num_variables)
        # ------------------------------------------
        # Step 3: Population (a list of individuals)
        # ------------------------------------------
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual, self.N_pop)

        self.toolbox.register("evaluate", self._evaluate)
        # TODO: available to be reset by the user.
        self.toolbox.register("mate", tools.cxTwoPoint)
        self.toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
        self.toolbox.register("select", tools.selTournament, tournsize=3)

    def _binary_to_physical(self, binary_individual):
        """
        Converts a binary individual into real-valued variables using linear mapping.
        
        Args:
            binary_individual (list): Binary list representing encoded variables.
        
        Returns:
            list: Real-valued representation of the individual.
        """
        physical_values = []
        for i in range(self.num_variables):
            # extract i_th N_bits binary number for the i_th dimensional variable
            binary_part = binary_individual[i * self.N_bits: (i + 1) * self.N_bits]
            # transform it into decimal number
            decimal_value = int("".join(map(str, binary_part)), 2)
            var_min, var_max = self.bounds[i]
            # based on the bounds, linear mapping the decimal value to the physical space
            physical_value = var_min + (decimal_value / (2**self.N_bits - 1)) * (var_max - var_min)
            physical_values.append(physical_value)
        return physical_values

    def _evaluate(self, individual):
        """
        Evaluation wrapper that converts binary to physical values and evaluates the fitness.
        
        Args:
            individual (list): Binary individual.
        
        Returns:
            tuple: Fitness value (as a single-element tuple).
        """
        x = self._binary_to_physical(individual)
        return self.objective_func(x),

    def initialize_population(self):
        """
        Initializes the population and evaluates fitness values.
        """
        self.pop = self.toolbox.population()
        fitnesses = list(map(self.toolbox.evaluate, self.pop))
        for ind, fit in zip(self.pop, fitnesses):
            ind.fitness.values = fit

    def optimize(self):
        """
        Runs the genetic algorithm to optimize the objective function.

        This function:
            - Initializes the population
            - Iteratively applies selection, crossover, mutation, and evaluation
            - Tracks progress and logs the best solution
            - Implements early stopping based on stagnation or convergence
        """
        self.initialize_population()
        best_fitness = -float("inf")
        stagnation_count = 0

        print("\n" + "#" * 80)
        print("Starting Genetic Algorithm Optimization")
        print("#" * 80)

        for g in range(self.maxiter):
            # Get physical value of the population for the logging.
            pop_phy = np.array([self._binary_to_physical(ind) for ind in self.pop])
            fitness_vals = np.array([ind.fitness.values[0] for ind in self.pop])
            self.logger.log_population(g, pop_phy, fitness_vals)
            ##########################################################################
            # 1 - Selection
            ##########################################################################
            # self.toolbox.clone() == deepcopy()
            offspring = list(map(self.toolbox.clone, self.toolbox.select(self.pop, len(self.pop))))
            ##########################################################################
            # 2- Crossover (Recombination)
            ##########################################################################
            # mate individuals in order
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < self.crossoverPB:
                    self.toolbox.mate(child1, child2)
                    del child1.fitness.values, child2.fitness.values # if crossover happens, deleting the current fitness value, since the gene already changed
            ##########################################################################
            # 3 - Mutation
            ##########################################################################
            for mutant in offspring:
                if random.random() < self.mutationPB:
                    self.toolbox.mutate(mutant)
                    del mutant.fitness.values # if mutation happens, deleting the current fitness value
            ##########################################################################
            # 4 - Evaluation
            ##########################################################################
            # Evaluate new individuals
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid] # find the individuals without fitness value and re-evaluation
            fitnesses = map(self.toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit # assign the new evaluation
            ##########################################################################
            # 5 - Update
            ##########################################################################
            # Replace old population
            self.pop[:] = offspring

            # Track best and worst fitness
            fitness_values = [ind.fitness.values[0] for ind in self.pop]
            current_best = max(fitness_values)
            self.logger.log_generation_summary(
                generation=g,
                best=current_best,
                worst=min(fitness_values),
                avg=np.mean(fitness_values))
            
            # Check for convergence
            if current_best <= best_fitness:
                stagnation_count += 1
            else:
                stagnation_count = 0

            if stagnation_count >= self.stagnation_limit or \
               abs(current_best - min(fitness_values)) < self.epsilon:
                print(f"Early stopping at generation {g} (stagnation or convergence reached).")
                break

            best_fitness = current_best

        best_ind = tools.selBest(self.pop, 1)[0]
        best_ind_phy = self._binary_to_physical(best_ind)

        print(f"Best individual (decoded): {best_ind_phy}")
        print(f"Fitness: {best_ind.fitness.values[0]:.5f}")
        print("#" * 80)
        print("Optimization Complete")
        print("#" * 80)

        self.logger.save()

    def log(self):
        """
        Logs the optimization history to a JSON file named 'optimization_history.json'.

        Converts internal history_log to a JSON-serializable format before saving.
        Requires a utility function: utility.make_json_serializable()
        """
        serializable_history = formatting.make_json_serializable(self.history_log)
        with open("optimization_history.json", "w") as f:
            json.dump(serializable_history, f, indent=4)


    def replay_evolution(self, pause_time=0.3):
        """
        Replays the optimization process generation-by-generation using 2D and 3D plots.

        Args:
            pause_time (float): Pause between frames in seconds (default: 0.3)
        """
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

        for gen_data in self.history_plot:
            gen = gen_data["generation"]
            pop_phy = gen_data["individuals"]
            fitnesses = gen_data["fitnesses"]

            x_vals = pop_phy[:, 0]  # First column
            y_vals = pop_phy[:, 1]  # Second column
            z_vals = np.array(fitnesses)

            # Plot 3D surface
            ax1.clear()
            ax1.plot_surface(xgrid, ygrid, zgrid, cmap="coolwarm", alpha=0.7)
            ax1.scatter(x_vals, y_vals, z_vals, c="tab:red", s=50, label=f"Gen {gen}")
            ax1.set_title("Surface")
            ax1.set_xlabel("X"); ax1.set_ylabel("Y"); ax1.set_zlabel("Z")
            ax1.legend()

            # Plot 2D contour
            ax2.clear()
            ax2.contour(xgrid, ygrid, zgrid, cmap="coolwarm")
            ax2.scatter(x_vals, y_vals, c="tab:red", s=50, label=f"Gen {gen}")
            ax2.set_title("Contour")
            ax2.set_xlabel("X"); ax2.set_ylabel("Y")
            ax2.legend()

            plt.pause(pause_time)

        best_ind = tools.selBest(self.pop, 1)[0]
        best_ind_phy = self._binary_to_physical(best_ind)

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
    sphere_func = ObjectiveFunction("sphere")
    sine_squared_func = ObjectiveFunction("sine_squared")
    absolute_sum_func = ObjectiveFunction("absolute_sum")
    quadratic_func = ObjectiveFunction("quadratic")
    ackley_func = ObjectiveFunction("ackley")
    levy_func = ObjectiveFunction("levy")
    eggholder_func = ObjectiveFunction("eggholder")

    bounds = [(-512, 512), (-512, 512)]
    # bounds = [(-20, 20), (-20, 20)]
    ga = GA_Optimizer(objective_func=eggholder_func.f, bounds=bounds)
    ga.optimize()
    # ga.log()
    # ga.replay_evolution()
