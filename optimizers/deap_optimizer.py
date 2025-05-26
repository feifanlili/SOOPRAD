import random
import numpy as np
import matplotlib.pyplot as plt
from deap import base, creator, tools
import inspect
import json
###########################################################################
from optimizers.utils import formatting
from optimizers.utils.logger import OptimizerLogger
from optimizers.utils.benckmark_functions import ObjectiveFunction

class GA_Optimizer:     
    """
    A Genetic Algorithm optimizer using binary encoding, based on the DEAP framework.

    Args:
        objective_func (callable): The objective function to minimize.
        bounds (list of tuples): Variable bounds for each variable.
        params (dict, optional): Custom parameters to override defaults.
        log_population (bool, optional): If True, logs full population phenotype+fitness. Defaults to True.
        log_summary (bool, optional): If True, logs generation-level best/worst/avg. Defaults to True.
    """       
    DEFAULT_PARAMS = {
    "N_bits": 10,              # Number of bits for binary encoding
    "N_pop": 40,               # Population size
    "maxiter": 1000,           # Max number of generations
    "crossoverPB": 0.5,        # Crossover probability
    "mutationPB": 0.2,         # Mutation probability
    "stagnation_limit": 50,    # Stop if no improvement for X generations
    "epsilon": 1e-6,           # Convergence threshold

    # Default DEAP operators (can be overridden by user)
    "operators": {
        "mate": ("cxTwoPoint", {}),
        "mutate": ("mutFlipBit", {"indpb": 0.05}),
        "select": ("selTournament", {"tournsize": 3}),
    }
    }

    def __init__(self, objective_func, bounds, params=None, log_population=True, log_summary=True):
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
        #########################################################################################
        # Parameter Handling:
        #########################################################################################
        # 1. Ensure input params is a dict
        params = params or {} 
        # 2. Handle the core parameters
        core_params = {**self.DEFAULT_PARAMS, **{k: v for k, v in params.items() if k != "operators"}}
        self.params = core_params
        vars(self).update(self.params) # Sets each param in self as an attribute for easy access
        # 3. Handle the operator configuration
        # Store operator config names and kwargs separately
        self.operator_names = {}
        self.operator_kwargs = {}
        user_operator_config = params.get("operators", {})
        operator_params = {**self.DEFAULT_PARAMS["operators"], **user_operator_config}
        #########################################################################################
        # Optional Logger Setup:
        #########################################################################################
        if log_population or log_summary:
            self.logger = OptimizerLogger(enable_population=log_population,enable_summary=log_summary)
        else:
            self.logger = None
        #########################################################################################
        # Initialize DEAP Framework
        #########################################################################################
        self._setup_deap()
        self.set_operators(**operator_params)


    def _setup_deap(self):
        """Initializes DEAP components (Individual, Population, Operators)."""
        # Safely create DEAP classes (avoid redefinition)
        if not hasattr(creator, "FitnessMin"):
            creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        if not hasattr(creator, "Individual"):
            creator.create("Individual", list, fitness=creator.FitnessMin)

        self.toolbox = base.Toolbox()
        # ------------------------------------
        # Gene: a single binary value (0 or 1)
        # ------------------------------------
        self.toolbox.register("gene_generator", random.randint, 0, 1)
        # ---------------------------------------------
        # Individual: list of genes wrapped in creator
        # ---------------------------------------------
        self.toolbox.register(
            "individual", tools.initRepeat,
            creator.Individual,
            self.toolbox.gene_generator,
            self.N_bits * self.num_variables
        )
        # -----------------------
        # Population: list of individuals
        # -----------------------
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual, self.N_pop)
        # -----------------------
        # Evaluation: pass-through to user's objective
        # -----------------------
        self.toolbox.register("evaluate", self._evaluate)
        # -------------------------------
        # Operators (mate, mutate, select)
        # -------------------------------
        # Only registered later via self.set_operators()

    def set_operators(self, mate=None, mutate=None, select=None):
        """
        Set genetic operators dynamically.

        Args:
            mate (tuple): (str, dict) e.g., ("cxUniform", {"indpb": 0.5})
            mutate (tuple): (str, dict) e.g., ("mutFlipBit", {"indpb": 0.05})
            select (tuple): (str, dict) e.g., ("selTournament", {"tournsize": 3})
        """
        op_config = {"mate": mate, "mutate": mutate, "select": select}

        for op_name, config in op_config.items():
            if config:
                func_name, kwargs = config
                deap_func = getattr(tools, func_name, None)
                if deap_func is None:
                    raise ValueError(f"Unknown DEAP function: {func_name}")
                self.operator_names[op_name] = func_name
                self.operator_kwargs[op_name] = kwargs
                self.toolbox.register(op_name, deap_func, **kwargs)
    
    def list_available_operators(self, category=None):
        """
        Prints available DEAP operators in ('operator_name', {}) format.

        Args:
            category (str, optional): One of "select", "mate", "mutate". If None, shows all.
        """

        op_map = {
            "select": "sel",
            "mate": "cx",
            "mutate": "mut"
        }

        valid_prefixes = (op_map[category],) if category else ("sel", "cx", "mut")

        print(f"\nAvailable DEAP Operators (format: ('name', {{}})) for: {category or 'all'}")
        print("-" * 60)

        for name, func in sorted(vars(tools).items()):
            if inspect.isfunction(func) and name.startswith(valid_prefixes):
                print(f"('{name}', {{}})")


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
            if self.logger:
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
            if self.logger:
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

        if self.logger:
            self.logger.save()


# Example usage
if __name__ == "__main__":
    # === 2D Objective Functions ===
    sphere_func = ObjectiveFunction("sphere", dimension=2)
    sine_squared_func = ObjectiveFunction("sine_squared", dimension=2)
    absolute_sum_func = ObjectiveFunction("absolute_sum", dimension=2)
    quadratic_func = ObjectiveFunction("quadratic", dimension=2)
    ackley_func = ObjectiveFunction("ackley", dimension=2)
    levy_func = ObjectiveFunction("levy", dimension=2)
    eggholder_func = ObjectiveFunction("eggholder", dimension=2)

    bounds = [(-512, 512), (-512, 512)]
    # bounds = [(-20, 20), (-20, 20)]
    ga = GA_Optimizer(objective_func=eggholder_func.f, bounds=bounds)
    ga.optimize()
