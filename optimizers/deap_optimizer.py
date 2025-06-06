import random
import numpy as np
from deap import base, creator, tools, algorithms
import array
import inspect
from abc import ABC, abstractmethod
###########################################################################
from .utils.logger import OptimizerLogger

class DeapOptimizer(ABC):
    """
    Abstract base class for evolutionary optimization using the DEAP framework.

    This class provides the foundational setup for optimization algorithms
    (e.g., Genetic Algorithm, Evolution Strategies) using DEAP. It manages
    the DEAP toolbox, logging, and operator registration, and exposes methods
    for dynamically inspecting and resetting evolutionary operators.

    Subclasses must implement the individual/population initialization and
    the evolutionary loop.
    """
    def __init__(self, objective_func, bounds, default_params, params=None, log_population=True, log_summary=True):
        """
        Initialize the optimizer.

        Args:
            objective_func (callable): The fitness/objective function to be minimized.
            bounds (list of tuple): Bounds for each variable (min, max) pairs.
            default_params (dict): Default operator configuration and other parameters.
            params (dict, optional): Custom user-provided parameters to override defaults.
            log_population (bool): Whether to log the population at each generation.
            log_summary (bool): Whether to log optimization summary statistics.
        """
        # Store the objective function and parameter bounds
        self.objective_func = objective_func
        self.bounds = bounds
        self.num_variables = len(bounds)
        # Parameter handling
        self.params = {**default_params, **(params or {})}
        # Optional logger setup
        if log_population or log_summary:
            self.logger = OptimizerLogger(
                enable_population=log_population,
                enable_summary=log_summary
            )
        else:
            self.logger = None
        # Initialize DEAP framework
        self._setup_deap()
    
    def _setup_deap(self):
        """Initializes DEAP toolbox and registers individuals, population, operators, and evaluation function."""
        self.toolbox = base.Toolbox()
        self._register_ind_and_pop()
        self._register_operators()
        self.toolbox.register("evaluate", self._evaluate) 

    @abstractmethod
    def _register_ind_and_pop(self):
        """
        Abstract method to register DEAP individual and population creation routines.
        
        Must be implemented by subclasses to define the encoding and structure
        of individuals and how populations are initialized.
        """
        pass
    
    @abstractmethod
    def _register_operators(self):
        """
        Abstract method to register DEAP operators (selection, crossover, mutation).
        
        Subclasses should register functions using `self.toolbox.register(...)`
        and handle any decorations or constraints.
        """
        pass

    def _evaluate(self, individual):
        """
        Evaluates an individual using the provided objective function.

        Args:
            individual (list or array): A candidate solution.

        Returns:
            tuple: A single-element tuple with the fitness value.
        """
        return self.objective_func(individual),

    @abstractmethod
    def optimize(self):
        """
        Runs the optimization process.

        Subclasses should implement the evolutionary algorithm logic
        using DEAP's algorithms or custom procedures.
        """
        pass
    
    ##################################################################################
    # Operators Customization
    ##################################################################################
    def list_registered_operators(self):
        """
        Prints the currently registered DEAP operators and their configuration.
        Useful for debugging or verification of operator setup.
        """

        print("\n" + "=" * 40)
        print(" Current DEAP Operator Configuration")
        print("=" * 40)

        for op in ['select', 'mate', 'mutate']:
            if op in self.params['operators']:
                func_name, kwargs = self.params['operators'][op]
                print(f"{op.capitalize()} Operator:")
                print(f"  Function: {func_name}  (from deap.tools)")
                print("  Parameters:")
                if kwargs:
                    for k, v in kwargs.items():
                        print(f"    {k} = {v}")
                else:
                    print("    (no parameters)")
            else:
                print(f"{op.capitalize()} Operator: Not configured")

        print("=" * 40 + "\n")
    
    def list_available_operators(self, category=None):
        """
        Lists available DEAP operators from `deap.tools`.

        Args:
            category (str, optional): Filter operators by category. One of "select", "mate", "mutate".
                                      If None, shows all available operators.
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

    def reset_select(self, func_name, **kwargs):
        """
        Resets the selection operator with a new one from `deap.tools`.

        Args:
            func_name (str): Name of the DEAP selection operator (e.g., 'selTournament').
            kwargs: Keyword arguments passed to the DEAP operator.
        """
        if hasattr(tools, func_name):
            self.toolbox.unregister("select")
            self.toolbox.register("select", getattr(tools, func_name), **kwargs)
            self.params['operators']['select'] = (func_name, kwargs)
        else:
            raise ValueError(f"Selection function '{func_name}' not found in deap.tools")

    def reset_mate(self, func_name, **kwargs):
        """
        Resets the mating operator.

        Args:
            func_name (str): Name of the DEAP crossover operator (e.g., 'cxTwoPoint').
            kwargs: Parameters for the crossover function.
        """
        if hasattr(tools, func_name):
            self.toolbox.unregister("mate")
            self.toolbox.register("mate", getattr(tools, func_name), **kwargs)
            self.params['operators']['mate'] = (func_name, kwargs)
        else:
            raise ValueError(f"Mate function '{func_name}' not found in deap.tools")

    def reset_mutate(self, func_name, **kwargs):
        """
        Resets the mutation operator.

        Args:
            func_name (str): Name of the DEAP mutation operator (e.g., 'mutFlipBit').
            kwargs: Parameters for the mutation function.
        """
        if hasattr(tools, func_name):
            self.toolbox.unregister("mutate")
            self.toolbox.register("mutate", getattr(tools, func_name), **kwargs)
            self.params['operators']['mutate'] = (func_name, kwargs)
        else:
            raise ValueError(f"Mutation function '{func_name}' not found in deap.tools")


class GA_Optimizer(DeapOptimizer):
    """
    Genetic Algorithm (GA) optimizer using binary encoding with the DEAP framework.

    This optimizer encodes each variable as a binary string and applies
    standard GA operators (selection, crossover, mutation) to evolve a population
    toward optimal solutions. It supports logging, early stopping via stagnation
    or convergence thresholds, and Hall of Fame tracking.

    Attributes:
        DEFAULT_PARAMS (dict): Default configuration for the GA, including operator
            setup, probabilities, and convergence parameters.
    """
    DEFAULT_PARAMS = {
    "N_bits": 10,              # Number of bits for binary encoding
    "N_pop": 50,               # Population size
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
    def __init__(self, objective_func, bounds, params=None,
                 log_population=True, log_summary=True):
        """
        Initializes the GA optimizer with a binary representation.

        Args:
            objective_func (callable): The objective function to minimize.
            bounds (list of tuple): Variable bounds as (min, max) pairs.
            params (dict, optional): Custom configuration parameters.
            log_population (bool): Whether to log population each generation.
            log_summary (bool): Whether to log summary statistics.
        """
        super().__init__(objective_func, bounds, self.DEFAULT_PARAMS, params,
                         log_population, log_summary)
        self._setup_deap()
    
    def _register_ind_and_pop(self):
        """
        Registers DEAP individual and population initialization using binary encoding.
        Each variable is represented by `N_bits` binary digits.
        """
        # Safely create DEAP classes (avoid redefinition)
        if not hasattr(creator, "FitnessMin"):
            creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        if not hasattr(creator, "Individual"):
            creator.create("Individual", list, fitness=creator.FitnessMin)
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
            self.params['N_bits'] * self.num_variables
        )
        # ---------------------------------
        # Population: list of individuals
        # ---------------------------------
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual, self.params['N_pop'])
    
    def _register_operators(self):
        """
        Registers evolutionary operators (selection, crossover, mutation) based
        on the configuration in self.params['operators'].
        """
        for name, (op_name, kwargs) in self.params['operators'].items():
            # Get the actual DEAP function by name
            op_func = getattr(tools, op_name)
            # Register it with the toolbox
            self.toolbox.register(name, op_func, **kwargs)

    def _binary_to_physical(self, binary_individual):
        """
        Converts a binary-encoded individual into its corresponding real-valued representation.

        Args:
            binary_individual (list): Flat list of 0s and 1s encoding all variables.

        Returns:
            list: Real-valued vector corresponding to the decoded individual.
        """
        physical_values = []
        for i in range(self.num_variables):
            # extract i_th N_bits binary number for the i_th dimensional variable
            binary_part = binary_individual[i * self.params['N_bits']: (i + 1) * self.params['N_bits']]
            # transform it into decimal number
            decimal_value = int("".join(map(str, binary_part)), 2)
            var_min, var_max = self.bounds[i]
            # based on the bounds, linear mapping the decimal value to the physical space
            physical_value = var_min + (decimal_value / (2**self.params['N_bits'] - 1)) * (var_max - var_min)
            physical_values.append(physical_value)
        return physical_values

    def _evaluate(self, individual):
        """
        Evaluates a binary individual by decoding to real-valued representation
        and passing it to the objective function.

        Args:
            individual (list): Binary-encoded DEAP individual.

        Returns:
            tuple: Single-element tuple with the fitness value.
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
        Runs the main genetic algorithm loop.

        Evolves a population using selection, crossover, and mutation, while tracking
        performance and supporting early stopping based on stagnation or convergence.
        Logs population and statistics if logging is enabled.
        """
        # Parameter setup
        npop = self.params['N_pop']
        cxpb = self.params['crossoverPB']
        mutpb = self.params['mutationPB']
        ngen = self.params['maxiter']

        # Initialize population, best fitness (for convergence check), and stagnation count
        self.initialize_population()
        best_fitness = -float("inf")
        stagnation_count = 0

        # Setup Hall of Fame
        self.hall_of_fame = tools.HallOfFame(1)
        self.hall_of_fame.update(self.pop)
        
        print("\n" + "#" * 80)
        print("Starting Genetic Algorithm Optimization")
        print("#" * 80)

        for g in range(1, ngen + 1):
            # Get physical value of the population for the logging.
            pop_phy = np.array([self._binary_to_physical(ind) for ind in self.pop])
            fitness_vals = np.array([ind.fitness.values[0] for ind in self.pop])
            if self.logger:
                self.logger.log_population(g, pop_phy, fitness_vals)
            # ------------------------------------
            # 1. Selection
            # ------------------------------------
            # self.toolbox.clone() == deepcopy()
            selected_individuals = list(map(self.toolbox.clone, self.toolbox.select(self.pop, len(self.pop))))
            # ------------------------------------
            # 2. Recombination & Mutation
            # ------------------------------------
            offspring = algorithms.varOr(selected_individuals, self.toolbox, npop, cxpb, mutpb)
            # ------------------------------------
            # 3. Evaluation
            # ------------------------------------
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = self.toolbox.map(self.toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
            # Updates
            self.pop[:] = offspring
            # Record stats
            record = self.logger.stats.compile(self.pop)
            self.logger.logbook.record(gen=g, nevals=len(invalid_ind), **record)
            # Update Hall of Fame
            if self.hall_of_fame is not None:
                self.hall_of_fame.update(offspring)
            # Track best and worst fitness
            fitness_values = [ind.fitness.values[0] for ind in self.pop]
            current_best = max(fitness_values)
            current_worst = min(fitness_values)
            # Check for convergence
            if current_best <= best_fitness:
                stagnation_count += 1
            else:
                stagnation_count = 0
            if stagnation_count >= self.params['stagnation_limit'] or \
               abs(current_best - current_worst) < self.params['epsilon']:
                print(f"Early stopping at generation {g} (stagnation or convergence reached).")
                break

            best_fitness = current_best
        
        best_individual = self.hall_of_fame[0]
        best_individual_phy = self._binary_to_physical(self.hall_of_fame[0])

        print(f"Best individual (decoded): {best_individual_phy}")
        print(f"Fitness: {best_individual.fitness.values[0]:.5f}")
        print("#" * 80)
        print("Optimization Complete")
        print("#" * 80)
        
        if self.logger:
            self.logger.save()


class ES_Optimizer(DeapOptimizer):
    """
    Evolution Strategy (ES) optimizer based on the DEAP framework.

    This class implements a (μ + λ) or (μ, λ) Evolution Strategy (ES) algorithm for
    real-valued optimization problems. Each individual consists of real-valued genes and
    associated strategy parameters (standard deviations for mutation).

    Attributes:
        DEFAULT_PARAMS (dict): Default configuration parameters including population size,
            offspring count, mutation/crossover probabilities, strategy bounds, convergence criteria,
            and DEAP operator settings.
    """
    DEFAULT_PARAMS = {         # Number of genes per individual
    "MU": 10,                # MU: Number of parents (survivors)
    "LAMBDA": 500,         # LAMBDA: Number of offspring per generation
    "maxiter": 500,             # Number of generations (ngen)
    "crossoverPB": 0.6,         # Probability of applying crossover
    "mutationPB": 0.3,          # Probability of applying mutation
    "MIN_STRATEGY": 0.001,      # Minimum strategy value for mutation strength
    "MAX_STRATEGY": 1.0,        # Maximum initial strategy value
    "stagnation_limit": 50,    # Stop if no improvement for X generations
    "epsilon": 1e-6,           # Convergence threshold

    # DEAP operators and parameters
    "operators": {
        "mate": ("cxESBlend", {"alpha": 0.1}),
        "mutate": ("mutESLogNormal", {"c": 1.0, "indpb": 0.03}),
        "select": ("selTournament", {"tournsize": 3})
    }
    }
    def __init__(self, objective_func, bounds, params=None,
                 log_population=True, log_summary=True):
        """
        Initializes the ES optimizer.

        Args:
            objective_func (callable): The objective function to minimize.
            bounds (list of tuple): Bounds for each decision variable [(min, max), ...].
            params (dict, optional): User-defined hyperparameters to override DEFAULT_PARAMS.
            log_population (bool): Whether to log the population every generation.
            log_summary (bool): Whether to log a summary of each generation.
        """
        
        super().__init__(objective_func, bounds, self.DEFAULT_PARAMS, params,
                         log_population, log_summary)
        self._setup_deap()

    def _register_ind_and_pop(self):
        """
        Registers DEAP individual and population generation functions.

        Each individual is composed of:
            - A list of real-valued decision variables.
            - An associated list of strategy parameters (mutation strengths).
        """
        # Safely create DEAP classes (avoid redefinition)
        if not hasattr(creator, "FitnessMin"):
            creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        if not hasattr(creator, "Individual"):
            creator.create("Individual", array.array, typecode="d", fitness=creator.FitnessMin, strategy=None)
        if not hasattr(creator, "Strategy"):
            creator.create("Strategy", array.array, typecode="d")
        # Individual generator
        def generateES(icls, scls, size, bounds, smin, smax):
            ind = icls(random.uniform(lower, upper) for lower, upper in bounds)
            ind.strategy = scls(random.uniform(smin, smax) for _ in range(size))
            return ind
        # ----------------------------------------------------------------------
        # Individual: list of variables with attributes of Fitness and Strategy
        # ----------------------------------------------------------------------
        self.toolbox.register("individual", generateES, creator.Individual, creator.Strategy, self.num_variables, self.bounds, self.params['MIN_STRATEGY'], self.params['MAX_STRATEGY'])
        # ---------------------------------
        # Population: list of individuals
        # ---------------------------------
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual, self.params['MU'])
    
    def _register_operators(self):
        """
        Registers and decorates genetic operators (mate, mutate, select) in the DEAP toolbox.

        Ensures that mutation strategies are clamped to a minimum value after application.
        """
        for name, (op_name, kwargs) in self.params['operators'].items():
            # Get the actual DEAP function by name
            op_func = getattr(tools, op_name)
            # Register it with the toolbox
            self.toolbox.register(name, op_func, **kwargs)
        # Operator decorator
        def checkStrategy(minstrategy):
            def decorator(func):
                def wrappper(*args, **kargs):
                    children = func(*args, **kargs)
                    for child in children:
                        for i, s in enumerate(child.strategy):
                            if s < minstrategy:
                                child.strategy[i] = minstrategy
                    return children
                return wrappper
            return decorator
        self.toolbox.decorate("mate", checkStrategy(self.params['MIN_STRATEGY']))
        self.toolbox.decorate("mutate", checkStrategy(self.params['MIN_STRATEGY']))

    def _evaluate(self, individual):
        """
        Evaluates the fitness of a real-valued individual using the objective function.

        Args:
            individual (array): A DEAP individual.

        Returns:
            tuple: A single-element tuple containing the fitness value.
        """
        return super()._evaluate(individual)

    def initialize_population(self):
        """
        Initializes the population and evaluates fitness values.
        """
        self.pop = self.toolbox.population()
        fitnesses = list(map(self.toolbox.evaluate, self.pop))
        for ind, fit in zip(self.pop, fitnesses):
            ind.fitness.values = fit

    def optimize(self, strategy="plus"):
        """
        Runs the ES optimization loop using either (μ + λ) or (μ, λ) survivor selection.

        Args:
            strategy (str): Selection strategy; must be either "plus" or "comma".
        
        Raises:
            AssertionError: If `LAMBDA < MU` or strategy is not one of the allowed options.

        Process:
            1. Generates λ offspring using mutation and recombination.
            2. Evaluates offspring fitness.
            3. Selects μ survivors using the given strategy.
            4. Tracks stagnation and convergence criteria.
            5. Terminates early if no progress or fitness spread is small.

        Logs:
            - Full population per generation (if enabled).
            - Summary statistics (min, max, avg, std).
            - Best solution found.
        """
        assert strategy in ("plus", "comma"), "Strategy must be either 'plus' or 'comma'."
        assert self.params['LAMBDA'] >= self.params['MU'], \
            "LAMBDA must be greater than or equal to MU (according to lecture material, LAMBDA should be remarkably higher than MU)."
        
        # Parameter setup
        mu = self.params['MU']
        lambda_ = self.params['LAMBDA']
        cxpb = self.params['crossoverPB']
        mutpb = self.params['mutationPB']
        ngen = self.params['maxiter']

        # Initialize population, best fitness (for convergence check), and stagnation count
        self.initialize_population()
        best_fitness = -float("inf")
        stagnation_count = 0

        # Setup Hall of Fame
        self.hall_of_fame = tools.HallOfFame(1)
        self.hall_of_fame.update(self.pop)
        
        print("\n" + "#" * 80)
        print("Starting Evolution Strategy Optimization")
        print("#" * 80)

        # Begin optimization loop
        for g in range(1, ngen + 1):
            # Save the population data to the logger
            fitness_vals = np.array([ind.fitness.values[0] for ind in self.pop])
            if self.logger:
                self.logger.log_population(g, self.pop, fitness_vals)
            # -----------------------------------
            # 1. Recombination & Mutation
            # -----------------------------------
            # Generate offspring from current population, applying mutate and mate to the population
            offspring = algorithms.varOr(self.pop, self.toolbox, lambda_, cxpb, mutpb)
            # -----------------------------------
            # 2. Evaluation
            # -----------------------------------
            # Evaluate offspring
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = self.toolbox.map(self.toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
            # -----------------------------------
            # 3. Selection
            # -----------------------------------
            # Survivor selection: either (μ + λ) or (μ, λ)
            if strategy == "plus":
                self.pop[:] = self.toolbox.select(self.pop + offspring, mu)
            else:  # comma
                self.pop[:] = self.toolbox.select(offspring, mu)
            # Record stats
            record = self.logger.stats.compile(self.pop)
            self.logger.logbook.record(gen=g, nevals=len(invalid_ind), **record)
            # Update Hall of Fame
            if self.hall_of_fame is not None:
                self.hall_of_fame.update(self.pop)
            # Track best and worst fitness
            fitness_values = [ind.fitness.values[0] for ind in self.pop]
            current_best = max(fitness_values)
            current_worst = min(fitness_values)
            # Check for convergence
            if current_best <= best_fitness:
                stagnation_count += 1
            else:
                stagnation_count = 0
            if stagnation_count >= self.params['stagnation_limit'] or \
               abs(current_best - current_worst) < self.params['epsilon']:
                print(f"Early stopping at generation {g} (stagnation or convergence reached).")
                break

            best_fitness = current_best
        
        best_individual = self.hall_of_fame[0]
        best_individual_phy = self.hall_of_fame[0].fitness.values[0]

        print(f"Best individual (decoded): {best_individual_phy}")
        print(f"Fitness: {best_individual.fitness.values[0]:.5f}")
        print("#" * 80)
        print("Optimization Complete")
        print("#" * 80)

        if self.logger:
            self.logger.save()