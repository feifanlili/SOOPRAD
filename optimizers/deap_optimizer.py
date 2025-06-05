import random
import numpy as np
from deap import base, creator, tools, algorithms
import array
import inspect
from abc import ABC, abstractmethod
###########################################################################
from .utils.logger import OptimizerLogger
from .utils.benckmark_functions import ObjectiveFunction

class DeapOptimizer(ABC):
    def __init__(self, objective_func, bounds, default_params, params=None, log_population=True, log_summary=True):
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
        """Initializes DEAP components (Individual, Population, Operators, Evaluation)."""
        self.toolbox = base.Toolbox()
        self._register_ind_and_pop()
        self._register_operators()
        self.toolbox.register("evaluate", self._evaluate) 
    
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

    @abstractmethod
    def _register_ind_and_pop(self):
        """This method aim at registering individual & population generation function of DEAP."""
        pass
    
    @abstractmethod
    def _register_operators(self):
        """This method aim at registering evolutionary operators function of DEAP."""
        pass

    def _evaluate(self, individual):
        """Default evaluation assuming individual is already in real-valued form."""
        return self.objective_func(individual),

    @abstractmethod
    def optimize(self):
        pass
    

class GA_Optimizer(DeapOptimizer):
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
        super().__init__(objective_func, bounds, self.DEFAULT_PARAMS, params,
                         log_population, log_summary)
        self._setup_deap()
    
    def _register_ind_and_pop(self):
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
        for name, (op_name, kwargs) in self.params['operators'].items():
            # Get the actual DEAP function by name
            op_func = getattr(tools, op_name)
            # Register it with the toolbox
            self.toolbox.register(name, op_func, **kwargs)

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
        Decode binary individual and evaluate.
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

        for g in range(ngen):
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
    DEFAULT_PARAMS = {         # Number of genes per individual
    "MU": 10,                # MU: Number of parents (survivors)
    "LAMBDA": 100,         # LAMBDA: Number of offspring per generation
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
        super().__init__(objective_func, bounds, self.DEFAULT_PARAMS, params,
                         log_population, log_summary)
        self._setup_deap()

    def _register_ind_and_pop(self):
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
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
    
    def _register_operators(self):
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
        return super()._evaluate(individual)