import random
import numpy as np
import array
from deap import base, creator, tools, algorithms
###########################################################################
from .utils.logger import OptimizerLogger
from .utils.benckmark_functions import ObjectiveFunction


class ES_Optimizer():
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

    def __init__(self, objective_func, bounds, params=None, log_population=True, log_summary=True):
        #################################################################
        # Store the objective function and parameter bounds
        #################################################################
        self.objective_func = objective_func
        self.bounds = bounds
        self.num_variables = len(bounds)
        #################################################################
        # Parameter Handling
        #################################################################
        params = params or {} 
        self.params = {**self.DEFAULT_PARAMS, **params}
        #########################################################################################
        # Optional Logger Setup:
        #########################################################################################
        if log_population or log_summary:
            self.logger = OptimizerLogger(enable_population=log_population,enable_summary=log_summary)
        else:
            self.logger = None

        self._setup_deap()

    def _setup_deap(self):
        """Initializes DEAP components (Individual, Population, Operators)."""
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", array.array, typecode="d", fitness=creator.FitnessMin, strategy=None)
        creator.create("Strategy", array.array, typecode="d")

        # Individual generator
        def generateES(icls, scls, size, bounds, smin, smax):
            ind = icls(random.uniform(lower, upper) for lower, upper in bounds)
            ind.strategy = scls(random.uniform(smin, smax) for _ in range(size))
            return ind
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
        
        self.toolbox = base.Toolbox()
        # ----------------------------------------------------------------------
        # Individual: list of variables with attributes of Fitness and Strategy
        # ----------------------------------------------------------------------
        self.toolbox.register("individual", generateES, creator.Individual, creator.Strategy, self.num_variables, self.bounds, self.params['MIN_STRATEGY'], self.params['MAX_STRATEGY'])
        # ---------------------------------
        # Population: list of individuals
        # ---------------------------------
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual, self.params['MU'])
        # -------------------------------
        # Operators (mate, mutate, select)
        # -------------------------------
        for name, (op_name, kwargs) in self.params['operators'].items():
            # Get the actual DEAP function by name
            op_func = getattr(tools, op_name)
            # Register it with the toolbox
            self.toolbox.register(name, op_func, **kwargs)
        self.toolbox.decorate("mate", checkStrategy(self.params['MIN_STRATEGY']))
        self.toolbox.decorate("mutate", checkStrategy(self.params['MIN_STRATEGY']))
        # ---------------------------------------------
        # Evaluation: pass-through to user's objective
        # ---------------------------------------------
        self.toolbox.register("evaluate", self._evaluate)
    
    def _evaluate(self,x):
        return self.objective_func(x),

    def initialize_population(self):
        # Initialize population
        self.pop = self.toolbox.population()

        # Evaluate initial population
        invalid_ind = [ind for ind in self.pop if not ind.fitness.valid]
        fitnesses = self.toolbox.map(self.toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit


    def optimize(self, strategy="plus", verbose=__debug__):
        """
        Run the (μ, λ) or (μ + λ) Evolution Strategy optimization.

        Args:
            strategy (str): Either "plus" for (μ + λ) or "comma" for (μ, λ).
            verbose (bool): If True, prints generation logs.

        Returns:
            self.pop (list): Final evolved population.
            self.logbook (Logbook): Evolution statistics over generations.
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

        best_fitness = -float("inf")
        stagnation_count = 0

        # Initialize Population
        self.initialize_population()

        # Setup Hall of Fame
        self.hall_of_fame = tools.HallOfFame(1)
        self.hall_of_fame.update(self.pop)

        # Record initial stats
        record = self.logger.stats.compile(self.pop)
        self.logger.logbook.record(gen=0, nevals=len(invalid_ind), **record)
        if verbose:
            print(self.logger.logbook.stream)

        print("\n" + "#" * 80)
        print("Starting Genetic Algorithm Optimization")
        print("#" * 80)

        # Begin optimization loop
        for gen in range(1, ngen + 1):
            # Save the population data to the logger
            fitness_vals = np.array([ind.fitness.values[0] for ind in self.pop])
            if self.logger:
                self.logger.log_population(gen, self.pop, fitness_vals)
            # Generate offspring from current population, applying mutate and mate to the population
            offspring = algorithms.varOr(self.pop, self.toolbox, lambda_, cxpb, mutpb)
            # Evaluate offspring
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = self.toolbox.map(self.toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            # Update Hall of Fame
            if self.hall_of_fame is not None:
                self.hall_of_fame.update(offspring)

            # Survivor selection: either (μ + λ) or (μ, λ)
            if strategy == "plus":
                self.pop[:] = self.toolbox.select(self.pop + offspring, mu)
            else:  # comma
                self.pop[:] = self.toolbox.select(offspring, mu)

            # Record stats
            record = self.logger.stats.compile(self.pop)
            self.logger.logbook.record(gen=gen, nevals=len(invalid_ind), **record)
            if verbose:
                print(self.logger.logbook.stream)

        if self.logger:
            self.logger.save()