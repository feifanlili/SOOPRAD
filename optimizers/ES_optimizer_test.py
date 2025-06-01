import random
import numpy as np
import array
from deap import base, creator, tools, algorithms

from optimizers.utils.benckmark_functions import ObjectiveFunction

class ES_Optimizer():
    DEFAULT_PARAMS = {         # Number of genes per individual
    "MU": 10,                # MU: Number of parents (survivors)
    "LAMBDA": 100,         # LAMBDA: Number of offspring per generation
    "maxiter": 500,             # Number of generations (ngen)
    "crossoverPB": 0.6,         # Probability of applying crossover
    "mutationPB": 0.3,          # Probability of applying mutation
    "MIN_STRATEGY": 0.001,      # Minimum strategy value for mutation strength
    "MAX_STRATEGY": 1.0,        # Maximum initial strategy value
    "random_seed": None,        # Optionally fix random seed

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
        # user_operator_config = params.get("operators", {})
        # self.operator_params = {**self.DEFAULT_PARAMS["operators"], **user_operator_config}

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
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
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


    def varOr(self, population, toolbox, lambda_, cxpb, mutpb):
        assert ( + mutpb) <= 1.0, (
            "The sum of the crossover and mutation probabilities must be smaller "
            "or equal to 1.0.")

        offspring = []
        for _ in range(lambda_):
            op_choice = random.random()
            if op_choice < cxpb:            # Apply crossover
                ind1, ind2 = [toolbox.clone(i) for i in random.sample(population, 2)]
                ind1, ind2 = toolbox.mate(ind1, ind2)
                del ind1.fitness.values
                offspring.append(ind1)
            elif op_choice < cxpb + mutpb:  # Apply mutation
                ind = toolbox.clone(random.choice(population))
                ind, = toolbox.mutate(ind)
                del ind.fitness.values
                offspring.append(ind)
        else:                           # Apply reproduction
            offspring.append(random.choice(population))

        return offspring

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

        mu = self.params['MU']
        lambda_ = self.params['LAMBDA']
        cxpb = self.params['crossoverPB']
        mutpb = self.params['mutationPB']
        ngen = self.params['maxiter']

        # Initialize population
        self.pop = self.toolbox.population(n=mu)

        # Evaluate initial population
        invalid_ind = [ind for ind in self.pop if not ind.fitness.valid]
        fitnesses = self.toolbox.map(self.toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Setup Hall of Fame
        self.hall_of_fame = tools.HallOfFame(1)
        self.hall_of_fame.update(self.pop)

        # Setup statistics tracker
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)

        # Setup logbook
        self.logbook = tools.Logbook()
        self.logbook.header = ['gen', 'nevals'] + stats.fields

        # Record initial stats
        record = stats.compile(self.pop)
        self.logbook.record(gen=0, nevals=len(invalid_ind), **record)
        if verbose:
            print(self.logbook.stream)

        # Begin evolution loop
        for gen in range(1, ngen + 1):
            # Generate offspring from current population
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
            record = stats.compile(self.pop)
            self.logbook.record(gen=gen, nevals=len(invalid_ind), **record)
            if verbose:
                print(self.logbook.stream)


sphere_func = ObjectiveFunction("sphere", dimension=2)
optimizer = ES_Optimizer(sphere_func.f,sphere_func.bounds)

