import random
import numpy as np
from deap import base
from deap import creator
from deap import tools

################################################################
# User Input
################################################################
N_bits = 10 # binary length
bounds = [(-512, 512), (-512, 512)]
num_variables = len(bounds)
N_pop = 40 # population size
maxiter = 1000
# Objective function
def eggholder(x):
    return (-(x[1] + 47) * np.sin(np.sqrt(abs(x[0]/2 + (x[1] + 47))))
            - x[0] * np.sin(np.sqrt(abs(x[0] - (x[1] + 47)))))


# CXPB  is the probability with which two individuals
    #       are crossed
    #
    # MUTPB is the probability for mutating an individual
CXPB, MUTPB = 0.5, 0.2
################################################################
# Function tool
################################################################
def binary_to_physical_nd(binary_individual, bounds, N_bits):
    """
    Convert a n-dimensional binary individual to physical values based on the bounds.
    
    Parameters:
    - binary_individual: A list of binary values representing the individual (e.g., [1, 0, 1, 0, ...]).
    - bounds: A list of tuples representing the bounds for each variable, 
              e.g., [ (x1_min, x1_max), (x2_min, x2_max), ..., (xn_min, xn_max) ].
    - N_bits: The number of bits used to represent each variable.
    
    Returns:
    - A list [x1_physical, x2_physical, ..., xn_physical] representing the physical values of the variables.
    """
    
    # Initialize a list to store the physical values
    physical_values = []
    
    # Number of variables (n-dimensional problem)
    n = len(bounds)
    
    # Split the binary string into n parts (one for each variable)
    for i in range(n):
        # Get the binary part for the i-th variable
        binary_part = binary_individual[i * N_bits: (i + 1) * N_bits]
        
        # Convert binary to decimal
        decimal_value = int("".join(map(str, binary_part)), 2)
        
        # Get the bounds for this variable
        variable_min, variable_max = bounds[i]
        
        # Map the decimal value to the physical range
        physical_value = variable_min + (decimal_value / (2**N_bits - 1)) * (variable_max - variable_min)
        
        # Append the physical value to the list
        physical_values.append(physical_value)
    
    # Return a list of physical values
    return physical_values

################################################################
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
# Create an individual class and store it in creator
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()

# Create a gene generator function and store it in the toolbox,
#       toolbox.register(<name>, function, **params_of_function)
toolbox.register("gene_generator", random.randint, 0, 1)
# Create an individual generator function with help of gene generator
#   Individual = A list of (N_bits * num_variables) binary values,
        # tools.initRepeat(container, creator, func, n)
            # container: The type of individual to create (in this case, creator.Individual).
            # creator: The type of object to create (in this case, creator.Individual).
            # func: The function that generates the individual attributes (in this case, toolbox.gene_generator).
            # n: The number of genes (or attributes) each individual should have (in this case, N_bits * num_variables).
toolbox.register("individual", tools.initRepeat, creator.Individual, 
    toolbox.gene_generator, N_bits*num_variables)
# Create a population generator function based on the individual generator
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

ind = toolbox.individual()

def evaluation(individual, bounds, N_bits):
    x = binary_to_physical_nd(individual,bounds,N_bits)
    return eggholder(x),

# Create a evaluation function for the calculation of the fitness of individual
toolbox.register("evaluate", evaluation, bounds=bounds, N_bits=N_bits)
# Create evolutionary operator
    # 1. Selection operator. 
    #   Tournament selection function provided by DEAP. Tournament selection works by randomly selecting a set of individuals, and then choosing the best one (the one with the highest fitness) from that set.
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
    # 3. Selection operator. 
    #   Tournament selection function provided by DEAP. Tournament selection works by randomly selecting a set of individuals, and then choosing the best one (the one with the highest fitness) from that set.
toolbox.register("select", tools.selTournament, tournsize=3)

pop = toolbox.population(N_pop)

print("Start of evolution")
fitnesses = list(map(toolbox.evaluate, pop))

for ind, fit in zip(pop, fitnesses):
    ind.fitness.values = fit

print("  Evaluated %i individuals" % len(pop))
# Extracting all the fitnesses of 
fits = [ind.fitness.values[0] for ind in pop]

# Variables for tracking convergence
g = 0  # Generation counter
stagnation_count = 0  # Counter for stagnation
stagnation_limit = 50  # Stop if no improvement for 50 generations
epsilon = 1e-6  # Convergence threshold (if best-worst fitness difference is small)

prev_best_fitness = max(fits)

# Begin the evolution
while g < maxiter:
    # A new generation
    g = g + 1
    print("-- Generation %i --" % g)

    # Select the next generation individuals, using a selection method to choose individuals from the population based on their fitness and the used method (such as tournament selection, roulette, etc.). 
    offspring = toolbox.select(pop, len(pop))
    # Clone the selected individuals, creating copies (clones) of those selected individuals to apply further evolutionary operations (crossover, mutation) without affecting the original population.
    offspring = list(map(toolbox.clone, offspring))

    # Apply crossover on the offspring
    for child1, child2 in zip(offspring[::2], offspring[1::2]):
        # cross two individuals with probability CXPB
        if random.random() < CXPB:
            toolbox.mate(child1, child2)

            # fitness values of the children in which the crossover happened need to be recalculated later
            del child1.fitness.values
            del child2.fitness.values

    # Apply mutation on the offspring
    for mutant in offspring:
            # mutate an individual with probability MUTPB
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                # fitness values of the who mutated need to be recalculated later
                del mutant.fitness.values

    # Revaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
    fitnesses = map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    print("  Evaluated %i individuals" % len(invalid_ind))

    # The population is entirely replaced by the offspring
    pop[:] = offspring

    # Gather all the fitnesses in one list and print the stats
    fits = [ind.fitness.values[0] for ind in pop]

    best_fitness = max(fits)
    worst_fitness = min(fits)

    print(f"  Best fitness: {best_fitness}, Worst fitness: {worst_fitness}")

    # Convergence check: fitness improvement stagnation
    if best_fitness <= prev_best_fitness:
        stagnation_count += 1
    else:
        stagnation_count = 0  # Reset stagnation counter if improvement occurs

    # Termination conditions
    if stagnation_count >= stagnation_limit:
        print(f"Stopping due to stagnation for {stagnation_limit} generations.")
        break

    if abs(best_fitness - worst_fitness) < epsilon:
        print("Stopping due to lack of diversity in the population.")
        break

    prev_best_fitness = best_fitness

print("-- End of (successful) evolution --")

best_ind = tools.selBest(pop, 1)[0]
print(f"Best individual is {best_ind}, Fitness: {best_ind.fitness.values}")

