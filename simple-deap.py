import random, sys
from deap import base
from deap import tools
from deap import creator

IND_SIZE = 5
NGEN = 1000
CXPB = 0.7
MUTPB = 0.2

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("attr_float", random.random)
toolbox.register("individual", tools.initRepeat, creator.Individual,
                 toolbox.attr_float, n=IND_SIZE)

def evaluateInd(individual):
    result = sum(abs(individual[i]) for i in range(0,len(individual)))
    return result,

toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", evaluateInd)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

pop = toolbox.population(n=100)

for g in range(NGEN):
    # Select the next generation individuals
    offspring = toolbox.select(pop, len(pop))
    # Clone the selected individuals
    offspring = list(map(toolbox.clone, offspring))


    # Apply crossover on the offspring
    for child1, child2 in zip(offspring[::2], offspring[1::2]):
        if random.random() < CXPB:
            toolbox.mate(child1, child2)
            del child1.fitness.values
            del child2.fitness.values

    # Apply mutation on the offspring
    for mutant in offspring:
        if random.random() < MUTPB:
            toolbox.mutate(mutant)
            del mutant.fitness.values

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    # The population is entirely replaced by the offspring
    pop[:] = offspring


# Gather all the fitnesses in one list and print the stats
fits = [ind.fitness.values[0] for ind in pop]
        
length = len(pop)
mean = sum(fits) / length
sum2 = sum((x - mean)**2 for x in fits)
std = (sum2 / (length-1))**0.5
        
print("  Min %s" % min(fits))
print("  Max %s" % max(fits))
print("  Avg %s" % mean)
print("  Std %s" % std)   
    
for ind in pop:
    if ind.fitness.values[0] == min(fits):
        print("Best individual: \n", ind)
        print("Best Fitness: ", ind.fitness.values[0])
        sys.exit()
