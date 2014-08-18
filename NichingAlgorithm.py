import random
import math
import operator
from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from FunctionObject import FunctionObject
from functools import partial
from scipy import misc
from ChartDrawer import ChartDrawer

class NichingAlgorithm:
    def __init__(self, function_object):
        self.function_object = function_object
        self.number_of_cords = 2


    def distance(self, p0, p1):
        return math.sqrt((p0[0] - p1[0])**2 + (p0[1] - p1[1])**2)

    def share_function(self, distance, alfa, radius):
        # print "share_function: distnace %s" % distance
        # print "share_function: radius %s" % radius
        if distance < radius:
            return 1 - (distance/radius)*(distance/radius)
        else:
            return 0

    def eval_function(self, individual, radius, representatives):
        distanceSum = 0.001+sum([self.share_function(self.distance(individual, r), 1, radius) for r in representatives])
        # print "distanceSum: %s" % distanceSum
        calcWithY = partial(self.function_object.calculateWithXY, individual[0])
        calcWithX = partial(self.function_object.calculateWithYX, individual[1])
        derivY = abs(misc.derivative(calcWithY, individual[1], dx=1e-6))
        derivX = abs(misc.derivative(calcWithX, individual[0], dx=1e-6))

        return self.function_object.calculate(individual)/distanceSum, distanceSum*derivX, distanceSum*derivY

    def cxSet(self, ind1, ind2):
        tmpY = ind1[1]
        ind1[1] = ind2[1]
        ind2[1] = tmpY
        return ind1, ind2

    def mutSet(self, individual):
        # print "Init ind: %s" % individual
        localMax = self.function_object.xMax
        localMin = self.function_object.xMin
        delta = localMin - localMax

        mutationDelta = delta*0.0001

        xDelta = (random.randint(-80, 80)*mutationDelta)
        yDelta = (random.randint(-80, 80)*mutationDelta)
        # print "xDelta: %s" % xDelta
        # print "yDelta: %s" % yDelta

        individual[0] += xDelta;
        if(individual[0] < localMin ):
            individual[0] = localMin
        if(individual[0] > localMax ):
            individual[0] = localMax

        individual[1] += yDelta;
        if(individual[1] < localMin):
            individual[1] = localMin
        if(individual[1] > localMax):
            individual[1] = localMax
        # print "individual: %s" % individual
        return individual,
    def configure_toolbox(self):
        creator.create("FitnessMax", base.Fitness, weights=(1.0, -1.0, -1.0))
        creator.create("Individual", list, fitness=creator.FitnessMax)
        toolbox = base.Toolbox()

        # Attribute generator

        toolbox.register("population", tools.initRepeat, list)
        toolbox.register("evaluate", self.eval_function)
        toolbox.register("mate", self.cxSet)
        toolbox.register("mutate", self.mutSet)
        toolbox.register("select", tools.selTournament, tournsize=3)
        toolbox.register("get_best", tools.selBest, k=1)
        return toolbox

    def generate_attribute(self, min, max):
        return random.uniform(min, max)

    def generate_individual(self, toolbox, min, max):
        ind = creator.Individual()

        for i in range(self.number_of_cords):
            attr = self.generate_attribute(min, max)
            # print "attr: %s" % attr
            ind.append(attr)
        return ind
    def generate_subpopulations(self, toolbox, subpopulation_size, number_of_subpopulations):
        subpopulations = []
        for i in range(number_of_subpopulations):
            indyvidual_func = partial(self.generate_individual, toolbox, self.function_object.xMin, self.function_object.xMax)
            subpopulations.append(toolbox.population(func = indyvidual_func, n=subpopulation_size))

        return subpopulations
    def choose_representatives(self, subpopulations):
        return [random.choice(s) for s in subpopulations]

    def get_radius_distance(self, number_of_subpopulations, distance_factor):
        max_distance = math.sqrt(2*abs(self.function_object.xMax-self.function_object.xMin))
        return max_distance/(number_of_subpopulations*distance_factor)

    def calculate_subpopulation_size(self, population_size, number_of_subpopulations):
        return int(population_size/number_of_subpopulations)

    def evaluate_population(self, subpopulations, toolbox, radius_distance, representatives):
        for subpopulation in subpopulations:
            self.evaluate_subpopulation(toolbox, subpopulation, radius_distance, representatives)

    def evaluate_subpopulation(self, toolbox, subpopulation, radius_distance, representatives):
        fitnesses = []
        for ind in subpopulation:
            fitnesses.append(toolbox.evaluate(ind,radius_distance, representatives))
        for ind, fit in zip(subpopulation, fitnesses):
            ind.fitness.values = fit
    def get_offspring(self, toolbox, subpopulation):
        offspring = toolbox.select(subpopulation, len(subpopulation))
        return list(map(toolbox.clone, offspring))
    def perform_crossing(self, toolbox, offspring, crossing_probability):
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < crossing_probability:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values
    def perform_mutation(self, toolbox, offspring, mutation_probability):
        for mutant in offspring:
            if random.random() < mutation_probability:
                toolbox.mutate(mutant)
                del mutant.fitness.values

    def run_generation(self, toolbox, subpopulations, representatives, crossing_probability, mutation_probability, radius_distance):
        new_subpopulations = [None] * len(subpopulations)

        for i, subpopulation in enumerate(subpopulations):
            offspring = self.get_offspring(toolbox, subpopulation)
            self.perform_crossing(toolbox, offspring, crossing_probability)
            self.perform_mutation(toolbox, offspring, mutation_probability)
            # Evaluate the individuals with an invalid fitness

            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            current_representatives = representatives[:i] + representatives[i+1:]

            self.evaluate_subpopulation(toolbox, invalid_ind, radius_distance, current_representatives)

            # print("  Evaluated %i individuals" % len(invalid_ind))

            # The population is entirely replaced by the offspring
            subpopulation[:] = offspring

            # Gather all the fitnesses in one list and print the stats
            fits = [ind.fitness.values[0] for ind in subpopulation]

            length = len(subpopulation)
            mean = sum(fits) / length
            sum2 = sum(x*x for x in fits)
            std = abs(sum2 / length - mean**2)**0.5

            # print("  Min %s" % min(fits))
            # print("  Max %s" % max(fits))
            # print("  Avg %s" % mean)
            # print("  Std %s" % std)
            # print("  Best: %s" % toolbox.get_best(subpopulation)[0])
            new_subpopulations[i] = toolbox.get_best(subpopulation)[0]
        return new_subpopulations

    def findMinimas(self, population_size, number_of_generations, number_of_subpopulations, crossing_probability, mutation_probability, distance_factor):
        random.seed(64)

        current_generation = 0
        subpopulation_size = self.calculate_subpopulation_size(population_size, number_of_subpopulations)
        toolbox =  self.configure_toolbox()
        radius_distance =  self.get_radius_distance(number_of_subpopulations, distance_factor)

        subpopulations = self.generate_subpopulations(toolbox, subpopulation_size, number_of_subpopulations)
        representatives = self.choose_representatives(subpopulations)
        # Evaluate the entire population
        self.evaluate_population(subpopulations, toolbox, radius_distance, representatives)

        while current_generation < number_of_generations:
            representatives = self.run_generation(toolbox, subpopulations, representatives, crossing_probability, mutation_probability, radius_distance)
            current_generation += 1

        # print "---------------------> Best indyviduals <---------------------------"
        # print representatives
        return representatives
        # self.function_object.plot()
