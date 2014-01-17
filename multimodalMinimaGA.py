#    This file is part of DEAP.
#
#    DEAP is free software: you can redistribute it and/or modify
#    it under the terms of the GNU Lesser General Public License as
#    published by the Free Software Foundation, either version 3 of
#    the License, or (at your option) any later version.
#
#    DEAP is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#    GNU Lesser General Public License for more details.
#
#    You should have received a copy of the GNU Lesser General Public
#    License along with DEAP. If not, see <http://www.gnu.org/licenses/>.

import random
import deap
import math
import operator
from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from functionObject import FunctionObject
from functools import partial
from scipy import misc
from ChartDrawer import ChartDrawer


class NichingAlgorithm:
    def __init__(self, functionObject):
        self.functionObject = functionObject
        self.numberOfCords = 2


    def distance(self, p0, p1):
        return math.sqrt((p0[0] - p1[0])**2 + (p0[1] - p1[1])**2)
    def shareFunction(self, distance, alfa, radius):
        # print "shareFunction: distnace %s" % distance
        # print "shareFunction: radius %s" % radius
        if distance < radius:
            return 1 - (distance/radius)*(distance/radius)
        else:
            return 0
    def evalFunction(self, individual, radius, representatives):
        distanceSum = 0.001+sum([self.shareFunction(self.distance(individual, r), 1, radius) for r in representatives])
        # print "distanceSum: %s" % distanceSum
        calcWithY = partial(self.functionObject.calculateWithXY, individual[0])
        calcWithX = partial(self.functionObject.calculateWithYX, individual[1])
        derivY = abs(misc.derivative(calcWithY, individual[1], dx=1e-6))
        derivX = abs(misc.derivative(calcWithX, individual[0], dx=1e-6))
        
        return self.functionObject.calculate(individual)/distanceSum, distanceSum*derivX, distanceSum*derivY

    def cxSet(self, ind1, ind2):
        tmpY = ind1[1]
        ind1[1] = ind2[1]
        ind2[1] = tmpY
        return ind1, ind2
        
    def mutSet(self, individual):
        # print "Init ind: %s" % individual
        localMax = self.functionObject.xMax
        localMin = self.functionObject.xMin
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
    def configureToolBox(self):    
        creator.create("FitnessMax", base.Fitness, weights=(1.0, -1.0, -1.0))
        creator.create("Individual", list, fitness=creator.FitnessMax)
        toolbox = base.Toolbox()

        # Attribute generator
       
        toolbox.register("population", tools.initRepeat, list)
        toolbox.register("evaluate", self.evalFunction)
        toolbox.register("mate", self.cxSet)
        toolbox.register("mutate", self.mutSet)
        toolbox.register("select", tools.selTournament, tournsize=3)
        toolbox.register("get_best", tools.selBest, k=1)
        return toolbox
    def generateAttribute(self, min, max):
        return random.uniform(min, max)

    def generateIndividual(self, toolbox, min, max):
        ind = creator.Individual()

        for i in range(self.numberOfCords):
            attr = self.generateAttribute(min, max)
            # print "attr: %s" % attr
            ind.append(attr)
        return ind
    def generateSubpopulations(self, toolbox, subpopulationSize, numberOfSubpopulations):
        subpopulations = []    
        for i in range(numberOfSubpopulations):
            indyvidualFunc = partial(self.generateIndividual, toolbox, self.functionObject.xMin, self.functionObject.xMax)
            subpopulations.append(toolbox.population(func = indyvidualFunc, n=subpopulationSize))

        return subpopulations
    def chooseRepresentatives(self, subpopulations):
        return [random.choice(s) for s in subpopulations]
    def getRadiusDistance(self, numberOfSubpopulations, distanceFactor):
        maxDistance = math.sqrt(2*abs(self.functionObject.xMax-self.functionObject.xMin))
        return maxDistance/(numberOfSubpopulations*distanceFactor)
    def findMinimas(self, populationSize, numberOfGenerations, numberOfSubpopulations, crossingProbability, mutationProbability, distanceFactor):
        random.seed(64)
        
        currentGeneration = 0
        subpopulationSize = int(populationSize/numberOfSubpopulations)
        toolbox =  self.configureToolBox()
        radiusDistance =  self.getRadiusDistance(numberOfSubpopulations, distanceFactor)

        subpopulations = self.generateSubpopulations(toolbox, subpopulationSize, numberOfSubpopulations)
        # print("Start of evolution")
        representatives = self.chooseRepresentatives(subpopulations)
        # Evaluate the entire population
        for subpopulation in subpopulations:
            fitnesses = []
            for ind in subpopulation:
                fitnesses.append(toolbox.evaluate(ind,radiusDistance, representatives))
            for ind, fit in zip(subpopulation, fitnesses):
                ind.fitness.values = fit
            # print("  Evaluated %i individuals" % len(subpopulation))

        # representatives = chooseRepresentatives(subpopulations)
        bestIndyviduals = dict()

        while currentGeneration < numberOfGenerations:
            # print("-- Generation %i --" % currentGeneration)
            next_repr = [None] * len(subpopulations)

            for i, subpopulation in enumerate(subpopulations):
                # Select the next generation individuals
                offspring = toolbox.select(subpopulation, len(subpopulation))
                # Clone the selected individuals
                offspring = list(map(toolbox.clone, offspring))
                # Apply crossover and mutation on the offspring
                for child1, child2 in zip(offspring[::2], offspring[1::2]):
                    if random.random() < crossingProbability:
                        toolbox.mate(child1, child2)
                        del child1.fitness.values
                        del child2.fitness.values
                
                

                for mutant in offspring:
                    if random.random() < mutationProbability:
                        toolbox.mutate(mutant)
                        del mutant.fitness.values
                
                # Evaluate the individuals with an invalid fitness

                invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
                fitnesses = []
                r = representatives[:i] + representatives[i+1:]
                for ind in invalid_ind:
                    fitnesses.append(toolbox.evaluate(ind,radiusDistance, r))
                for ind, fit in zip(invalid_ind, fitnesses):
                    ind.fitness.values = fit
                
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
                next_repr[i] = toolbox.get_best(subpopulation)[0]
            representatives = next_repr
            currentGeneration += 1

        # print "---------------------> Best indyviduals <---------------------------"
        # print representatives
        return representatives
        # self.functionObject.plot()

class TestSuite:
    def init(self):
        self.A = [[0.5, 0.5], [0.25, 0.25], [0.25, 0.75], 
            [0.75, 0.25], [0.75, 0.75], [0.35, 0.35],
            [1.0, 1.3], [1.4, 0.2], [0.34, 1.7],
            [1.5, 1.3], [1.1, 0.9], [0.9, 1.7]]
        self.C = [0.002, 0.005, 0.004,
            0.003, 0.002, 0.001,
            0.002, 0.0023, 0.0035,
            0.005, 0.0055, 0.0022]
        self.REPEAT_COUNT = 5
        self.MAX_RANGE = 2.0
        self.MIN_RANGE = 0.0
        self.STEP = 0.05
        self.POPULATION_SIZE = 150
        self.SUBPOPULATION_NUMBER = 12
        self.CXPB = 0.6
        self.MUTPB = 0.1
        self.RAY_DISTANCE = 25
        self.GEN_NUMBER = 200
        self.chartDraw = ChartDrawer()     
        self.functionObject = FunctionObject(xMin = self.MIN_RANGE, xMax = self.MAX_RANGE, yMin = self.MIN_RANGE, yMax = self.MAX_RANGE, step = self.STEP);
        self.functionObject.setExtremums(self.A, self.C)
        self.algorithm = NichingAlgorithm(self.functionObject)
    def getStatisticalResultOfAlgorithm(self):
        print "ray: %s" % self.RAY_DISTANCE
        avarageErrSum = 0       
        for i in range(self.REPEAT_COUNT):     
            results= self.algorithm.findMinimas(self.POPULATION_SIZE, self.GEN_NUMBER, self.SUBPOPULATION_NUMBER, self.CXPB, self.MUTPB, self.RAY_DISTANCE)
            tests = TestSuite()
            avarageErrSum += self.getAvarageError(self.functionObject, self.A, results)
        print "Avarage err: %s" % str(avarageErrSum/self.REPEAT_COUNT)
        return avarageErrSum/self.REPEAT_COUNT

    def performGenerationsTest(self):
        self.init()
        self.GEN_NUMBER = 0
        results = list()
        avErrors = []
        xLabels = []
        for g in range(20):
            print "-- performGenerationsTest: %s --" % str(g)
            self.GEN_NUMBER += 5
            xLabels.append(str(self.GEN_NUMBER))
            avErrors.append(self.getStatisticalResultOfAlgorithm())
        self.chartDraw.draw("Avarage error of found maximums values using different number of generations", "Generations", avErrors, xLabels, "Number of generations", "Avarage error value")

    def performPopulationsTest(self):
        self.init()
        self.POPULATION_SIZE = 0
        results = list()
        avErrors = []
        xLabels = []
        for g in range(30):
            print "-- performPopulationsTest: %s --" % str(g)
            self.POPULATION_SIZE += 10
            xLabels.append(str(self.POPULATION_SIZE))
            avErrors.append(self.getStatisticalResultOfAlgorithm())
        self.chartDraw.draw("Avarage error of found maximums values using different number of populations", "Populations", avErrors, xLabels, "Number of populations", "Avarage error value")

    def performSubPopulationsTest(self):
        self.init()
        self.SUBPOPULATION_NUMBER = 0
        results = list()
        avErrors = []
        xLabels = []
        for g in range(40):
            print "-- performSubPopulationsTest: %s --" % str(g)
            self.SUBPOPULATION_NUMBER += 1
            xLabels.append(str(self.SUBPOPULATION_NUMBER))
            avErrors.append(self.getStatisticalResultOfAlgorithm())
        self.chartDraw.draw("Avarage error of found maximums values using different number of subpopulations", "Subpopulations", avErrors, xLabels, "Number of subpopulations", "Avarage error value")

    def performCrossingTest(self):
        self.init()
        self.CXPB = 0.00
        results = list()
        avErrors = []
        xLabels = []
        for g in range(20):
            print "-- performCrossingTest: %s --" % str(g)
            self.CXPB += 0.05
            xLabels.append(str(self.CXPB))
            avErrors.append(self.getStatisticalResultOfAlgorithm())
        self.chartDraw.draw("Avarage error of found maximums values using different crossing probability", "Crossing probabilities", avErrors, xLabels, "Crossing probability", "Avarage error value")

    def performMutationTest(self):
        self.init()
        self.MUTPB = 0.00
        results = list()
        avErrors = []
        xLabels = []
        for g in range(20):
            print "-- performMutationTest: %s --" % str(g)
            self.MUTPB += 0.05
            xLabels.append(str(self.MUTPB))
            avErrors.append(self.getStatisticalResultOfAlgorithm())
        self.chartDraw.draw("Avarage error of found maximums values using different mutation probability", "Mutation probabilities", avErrors, xLabels, "Mutation probability", "Avarage error value")

    def performRayTest(self):
        self.init()
        self.RAY_DISTANCE = 0
        results = list()
        avErrors = []
        xLabels = []
        for g in range(20):
            print "-- performRayTest: %s --" % str(g)
            self.RAY_DISTANCE += 1
            avErrors.append(self.getStatisticalResultOfAlgorithm())
            xLabels.append(str(self.algorithm.getRadiusDistance(self.SUBPOPULATION_NUMBER, self.RAY_DISTANCE)))
        self.chartDraw.draw("Avarage error of found maximums values using different ray distance between niches", "Distance rays", avErrors, xLabels, "Distance ray", "Avarage error value")


    def getAvarageError(self, function, extremums, results):
        extremums = sorted(extremums, key=lambda cords: function.calculate(cords))
        results = sorted(results, key=lambda cords: function.calculate(cords))

        # print "---------------extremums--------------------"
        # print extremums
        # print "---------------results--------------------"
        # print results
        errorSum = 0
        resultCount = 0
        avarageValueSum = 0
        for extremum, result in zip(extremums, results):
            resultCount += 1
            avarageValueSum += function.calculate(extremum)
            errorSum += abs(function.calculate(extremum)-function.calculate(result))
        return errorSum/resultCount



def main():  
    tests = TestSuite()
    tests.performGenerationsTest()
    tests = TestSuite()
    tests.performSubPopulationsTest()
    tests = TestSuite()
    tests.performPopulationsTest()
    tests = TestSuite()
    tests.performCrossingTest()
    tests = TestSuite()
    tests.performMutationTest()
    tests = TestSuite()
    tests.performRayTest()

if __name__ == "__main__":
    main()                 


    
    # algorithms.eaMuPlusLambda(pop, toolbox, MU, LAMBDA, CXPB, MUTPB, NGEN, stats,
    #                           halloffame=hof)


# Pierwszy raport powinien byc wprawka przed magisterka, 
# brakowalo spisu tresci!
# na koncach punktowania powinny byc przecinki a na koncu kropka
# punktowania powinny byc mala litera
# pisze sie "In the figure"
# Jak sa wykresy to powinny byc tytuly.bic
# Formatowanie powinno byc duzo lepsze, bo wykresy byly na osobnych stronach.

# Cel zadania jest zrobic algorytm znajdujacy jakies minima sensownie.
# Mamy pewne klasy fnkcji 
# Zrobic testy przy roznych parametrach krzyzowania, mutacji , rateach tych rzeczy i pokazac wyniki.

# Popytac z dziale dydaktyki i zapytac o to jakie papiery i kiedy.
# Prawdopodobnie trzeba wydrukowac karte i zaniesc z podpisami do dzialu dydaktyki (?) 

