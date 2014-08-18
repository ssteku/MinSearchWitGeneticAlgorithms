from FunctionObject import FunctionObject
from ChartDrawer import ChartDrawer
from NichingAlgorithm import NichingAlgorithm
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
        self.STEP = 0.01
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

    def plotFunction(self):
        self.init()
        self.functionObject.plot()

    def getStatisticalResultOfAlgorithm(self):
        print "ray: %s" % self.RAY_DISTANCE
        avarageErrSum = 0
        for i in range(self.REPEAT_COUNT):
            results= self.algorithm.find_minimas(self.POPULATION_SIZE, self.GEN_NUMBER, self.SUBPOPULATION_NUMBER, self.CXPB, self.MUTPB, self.RAY_DISTANCE)
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
            self.POPULATION_SIZE += 15
            xLabels.append(str(self.POPULATION_SIZE))
            avErrors.append(self.getStatisticalResultOfAlgorithm())
        self.chartDraw.draw("Avarage error of found maximums values using different size of populations", "Populations", avErrors, xLabels, "Size of populations", "Avarage error value")

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
        error_sum = 0
        result_count = 0
        avarage_value_sum = 0
        for extremum, result in zip(extremums, results):
            result_count += 1
            avarage_value_sum += function.calculate(extremum)
            error_sum += abs(function.calculate(extremum)-function.calculate(result))
        return error_sum/result_count