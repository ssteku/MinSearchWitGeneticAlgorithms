from TestSuite import TestSuite

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
    tests = TestSuite()
    tests.plotFunction()

if __name__ == "__main__":
    main()