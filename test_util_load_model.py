
"""
Test util load_model
Qing Liu, 02/23/2024
"""
from StarV.util import load_model
import polytope as pc

class Test(object):
    """
    Testing module net class and methods
    """

    def __init__(self):

        self.n_fails = 0
        self.n_tests = 0

    def test_load_harmonic_oscillator_model(self):

        self.n_tests = self.n_tests + 1

        try:

            plant, lb, ub, input_lb, input_ub = load_model.load_harmonic_oscillator_model()
            print('plant info: ')
            plant.info()
            print('initial conditions: ')
            print('lower bound: {}'.format(lb))
            print('upper bound: {}'.format(ub))
            print('input conditions: ')
            print('input lower bound: {}'.format(input_lb))
            print('input upper bound: {}'.format(input_ub))
        except Exception:
            print('Test Fail!')
            self.n_fails = self.n_fails + 1
        else:
            print('Test Successfull!')

    def test_load_building_model(self):
      
        self.n_tests = self.n_tests + 1
     
        try:
            print("Building,begin print A plant,, B, C")
            plant, A, B, C = load_model.load_building_model()
            print('plant: ',plant.info())
        except Exception:
            print('Test Fail!')
            self.n_fails = self.n_fails + 1
        else:
            print('Test Successfull!')

    
    def test_load_iss_model(self):
     
        self.n_tests = self.n_tests + 1
     
        try:
            print("iss,begin print A plant,, B, C")
            plant, A, B, C = load_model.load_iss_model()
            print('plant: ',plant.info())
        except Exception:
            print('Test Fail!')
            self.n_fails = self.n_fails + 1
        else:
            print('Test Successfull!')


    def test_load_helicopter_model(self):
        
        self.n_tests = self.n_tests + 1
     
        try:
            print("heli28,begin print plant,A, B, C")
            plant, A= load_model.load_helicopter_model()
            print('plant: ',plant.info())
        except Exception:
            print('Test Fail!')
            self.n_fails = self.n_fails + 1
        else:
            print('Test Successfull!')


    def test_load_MNA5_model(self):
        
        self.n_tests = self.n_tests + 1
     
        try:
            print("MNA5,begin print plant,A, B, C")
            plant, A, B= load_model.load_MNA5_model()
            print('\ngA = {}'.format(plant.gA))
            print('\ngB = {}'.format(plant.gB))
            print('plant: ',plant.info())
        except Exception:
            print('Test Fail!')
            self.n_fails = self.n_fails + 1
        else:
            print('Test Successfull!')



if __name__ == "__main__":

    test_load = Test()
    print('\n=======================\
    ================================\
    ================================\
    # ===============================\n')
    test_load.test_load_harmonic_oscillator_model()
    test_load.test_load_building_model()
    test_load.test_load_iss_model()
    test_load.test_load_helicopter_model()
    test_load.test_load_MNA5_model()
    print('\n========================\
    =================================\
    =================================\
    =================================\n')
    print('Testing Load module: fails: {}, successfull: {}, \
    total tests: {}'.format(test_load.n_fails,
                            test_load.n_tests - test_load.n_fails,
                            test_load.n_tests))
    


