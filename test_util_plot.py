"""
Test of plot module
Author: Dung Tran
Date: 9/11/2022
"""

import numpy as np
from StarV.set.probstar import ProbStar
<<<<<<< HEAD
from StarV.util.plot import  plot_probstar
# from StarV.util.plot import probstar2polytope
=======
from StarV.set.star import Star
from StarV.util.plot import plot_probstar, plot_star, plot_2D_UnsafeSpec, get_bounding_box

>>>>>>> d6f456f630416bad2b82975fff47be17aa363e64


class Test(object):
    """
       Testing class methods
    """

    def __init__(self):

        self.n_fails = 0
        self.n_tests = 0

    def test_probstar2polytope(self):

        self.n_tests = self.n_tests + 1

        mu = np.random.rand(3,)
        Sig = np.eye(3)
        pred_lb = np.random.rand(3,)
        pred_ub = pred_lb + 0.2
        print('Testing probstar2Polytope method...')
        S = ProbStar(mu, Sig, pred_lb, pred_ub)
        # try:
        #     P = probstar2polytope(S)
        #     print('Polytope = {}'.format(P))
        # except Exception:
        #     print('Test Fail!')
        #     self.n_fails = self.n_fails + 1
        # else:
        #     print("Test Successfull!")

    def test_plot_probstar(self):

        self.n_tests = self.n_tests + 1

        mu = np.random.rand(2,)
        # print("mu:",mu)
        Sig = np.eye(2)
        # print("Sig:",Sig)
        pred_lb = np.random.rand(2,)
        # print("pred-lb:",pred_lb)
        pred_ub = pred_lb + 0.2
        print('Testing plot_probstar method...')
        # S = ProbStar(mu, Sig, pred_lb, pred_ub)
        # S1 = S.affineMap(A=np.random.rand(2,2))
        # P = []
        # P.append(S)
        # P.append(S1)
        # plot_probstar(P)
        try:
            S = ProbStar(mu, Sig, pred_lb, pred_ub)
            S1 = S.affineMap(A=np.random.rand(2,2))
            P = []
            P.append(S)
            P.append(S1)
            plot_probstar(P)
        except Exception:
            print('Test Fail!')
            self.n_fails = self.n_fails + 1
        else:
            print("Test Successfull!")

    def test_plot_2D_UnsafeSpec(self):
        
         # 0.5 <= d_k <= 2.5 AND 0.2 <= v_k <= v_ub
        unsafe_mat = np.array([[1.0, 0.0], [-1., 0.], [0., 1.], [0., -1.]])
        unsafe_vec = np.array([2.5, -0.5, 90.5, -0.2])

        try:
            plot_2D_UnsafeSpec(unsafe_mat, unsafe_vec)
        except Exception:
            print('Test Fail!')
            self.n_fails = self.n_fails + 1
        else:
            print("Test Successfull!")

    def test_get_bounding_box(self):
        
         # 0.5 <= d_k <= 2.5 AND 0.2 <= v_k <= v_ub
        unsafe_mat = np.array([[1.0, 0.0], [-1., 0.], [0., 1.], [0., -1.]])
        unsafe_vec = np.array([2.5, -0.5, 90.5, -0.2])

        try:
            lb, ub = get_bounding_box(unsafe_mat, unsafe_vec)
            print('lb = {}, ub = {}'.format(lb, ub))
        except Exception:
            print('Test Fail!')
            self.n_fails = self.n_fails + 1
        else:
            print("Test Successfull!")


if __name__ == "__main__":

    test_plot = Test()
    print('\n=======================\
    ================================\
    ================================\
    ===============================\n')
    # test_plot.test_probstar2polytope()
<<<<<<< HEAD
    test_plot.test_plot_probstar()
=======
    # test_plot.test_plot_probstar()
    test_plot.test_plot_2D_UnsafeSpec()
    #test_plot.test_get_bounding_box()
>>>>>>> d6f456f630416bad2b82975fff47be17aa363e64
    
    print('\n========================\
    =================================\
    =================================\
    =================================\n')
    print('Testing ProbStar Class: fails: {}, successfull: {}, \
    total tests: {}'.format(test_plot.n_fails,
                            test_plot.n_tests - test_plot.n_fails,
                            test_plot.n_tests))
