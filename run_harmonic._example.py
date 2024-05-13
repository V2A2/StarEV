"""

"""


from StarV.set.probstar import ProbStar
import numpy as np
from StarV.util.load_model  import load_harmonic_oscillator_model
import time
from scipy.signal import lti, step, impulse, lsim, cont2discrete
from StarV.plant.lode import LODE
from StarV.set.star import Star
import numpy as np
import math
from scipy.sparse import csr_matrix
from StarV.util.plot import  plot_probstar,plot_star

 
def run_harmonic():


    # autonomous probstar basis matrix--- input free
    A = np.array([[0., 1.], [-1., 0]], dtype = float) 
    B = np.eye(2)

    plant = LODE(A,B)

    # initial sate for x = [-6, -5] and y = [0, 1]
    init_lb = np.array([-6., 0.])
    init_ub = np.array([-5., 1.])


    # input ranges  u1, u2 = [-0.5, 0.5]
    input_lb = np.array([-0.5, -0.5])
    input_ub = np.array([0.5, 0.5])

     
     # construct Star with initial state range and input range
    X0 = Star(init_lb,init_ub)
    U = Star(input_lb,input_ub)
    
    
    # covert initial state x ,y and input u1, u2 to probstar
    mu_X0 = 0.5*(X0.pred_lb + U.pred_ub)
    mu_U = 0.5*(U.pred_lb + U.pred_ub)

    print('Mean of initial state variables: mu = {}'.format(mu_X0))
    # print('Standard deviation of predicate variables: sig = {}'.format(sig))
    a  = 3.0 
    sig_X0 = (mu_X0 - X0.pred_lb)/a
    sig_U = (mu_U - U.pred_lb)/a

    Sig_X0 = np.diag(np.square(sig_X0))
    Sig_U = np.diag(np.square(sig_U))
    print('Variance matrix of initial state variables: Sig = {}'.format(Sig_X0))

    X0_probstar = ProbStar(X0.V, X0.C, X0.d,mu_X0, Sig_X0,X0.pred_lb,X0.pred_ub)
    U_probstar = ProbStar(U.V, U.C, U.d,mu_U, Sig_U,U.pred_lb,U.pred_ub)
 
    end_time=math.pi/2

    dt= math.pi/4 # step size pi/4
    k= int(end_time/dt)
    print("k:",type(k))
    print("k:=",k)

    # rechibility analysis for multisteps
    Xt_ProbStar = plant.multiStepReach(dt, X0=X0_probstar, U=U_probstar,k =k)
    # Xt_Star = plant.multiStepReach(dt, X0=X0, U=U,k =k)

    # Xt_ProbStar = plant.multiStepReach(dt, X0=X0_probstar,k =k)
    # Xt_Star = plant.multiStepReach(dt, X0=X0,k =k)

    return Xt_ProbStar



if __name__ == '__main__':

    Xt_Prob = run_harmonic()
    print('Xt_Prob = ',Xt_Prob)
    # print('Xt_Star = ',Xt_Star)
   
    #  plot probstar of rechibility with all steps 
    plot_probstar(Xt_Prob)
    # plot_star(Xt_Star)

 
