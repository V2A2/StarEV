"""

"""


from StarV.set.probstar import ProbStar
import numpy as np
from StarV.util.load_model import load_MNA5_model
import time
from StarV.plant.lode import LODE
from StarV.set.probstar import ProbStar
from StarV.set.star import Star
import numpy as np
import math
from scipy.sparse import csr_matrix
from StarV.util.plot import  plot_probstar,plot_star

 
def run_MNA5_model():

    plant = load_MNA5_model()

    dims = plant.dim
    print("dim:",dims)

    # input constrait
    # u1,u2,u3,u4,u5 = 0.1
    # u6,u7,u8,u9 = 0.2
    # U = [[0.1],[0.1],[0.1],[0.1],[0.1],[0.2],[0.2],[0.2],[0.2]]
    # print("U_type:",type(U)
    U = np.empty(9)

    for i in range(9):
        if i < 5:
            U[i] = 0.1
        else:
            U[i] = 0.2

    # print("U:",U)
    # print("U_type:",type(U))
    # print("U_shape:",U.shape)

    input_lb = np.array(U)
    input_ub = input_lb
    print("input_lb:",input_lb)
    print("input_type:",type(input_lb))
    print("input_shape:",input_lb.shape)

    # create Star for initial input bounds
    U = Star(input_lb,input_ub)

    # create ProbStar for initial input
    mu_U = 0.5*(U.pred_lb + U.pred_ub)
    a  = 3.0 
    sig_U = (mu_U- U.pred_lb)/a
    Sig_U= np.diag(np.square(sig_U))
    U_probstar = ProbStar(U.V, U.C, U.d,mu_U, Sig_U,U.pred_lb,U.pred_ub)


    #  returns list of initial states bounds for each dimension, construct a ProbSatr for initial state
    init_state_bounds_list = []

    for dim in range(dims):
        if dim < 10:
            lb = 0.0002
            ub = 0.00025
        elif dim < dims:
            lb = ub = 0
        elif dim >= dims and dim < dims + 5:
            # first 5 inputs
            lb = ub = 0.1
        elif dim >= dims + 5 and dim < dims + 9:
            # second 4 inputs
            lb = ub = 0.2
        else:
            raise RuntimeError('Unknown dimension: {}'.format(dim))
            
        init_state_bounds_list.append((lb, ub))

    init_state_bounds_array = np.array(init_state_bounds_list)

    init_state_lb = init_state_bounds_array[:, 0]
    init_state_ub = init_state_bounds_array[:, 1]
    # print("init_state_bounds_list_shape:",len(init_state_bounds_list))
    # print("init_sate_bounds_list:",init_state_bounds_list)
    # print("init_state_bounds_array_shape:",init_state_bounds_array.shape)
    # print("init_sate_bounds_array:",init_state_bounds_array)

    # print("init_state_lb_shape:",init_state_lb.shape)
    # print("init_state_lb:",init_state_lb)

    # create Star for initial state 
    X0 = Star(init_state_lb,init_state_ub)


    # create ProbStar for initial state 
    mu_X0 = 0.5*(X0.pred_lb + X0.pred_ub)
    a  = 3.0 
    sig_X0 = (mu_X0 - X0.pred_lb)/a
    Sig_X0 = np.diag(np.square(sig_X0))
    X0_probstar = ProbStar(X0.V, X0.C, X0.d,mu_X0, Sig_X0,X0.pred_lb,X0.pred_ub)
    
    
    
    dt = 0.1
    time_bound = math.pi*2
    print("time_bound:",time_bound)
    k = int (time_bound/ dt)
    print("num_steps_k:",k)
    k = int(time_bound / dt)
    Xt = plant.multiStepReach(dt=dt, X0=X0_probstar,U=U_probstar,k =k)
    # Xt = plant.multiStepReach(dt, X0=X0,U=U,k =k)


    return Xt



if __name__ == '__main__':

    start_time = time.time()
    Xt = run_MNA5_model()

    time_duration = time.time() - start_time
    print("computation time: {} seconds".format(time_duration))
    # print('Xt = ',Xt)
    # print('prob_dim:',Xt[0].dim) # dim = 48 = A.shape[0]
    # print('prob_nVars:',Xt[0].nVars) 
    # print('prob_V:',Xt[0].V.shape) 
    print("start mapping matrix")
    dir_mat =np.array([[1, 0, 0],[0, 1, 0]])
    # print("dir_mat_3:",dir_mat)
    rest_of_dims = np.zeros((2, Xt[0].dim - 3))
    # print("rest:",rest_of_dims.shape)
    dir_mat = np.hstack((dir_mat, rest_of_dims))
    print("dir_mat_shape:",dir_mat.shape)
    # print("dir_mat:",dir_mat)
    print("start 2D ploting")
    # plot_probstar(Xt,dir_mat=dir_mat)
    print("end of 2D ploting")


