"""

"""


from StarV.set.probstar import ProbStar
import numpy as np
from StarV.util.load_model import load_helicopter_model
import time
from scipy.signal import lti, step, impulse, lsim, cont2discrete
from StarV.plant.lode import LODE
from StarV.set.probstar import ProbStar
from StarV.set.star import Star
import numpy as np
import math
from scipy.sparse import csr_matrix
from StarV.util.plot import  plot_probstar,plot_star

 
def run_helicopter_model():

    plant = load_helicopter_model()

    print("A:",plant.A.shape)
    print("A_type:",type(plant.A))

    dims = plant.dim
    print("dim:",dims)
    
    #  returns list of initial states for each dimension
    init_state_bounds_list = []

    for dim in range(dims):
        ub = lb = 0.0
        if  dim % 28 >= 0 and dim % 28 <= 7:
            ub = 0.1
            lb = -0.1

        init_state_bounds_list.append((lb, ub))

    # init_sate_bounds_array=[np.array(list).reshape(48, 1) for list in init_sate_bounds_list]
    init_state_bounds_array = np.array(init_state_bounds_list)

    init_state_lb = init_state_bounds_array[:, 0]
    init_state_ub = init_state_bounds_array[:, 1]
    # print("init_state_bounds_list_shape:",len(init_state_bounds_list))
    # print("init_sate_bounds_list:",init_state_bounds_list)
    # print("init_state_bounds_array_shape:",init_state_bounds_array.shape)
    # print("init_sate_bounds_array:",init_state_bounds_array)

    # print("init_state_lb_shape:",init_state_lb.shape)
    # print("init_state_lb:",init_state_lb)


    X0 = Star(init_state_lb,init_state_ub)

    mu_X0 = 0.5*(X0.pred_lb + X0.pred_ub)
    a  = 3.0 
    sig_X0 = (mu_X0 - X0.pred_lb)/a
    Sig_X0 = np.diag(np.square(sig_X0))

    X0_probstar = ProbStar(X0.V, X0.C, X0.d,mu_X0, Sig_X0,X0.pred_lb,X0.pred_ub)

    # end_time=2
    # dt= 1 # time step
    # k= int(end_time/dt)
    k=2
    Xt = plant.multiStepReach(dt=1, X0=X0_probstar,U=None,k =k)
    Xt_Star= plant.multiStepReach(dt=1, X0=X0,k =k)


    return Xt,Xt_Star



if __name__ == '__main__':

    Xt,Xt_Star = run_helicopter_model()
    # print('Xt = ',Xt)
    # print('prob_dim:',Xt[0].dim) # dim = A.shape[0]
    # print('prob_nVars:',Xt[0].nVars) 
    # print('prob_V:',Xt[0].V.shape) 
    # dir_mat =np.array([[1, 0, 0],[0, 1, 0]])
    # print("dir_mat_3:",dir_mat)
    # rest_of_dims = np.zeros((2, Xt[0].dim - 3))
    # print("rest:",rest_of_dims.shape)
    # dir_mat = np.hstack((dir_mat, rest_of_dims))
    # print("dir_mat_shape:",dir_mat.shape)
    # print("dir_mat:",dir_mat)
    # plot_probstar(Xt,dir_mat=dir_mat)

