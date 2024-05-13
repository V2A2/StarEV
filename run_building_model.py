"""

"""


from StarV.set.probstar import ProbStar
import numpy as np
from StarV.util.load_model import load_building_model
import time
from scipy.signal import lti, step, impulse, lsim, cont2discrete
# from StarV.plant.lode1 import LODE
from StarV.set.probstar import ProbStar
from StarV.set.star import Star
import numpy as np
import math
from scipy.sparse import csr_matrix
from StarV.util.plot import  plot_probstar,plot_star
import matplotlib.pyplot as plt


    
def run_building_model():

        plant = load_building_model()

        print("A:",plant.A.shape)
        print("A_type:",type(plant.A))

        # 0.8 <= u1 <= 1.0 only one input
        input_lb = np.array([0.8])
        input_ub = np.array([1])
        print("input_shape:",input_lb.shape)

        # X0 = A[:,1]
        # print("X0_type:",type(X0))
        # print("X0:",X0.shape)
        # U = B[:,0]
        # U1 = U.reshape(1, -1)
        # print("U_shape:",U.shape)
        # print("U_type:",type(U))
        # print("U:",U)
        # print("U1_shape:",U1.shape)
        # print("U1_type:",type(U1))
        # print("U1:",U1)
        U = Star(input_lb,input_ub)

        mu_U = 0.5*(U.pred_lb + U.pred_ub)
        a  = 3.0 
        sig_U = (mu_U- U.pred_lb)/a
        Sig_U= np.diag(np.square(sig_U))
        U_probstar = ProbStar(U.V, U.C, U.d,mu_U, Sig_U,U.pred_lb,U.pred_ub)


        dims = plant.dim
        print("dim:",dims)
        
        #  returns list of initial states for each dimension
        init_state_bounds_list = []

        for dim in range(dims):
            if dim < 10:
                lb = 0.0002
                ub = 0.00025
            elif dim == 24:
                lb = -0.0001
                ub = 0.0001
            else:
                lb = ub = 0

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
        dt = 0.5
        time_bound = math.pi*2
        print("time_bound:",time_bound)
        k = int (time_bound/ dt)
        print("num_steps_k:",k)
        k = int(time_bound / dt)
        Xt = plant.multiStepReach(dt=dt, X0=X0_probstar,U=U_probstar,k = k)
        # Xt = plant.multiStepReach(dt=1, X0=X0,U = U, k =k)
        # Xt_1 = plant.multiStepReach(dt=1, X0=X0_probstar,k =k)


        return Xt


def random_two_dims_mapping( Xt, dim1,dim2):
    #  if isinstance(I,ProbStar) or isinstance(I, Star):
        dims = Xt[0].dim
        M = np.zeros((2, dims))
        print("M_size:",M.shape)
        print("M:",M)
        M[0][dim1-1] = 1
        M[1][dim2-1] = 1
        print("M_2:",M)
        return M


     

if __name__ == '__main__':

    start_time = time.time()
    Xt = run_building_model()

    time_duration = time.time() - start_time
    print("computation time: {} seconds".format(time_duration))       

    i = len(Xt)
    print('length_Xt:',i)
    # print('prob_dim:',Xt[0].dim) # dim = 48 = A.shape[0]
    # print('prob_nVars:',Xt[0].nVars) 
    # print('prob_V:',Xt[0].V.shape) 
    print("start mapping matrix")
    dir_mat =np.array([[1, 0, 0],[0, 1, 0]])
    # print("dir_mat_3:",dir_mat)
    rest_of_dims = np.zeros((2, Xt[0].dim - 3))
    # print("rest:",rest_of_dims.shape)
    dir_mat = np.hstack((dir_mat, rest_of_dims))
    # print("dir_mat_shape:",dir_mat.shape)
    # print("dir_mat:",dir_mat)
    # plot_probstar(Xt,dir_mat=dir_mat)
    # print("start random 2D mapping")
    # dir_mat = random_two_dims_mapping(Xt[0],2,9)
    print("start 2D ploting")
    # plot_probstar(Xt,dir_mat=dir_mat)
    # dir_mat1 = random_two_dims_mapping(Xt_star,2,9)
    # plot_star(Xt_star,dir_mat=dir_mat1)
    print("end of 2D ploting")


