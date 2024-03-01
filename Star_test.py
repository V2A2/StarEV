import numpy as np
from StarV.set.star import Star
from StarV.util.plot import  plot_star
import copy
import scipy.sparse as sp
from scipy.sparse import csc_matrix

# # # Create star set with lb, ub --args == 2
# lb = np.array([0.2,0.3])
# ub = np.array([0.7,0.7])
# S = Star(lb,ub)
# Star.__str__(S)

# # plot_star(S)

# try:
#     S1 = S.affineMap(A=np.random.rand(2,2))
#     P = []
#     P.append(S)
#     P.append(S1)
# except Exception:
#     print('Test Fail!')
# else:
#     print("Test Successfull!")

# def estimateRange(self, index):
#     """Quickly estimate minimum value of a state x[index]"""

#     assert index >= 0 and index <= self.dim-1, 'error: invalid index'

#     v = self.V[index, 1:self.nVars+1]
#     c = self.V[index, 0]
#     v1 = copy.deepcopy(v)
#     v2 = copy.deepcopy(v)
#     c1 = copy.deepcopy(c)
#     print('v_type:',type(v1))
#     v1[v1 > 0] = 0  # negative part
#     v2[v1 < 0] = 0  # positive part
#     v1 = v1.reshape(1, self.nVars)
#     v2 = v2.reshape(1, self.nVars)
#     c1 = c1.reshape((1,))
#     print('v1_type:',type(v1))
#     min_val = c1 + np.matmul(v1, self.pred_ub) + \
#         np.matmul(v2, self.pred_lb)
#     max_val = c1 + np.matmul(v1, self.pred_lb) + \
#         np.matmul(v2, self.pred_ub)

#     return min_val, max_val


# min_val, max_val = estimateRange(S1,1)
# print('MinValue = {}, true_val = {}'.format(min_val, S1.pred_lb[1]))
# print('MaxValue = {}, true_val = {}'.format(max_val, S1.pred_ub[1]))


# print('min_value:',type(max_val))
# # assert min_val == S1.pred_lb[1] and \
# #     max_val == S1.pred_ub[1], 'error: wrong results'


# lb, ub = S1.estimateRanges()
# print('lb = {}, true_lb = {}'.format(lb, S1.pred_lb))
# print('ub = {}, true_ub = {}'.format(ub, S1.pred_ub))

# # plot_star(P)        

# X=np.array([[1,2,3],[2,3,4]])
# X1=np.transpose(X)
# pred_lb = np.random.rand(3,1)

# pred_ub = np.random.rand(1,3)
# M = np.matmul(pred_lb,pred_ub)
# print("M:",M)
# print("pred_lb:",pred_lb)
# print("pred_ub:",pred_ub)
# print("pred_lb_sz:",np.shape(pred_lb))
# print("X:",X)
# print("X1:",X1)
# print("X_sz:",X.shape)


# Cmin,dmin= Star.getMinimizedConstraints(S1)
# print("Cmin:",Cmin)
# print("dmin:",dmin)

# Create star set with [V, C, d, pred_lb, pred_ub] --args == 5
    
# V = np.array([[0, 1 ,2 ],[0, -1,-2]])
# V2= np.delete(V,0,1)
# print("V2:",V2)

a = np.array([1, 2])
b = np.array([5, 6])
c= np.concatenate((a, b),axis=0)
print("c:",c)
print("c:",len(c))
print("c_shape:",c.shape)
print("a_type:",a.shape)

# C = np.array([[ -1 ,2 ],[ -1,2]])

# d = np.array([ 1 ,1])
# pred_lb = np.array([1, 2])
# pred_ub = np.array([4, 3])
# S3 = Star(V, C, d, pred_lb, pred_ub)
# Star.__str__(S3)
# print("-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- \n")

# try:
#     S4 = S3.affineMap(A=np.random.rand(2,2))
#     Star.__str__(S4)
#     Star.__str__(S4)
#     P1 = []
#     P1.append(S3)
#     P1.append(S4)
#     plot_star(P1)
# except Exception:
#     print('Test Fail!')
# else:
#     print("Test Successfull!")


# Cmin,dmin= Star.getMinimizedConstraints(S3)
# print("Cmin:",Cmin)
# print("dmin:",dmin)

# min_val, max_val = S4.estimateRange(0)
# print('MinValue = {}, true_val = {}'.format(min_val, pred_lb[0]))
# print('MaxValue = {}, true_val = {}'.format(max_val, pred_ub[0]))
# assert min_val == pred_lb[0] and \
#     max_val == pred_ub[0], 'error: wrong results'

# C = np.zeros((1, 3))
# print("C:", C)