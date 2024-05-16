# from scipy.sparse.linalg import norm 
import numpy as np
np.set_printoptions(precision=4)
from scipy.linalg import expm
from numpy.linalg import norm
from StarV.set.star import Star
from StarV.plant.krylov_arnoldi import simKrylov, simReachKrylov,arnoldi
import math
from StarV.util.plot import  plot_probstar,plot_star
import copy

# ----------------test example 1: x0 is array 
A = np.array([[1, 2,3 ], [3, 4,3],[1,2,3]])
x0 = np.array([[1,1,1]])
h = 0.1
N = 5
m = 3

Vm, Hm = arnoldi(A,x0,m)
print("arnoldi_Vm:",Vm)
print("arnoldi_Hm:",Hm)
X = simKrylov(A, x0, h, N,m)
print("simKrylov_X:",X)


# ----------------harmonic example: x0 is star set
A = np.array([[0,1],[-1,0]])
h = math.pi/4
# print("h:",h)
N = int((math.pi/2)/h)
m = 2
lb = np.array([-6,0])
ub = np.array([-5,1])
# print("lb_shape:",lb.shape)

X0_star = Star(lb,ub)
# plot_star(X0_star)

R = simReachKrylov(A,X0_star,h, N, m)
print("R:",R)
plot_star(R)
# x0 = np.array([0, 1])
# X = simKrylov(A, x0, h, N,m)
# print("X:",X)

