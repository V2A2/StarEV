# from scipy.sparse.linalg import norm 
import numpy as np
np.set_printoptions(precision=4)
from scipy.linalg import expm
from numpy.linalg import norm
from StarV.set.star import Star
import copy



def simKrylov(A,x0,t,N,m):
    """
    Simulate the time evolution of a linear dynamical system using the Krylov subspace method.

    Args:
    - A: The input matrix representing the system dynamics.
    - x0: The initial state vector.
    - t: The time step for simulation.
    - N: The number of steps to simulate.
    - m: The number of basic vectors for the Krylov subspace Km, Vm = [v0, .., vm-1], number of iterations in othonoral basis medhod.

    Returns:
    - X: Simulation result as a matrix X = [x0, x1, ..., xN].
    """
    m = A.shape[0] # get the number of rows(dims) of A
    Im = np.eye(m)
    e1 = Im[:, 0]

    X = np.zeros((m, N+1)) # create all zero matrix and store initial x0
    X[:, 0] = x0 # store initial x0

    # approximate e^At * x0 = x0_norm * V * e^Ht * e1
    if norm(x0, ord=2) > 0:
        Vm, Hm = arnoldi(A, x0,m)

        x0_norm = norm(x0,ord=2)
        V = x0_norm * Vm
        H = t * Hm
        for i in range(1, N+1): # calculate e^At * x0 for each time step
            exmp = expm(i * H)
            X[:, i] = V @ exmp @ e1

    # print("X:",X)
    # X[:, 0] = x0
    return X


def arnoldi(A, x0, m=None):
    """
    Arnoldi iteration to compute orthonormal basis for Krylov subspace.
    Projecting the initial n-dimensional state onto the smaller, m-dimensional Krylov subspace, 
    Computing the matrix exponential using the projected linear transformation Hm ,    

    Args:
    - A: n x n input matrix. 
    - x0: normalized n x 1 initial vector.
    - m: The number of basic vectors for the (m-dims) Krylov subspace Km, m iterations in arnoldi method, user defined.

    Returns:
    - Vm: n x m orthonormal basis matrix.
    - Hm: m x m upper Hessenberg matrix, is also symmetirc and tridiagonal.
    """

    # [A, x0, m] = copy.deepcopy(args)

    n = A.shape[0] # A matrix dims

    if m is None:
        m = n
    x0 = x0 / norm(x0,ord=2) # normalized n x 1 init vectoer x0
    x0 = x0.reshape(-1)  # reshape x0 to column vector

    Hm = np.zeros((m, m)) # initial Hm : m x m hessenberg matrix
    Vm = np.zeros((n, m)) # initial Vm : n x m orthonormal basis vectors
    Vm[:,0] = x0


    for i in range(m):

        wi = A @ Vm[:, i]  # multiplying current vector by A
        
        for j in range(i+1): # projecting out the previous rothonormal directions from current direction
            Vm_T = np.transpose(Vm[:, j])
            Hm[j, i] = Vm_T @ wi
            wi = wi - Hm[j, i] * Vm[:, j]
        if i < n-1:
            if i + 1 < m: 
                Hm[i+1, i] = norm(wi,ord=2) # normalize current vector, and adding it to the list of orthonormal directions  Vm[:, i+1]
                if Hm[i+1, i] == 0: # check if the norm computed above  Hm[i+1, i] is ever zero, the loop can terminate early and the approximation will be exact.
                    break
                Vm[:, i+1] = wi/ Hm[i+1, i] 

    Vm = Vm[:,:m]
    Hm = Hm[:m,:m]

    return Vm, Hm

def lanczos(A, x0, m=None):
    """
    Lanczos iteration to compute orthonormal basis for Krylov subspace when for large dimension system matrix, which is both sparse and symmetric.
    Projecting the initial n-dimensional state onto the smaller, m-dimensional Krylov subspace, 
    Computing the matrix exponential using the projected linear transformation Hm,    

    Args:
    - A: n x n input matrix, A = A^T
    - x0: normalized n x 1 initial vector.
    - m: The number of basic vectors for the (m-dims) Krylov subspace Km, m iterations in arnoldi method, user defined.

    Returns:
    - Vm: n x m orthonormal basis matrix.
    - Hm: m x m upper Hessenberg matrix, is also symmtric and tridiagonal.
    """

    # [A, x0, m] = copy.deepcopy(args)

    n = A.shape[0] # A matrix dims

    if m is None:
        m = n
    x0 = x0 / norm(x0,ord=2) # normalized n x 1 init vectoer x0
    x0 = x0.reshape(-1)  # reshape x0 to column vector
    # x0 = np.round(x0, decimals=4)

    Hm = np.zeros((m, m)) # initial Hm : m x m hessenberg matrix
    Vm = np.zeros((n, m)) # initial Vm : n x m orthonormal basis vectors
    Vm[:,0] = x0

    for i in range(m):

        wi = A @ Vm[:, i] 

        if i > 0: # projecting out the two previous orthonormal directions from the current vector, and oly one doy product needed.
            Hm[i-1,i] = Hm[i,i-1]
            wi = wi - Hm[i-1,i] * Vm[:,i-1]
        Vm_T = np.transpose(Vm[:, i])
        Hm[i, i] = Vm_T @ wi
        wi = wi - Hm[i, i] * Vm[:, i]
        if i < n-1:
            if i + 1 < m:
                Hm[i+1, i] = norm(wi,ord=2)
                if Hm[i+1, i] == 0:
                    break
                Vm[:, i+1] = wi/ Hm[i+1, i]

    Vm = Vm[:,:m]
    Hm = Hm[:m,:m]

    return Vm, Hm


# simulation-based reachability analysis for dot{x} = Ax using
# Krylov subspace method and star set
def simReachKrylov(A, X0_star, h, N, m):
    """
    Simulate reachable set using the Krylov subspace method.

    Args:
    - A: System matrix.
    - X0: Initial set of states (a Star set).
    - h: Time step for simulation.
    - N: Number of steps.
    - m: The number of basic vectors for Krylov subspace Km.Vm = [v0, .., vm-1]

    Returns:
    - R: Reachable set.
    """
    k = X0_star.V.shape[1]  # number of basic vectors
    dim =  X0_star.V.shape[0]  # dimension of input Star

    Z = []

    for i in range(k):
        X = simKrylov(A, X0_star.V[:, i], h, N, m) 
        Z.append(X)

    R = []

    for i in range(N+1):
        V =[]
        V = np.vstack(Z[j][:,i] for j in range(k))
        V= V.T
        R.append(Star(V, X0_star.C, X0_star.d, X0_star.pred_lb, X0_star.pred_ub)) # R = N+1 Star sets

    return R

