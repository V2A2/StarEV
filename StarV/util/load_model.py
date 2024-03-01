"""
load module, load iss, heli28.MNA5,build model.
Qing Liu, 02/23/2024
"""
import os
from scipy.io import loadmat
from StarV.plant.lode import LODE
import numpy as np
import torch
import math
from scipy.sparse import csc_matrix

def load_harmonic_oscillator_model():
    """Load LODE harmonic oscillator model"""

    # model: x' = y + u1, y' = -x + u2
    # input range: u1, u2 is in [-0.5, 0.5]
    # initial conditions: x in [-6, -5], y in [0, 1]
    # ref: Bak2017CAV: Simulation-Equivalent Reachability of Large Linear Systems with Inputs
    
    A = np.array([[0., 1.], [-1., 0]])  # system matrix
    B = np.eye(2)

    lb = np.array([-6., 0.])
    ub = np.array([-5., 1.])

    input_lb = np.array([-0.5, -0.5])
    input_ub = np.array([0.5, 0.5])

    plant = LODE(A, B)

    return plant, lb, ub, input_lb, input_ub

def convert_to_numpy(matrix):
    if matrix is None:
        return None
    elif isinstance(matrix, csc_matrix):
        return matrix.toarray()
    else:
        return matrix

def load_building_model():
    """Load LODE building model"""
    cur_path = os.path.dirname(__file__)
    # print("cur_path1:",cur_path)
    cur_path = cur_path + '/data/lodes/build.mat' 
    # print("cur_path2:",cur_path)
    mat_contents = loadmat(cur_path)
    # print('load success')
    A = mat_contents['A']
    # print("A_type:",type(A))
    B = mat_contents['B']
    # print("B_type:",type(B))
    # print("B:",B)
    C = mat_contents['C']
    # print("C:",C)

    plant = LODE(A.toarray(), B,C)
    # print("print_plant:",plant.info())
    return plant, A, B, C

def load_iss_model():
    """Load LODE International State Space Model"""
    cur_path = os.path.dirname(__file__)
    # print("cur_path1:",cur_path)
    cur_path = cur_path + '/data/lodes/iss.mat' 
    # print("cur_path2:",cur_path)
    mat_contents = loadmat(cur_path)
    # print('load success')
    A = mat_contents['A']
    # print("A_type",type(A))
    B = mat_contents['B']
    # print("B_type:",type(B))
    # print("B:",B)
    C = mat_contents['C']
    # print("C":,C)
    plant = LODE(A.toarray(), B.toarray(),C.toarray())
    # print("print_plant:",plant.info())
    return plant, A, B, C


def load_helicopter_model():
    """Load LODE helicopter model"""
    cur_path = os.path.dirname(__file__)
    # print("cur_path1:",cur_path)
    cur_path = cur_path + '/data/lodes/heli28.mat' 
    # print("cur_path2:",cur_path)
    mat_contents = loadmat(cur_path)
    # print('load success')
    A = mat_contents['A']
    # print("A_type:",type(A))
    # print("A:",A)
    A = convert_to_numpy(A)
    plant = LODE(A)
    # print("print_plant:",plant.info())
    return plant, A
   

def load_MNA5_model():
    """Load LODE MNA5 model"""
    cur_path = os.path.dirname(__file__)
    # print("cur_path1:",cur_path)
    cur_path = cur_path + '/data/lodes/MNA_5.mat' 
    # print("cur_path2:",cur_path)
    mat_contents = loadmat(cur_path)
    # print('load success')
    A = mat_contents['A']
    B = mat_contents['B']
    # print("A_type:",type(A))
    # print("A:",A)
    A = convert_to_numpy(A)
    # print("A_type:",type(A))
    # print("A:",A)
    B = convert_to_numpy(B)
    # print("B:",B)
    plant = LODE(A,B)
    # plant.compute_gA_gB(dt=0.1)
    # print('\ngA = {}'.format(plant.gA))
    # print('\ngB = {}'.format(plant.gB))

    return plant, A,B
