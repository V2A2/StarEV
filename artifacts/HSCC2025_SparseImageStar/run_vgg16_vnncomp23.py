"""
Verify vgg16 (VNNCOMP2023) against infinity norm
Author: Sung Woo Choi
Date: 06/24/2024
"""

import scipy
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
import pickle
import re

import os
import sys
sys.path.append(os.path.abspath(os.curdir))

from StarV.set.imagestar import ImageStar
from StarV.set.sparseimagestar2dcoo import SparseImageStar2DCOO
from StarV.set.sparseimagestar2dcsr import SparseImageStar2DCSR
from StarV.verifier.certifier import certifyRobustness
from StarV.util.load import *
from StarV.util.vnnlib import *

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split(r'(\d+)', text)]

def verify_vgg16_network(dtype='float64'):

    print('=================================================================================')
    print(f"Verification of VGG16 Network against Infinity Norm Attack")
    print('=================================================================================\n')

    folder_dir = f"StarV/util/data/nets/vggnet16"
    net_dir = f"{folder_dir}/onnx/vgg16-7.onnx"
    num_inputs, num_outputs, inp_dtype = get_num_inputs_outputs(net_dir)

    # loading DNNs into StarV network
    starvNet = load_neural_network_file(net_dir, dtype=dtype, channel_last=False, in_shape=None, sparse=False, show=False)
    print()
    print(starvNet.info())


    shape = (3, 224, 224)
    
    # VNNLIB_FILE = 'vnncomp2023_benchmarks/benchmarks/vggnet16/vnnlib/spec0_pretzel.vnnlib'
    vnnlib_dir = f"{folder_dir}/vnnlib"
    vnnlib_files = [f for f in os.listdir(vnnlib_dir) if f.endswith('.vnnlib')]
    vnnlib_files.sort(key = natural_keys)

    N = len(vnnlib_files)
    rbIM = np.zeros(N)
    vtIM = np.zeros(N)
    rbCSR = np.zeros(N)
    vtCSR = np.zeros(N)
    rbCOO = np.zeros(N)
    vtCOO = np.zeros(N)
    numPred = np.zeros(N)

    rb_table = []
    vt_table = []
 
    print(f"\n\nVerifying vggnet16 with ImageStar")
    for i, vnnlib_file in enumerate(vnnlib_files):
        vnnlib_file_dir = f"{vnnlib_dir}/{vnnlib_file}"

        with open(vnnlib_file_dir) as f:
            first_line = f.readline().strip('\n')
        label = int(re.findall(r'\b\d+\b', first_line)[0])

        vnnlib_rv = read_vnnlib_simple(vnnlib_file_dir, num_inputs, num_outputs)

        box, spec_list = vnnlib_rv[0]
        bounds = np.array(box, dtype=dtype)
        # transpose from [C, H, W] to [H, W, C]
        lb = bounds[:, 0].reshape(shape).transpose([1, 2, 0])
        ub = bounds[:, 1].reshape(shape).transpose([1, 2, 0])

        num_attack_pixel = (lb != ub).sum()
        print(f"\nVerifying {vnnlib_file} with {num_attack_pixel} attacked pixels")

        if num_attack_pixel > 50:
            print(f"Skipping {vnnlib_file} to avoid RAM issue")
            rbIM[i] = np.nan
            vtIM[i] = np.nan
        else:
            IM = ImageStar(lb, ub)
            rbIM[i], vtIM[i], _, _ = certifyRobustness(net=starvNet, inputs=IM, labels=label,
                veriMethod='BFS', reachMethod='approx', lp_solver='gurobi', pool=None, 
                RF=0.0, DR=0, return_output=False, show=False)
            
            if rbIM[i] == 1:
                print(f"ROBUSTNESS RESULT: ROBUST")
            elif rbIM[i] == 2:
                print(f"ROBUSTNESS RESULT: UNKNOWN")
            elif rbIM[i] == 0:
                print(f"ROBUSTNESS RESULT: UNROBUST")

            print(f"VERIFICATION TIME: {vtIM[i]}")

    rb_table.append((rbIM == 1).sum())
    vt_table.append((vtIM.sum() / N))
    del IM

    print(f"\n\nVerifying vggnet16 with SparseImageStar in CSR format")
    for i, vnnlib_file in enumerate(vnnlib_files):
        vnnlib_file_dir = f"{vnnlib_dir}/{vnnlib_file}"

        with open(vnnlib_file_dir) as f:
            first_line = f.readline().strip('\n')
        label = int(re.findall(r'\b\d+\b', first_line)[0])


        vnnlib_rv = read_vnnlib_simple(vnnlib_file_dir, num_inputs, num_outputs)

        box, spec_list = vnnlib_rv[0]
        bounds = np.array(box, dtype=inp_dtype)
        # transpose from [C, H, W] to [H, W, C]
        lb = bounds[:, 0].reshape(shape).transpose([1, 2, 0]).astype(dtype)
        ub = bounds[:, 1].reshape(shape).transpose([1, 2, 0]).astype(dtype)

        num_attack_pixel = (lb != ub).sum()
        print(f"\nVerifying {vnnlib_file} with {num_attack_pixel} attacked pixels")

        if num_attack_pixel > 150:
            print(f"Skipping {vnnlib_file} to avoid RAM issue")
            rbCSR[i] = np.nan
            vtCSR[i] = np.nan
        else:
            CSR = SparseImageStar2DCSR(lb, ub)
            rbCSR[i], vtCSR[i], _, Y = certifyRobustness(net=starvNet, inputs=CSR, labels=label,
                veriMethod='BFS', reachMethod='approx', lp_solver='gurobi', pool=None, 
                RF=0.0, DR=0, return_output=False, show=False)
            numPred[i] = Y.num_pred
        
            if rbCSR[i] == 1:
                print(f"ROBUSTNESS RESULT: ROBUST")
            elif rbCSR[i] == 2:
                print(f"ROBUSTNESS RESULT: UNKNOWN")
            elif rbCSR[i] == 0:
                print(f"ROBUSTNESS RESULT: UNROBUST")

            print(f"VERIFICATION TIME: {vtCSR[i]}")
            print(f"NUM_PRED: {numPred[i]}")

    rb_table.append((rbCSR == 1).sum())
    vt_table.append((vtCSR.sum() / N))
    del CSR, Y

    print(f"\n\nVerifying vggnet16 with SparseImageStar in COO format")
    for i, vnnlib_file in enumerate(vnnlib_files):
        vnnlib_file_dir = f"{vnnlib_dir}/{vnnlib_file}"

        with open(vnnlib_file_dir) as f:
            first_line = f.readline().strip('\n')
        label = int(re.findall(r'\b\d+\b', first_line)[0])


        vnnlib_rv = read_vnnlib_simple(vnnlib_file_dir, num_inputs, num_outputs)

        box, spec_list = vnnlib_rv[0]
        bounds = np.array(box, dtype=inp_dtype)
        # transpose from [C, H, W] to [H, W, C]
        lb = bounds[:, 0].reshape(shape).transpose([1, 2, 0]).astype(dtype)
        ub = bounds[:, 1].reshape(shape).transpose([1, 2, 0]).astype(dtype)

        num_attack_pixel = (lb != ub).sum()
        print(f"\nVerifying {vnnlib_file} with {num_attack_pixel} attacked pixels")

        if num_attack_pixel > 150:
            print(f"Skipping {vnnlib_file} to avoid RAM issue")
            rbCOO[i] = np.nan
            vtCOO[i] = np.nan
        else:
            COO = SparseImageStar2DCOO(lb, ub)
            rbCOO[i], vtCOO[i], _, _ = certifyRobustness(net=starvNet, inputs=COO, labels=label,
                veriMethod='BFS', reachMethod='approx', lp_solver='gurobi', pool=None, 
                RF=0.0, DR=0, return_output=False, show=False)
            
            if rbCOO[i] == 1:
                print(f"ROBUSTNESS RESULT: ROBUST")
            elif rbCOO[i] == 2:
                print(f"ROBUSTNESS RESULT: UNKNOWN")
            elif rbCOO[i] == 0:
                print(f"ROBUSTNESS RESULT: UNROBUST")

            print(f"VERIFICATION TIME: {vtCOO[i]}")

    rb_table.append((rbCOO == 1).sum())
    vt_table.append((vtCOO.sum() / N))
    del COO

    # save verification results
    path = f"artifacts/HSCC2025_SparseImageStar/results"
    if not os.path.exists(path):
        os.makedirs(path)

    save_file = path + f"/vggnet16_vnncomp23_results.pkl"
    pickle.dump([rbIM, vtIM, rbCSR, vtCSR, rbCOO, vtCOO, numPred], open(save_file, "wb"))

    headers = [f"ImageStar", f"SIM_CSR", f"SIM_COO"]

    # Robustness Resluts
    print('-----------------------------------------------------')
    print('Robustness')
    print('-----------------------------------------------------')
    print(tabulate([rb_table], headers=headers))
    print()

    Tlatex = tabulate([rb_table], headers=headers, tablefmt='latex')
    with open(path+f"vggnet16_vnncomp23_results_rb.tex", "w") as f:
        print(Tlatex, file=f)

    # Verification Time Results
    print('-----------------------------------------------------')
    print('Verification Time')
    print('-----------------------------------------------------')
    print(tabulate([vt_table], headers=headers))
    print()

    Tlatex = tabulate([vt_table], headers=headers, tablefmt='latex')
    with open(path+f"vggnet16_vnncomp23_results_vt.tex", "w") as f:
        print(Tlatex, file=f)

    print('=====================================================')
    print('DONE!')
    print('=====================================================')


def verify_vgg16_converted_network(dtype='float64'):

    print('=================================================================================')
    print(f"Verification of VGG16 Converted Network against Infinity Norm Attack")
    print('=================================================================================\n')

    folder_dir = f"StarV/util/data/nets/vggnet16"
    net_dir = f"{folder_dir}/onnx/vgg16-7_converted.onnx"
    num_inputs, num_outputs, inp_dtype = get_num_inputs_outputs(net_dir)

    # loading DNNs into StarV network
    starvNet = load_neural_network_file(net_dir, dtype=dtype, channel_last=False, in_shape=None, sparse=False, show=False)
    print()
    print(starvNet.info())


    shape = (3, 224, 224)
    
    # VNNLIB_FILE = 'vnncomp2023_benchmarks/benchmarks/vggnet16/vnnlib/spec0_pretzel.vnnlib'
    vnnlib_dir = f"{folder_dir}/vnnlib"
    vnnlib_files = [f for f in os.listdir(vnnlib_dir) if f.endswith('.vnnlib')]
    vnnlib_files.sort(key = natural_keys)

    N = len(vnnlib_files)
    rbIM = np.zeros(N)
    vtIM = np.zeros(N)

    print(f"\n\nVerifying vggnet16 with ImageStar")
    for i, vnnlib_file in enumerate(vnnlib_files):
        vnnlib_file_dir = f"{vnnlib_dir}/{vnnlib_file}"

        with open(vnnlib_file_dir) as f:
            first_line = f.readline().strip('\n')
        label = int(re.findall(r'\b\d+\b', first_line)[0])

        vnnlib_rv = read_vnnlib_simple(vnnlib_file_dir, num_inputs, num_outputs)

        box, spec_list = vnnlib_rv[0]
        bounds = np.array(box, dtype=dtype)
        # transpose from [C, H, W] to [H, W, C]
        lb = bounds[:, 0].reshape(shape).transpose([1, 2, 0])
        ub = bounds[:, 1].reshape(shape).transpose([1, 2, 0])

        num_attack_pixel = (lb != ub).sum()
        print(f"\nVerifying {vnnlib_file} with {num_attack_pixel} attacked pixels")

        if num_attack_pixel > 50:
            print(f"Skipping {vnnlib_file} to avoid RAM issue")
            rbIM[i] = np.nan
            vtIM[i] = np.nan
        else:
            IM = ImageStar(lb, ub)
            rbIM[i], vtIM[i], _, _ = certifyRobustness(net=starvNet, inputs=IM, labels=label,
                veriMethod='BFS', reachMethod='approx', lp_solver='gurobi', pool=None, 
                RF=0.0, DR=0, return_output=False, show=False)
            
            if rbIM[i] == 1:
                print(f"ROBUSTNESS RESULT: ROBUST")
            elif rbIM[i] == 2:
                print(f"ROBUSTNESS RESULT: UNKNOWN")
            elif rbIM[i] == 0:
                print(f"ROBUSTNESS RESULT: UNROBUST")

            print(f"VERIFICATION TIME: {vtIM[i]}")
    del IM

    # save verification results
    path = f"artifacts/HSCC2025_SparseImageStar/results"
    if not os.path.exists(path):
        os.makedirs(path)

    save_file = path + f"/vggnet16_vnncomp23_converted_results.pkl"
    pickle.dump([rbIM, vtIM], open(save_file, "wb"))

    headers = [f"ImageStar"]

    # Robustness Resluts
    print('-----------------------------------------------------')
    print('Robustness')
    print('-----------------------------------------------------')
    print(tabulate([np.arange(N), rbIM], headers=headers))
    print()

    Tlatex = tabulate([np.arange(N), rbIM], headers=headers, tablefmt='latex')
    with open(path+f"vggnet16_vnncomp23_converted_results_rb.tex", "w") as f:
        print(Tlatex, file=f)

    # Verification Time Results
    print('-----------------------------------------------------')
    print('Verification Time')
    print('-----------------------------------------------------')
    print(tabulate([np.arange(N), vtIM], headers=headers))
    print()

    Tlatex = tabulate([np.arange(N), vtIM], headers=headers, tablefmt='latex')
    with open(path+f"vggnet16_vnncomp23_converted_results_vt.tex", "w") as f:
        print(Tlatex, file=f)

    print('=====================================================')
    print('DONE!')
    print('=====================================================')


def verify_vgg16_network_spec_cn(dtype='float64'):

    print('=================================================================================')
    print(f"Verification of VGG16 Network against Infinity Norm Attack Spec_cn")
    print('=================================================================================\n')

    folder_dir = f"StarV/util/data/nets/vggnet16"
    net_dir = f"{folder_dir}/onnx/vgg16-7.onnx"
    num_inputs, num_outputs, inp_dtype = get_num_inputs_outputs(net_dir)

    # loading DNNs into StarV network
    starvNet = load_neural_network_file(net_dir, dtype=dtype, channel_last=False, in_shape=None, sparse=False, show=False)
    print()
    print(starvNet.info())


    shape = (3, 224, 224)
    
    # VNNLIB_FILE = 'vnncomp2023_benchmarks/benchmarks/vggnet16/vnnlib/spec_cn/spec_c0_corn_atk200.vnnlib'
    vnnlib_dir = f"{folder_dir}/vnnlib/spec_cn"
    vnnlib_files = [f for f in os.listdir(vnnlib_dir) if f.endswith('.vnnlib')]
    vnnlib_files.sort(key = natural_keys)

    # save verification results
    path = f"artifacts/HSCC2025_SparseImageStar/results"
    if not os.path.exists(path):
        os.makedirs(path)

    save_file = path + f"/vggnet16_vnncomp23_spec_cn_results.pkl"

    N = len(vnnlib_files)
    rbCSR = np.zeros(N)
    vtCSR = np.zeros(N)
    rbCOO = np.zeros(N)
    vtCOO = np.zeros(N)
    numPred = np.zeros(N)

    show = True
 
    print(f"\n\nVerifying vggnet16 with SparseImageStar in CSR format")
    for i, vnnlib_file in enumerate(vnnlib_files):
        vnnlib_file_dir = f"{vnnlib_dir}/{vnnlib_file}"

        with open(vnnlib_file_dir) as f:
            first_line = f.readline().strip('\n')
        label = int(re.findall(r'\b\d+\b', first_line)[0])

        print(f"\n Loading a VNNLIB file")
        vnnlib_rv = read_vnnlib_simple(vnnlib_file_dir, num_inputs, num_outputs)

        box, spec_list = vnnlib_rv[0]
        bounds = np.array(box, dtype=inp_dtype)
        # transpose from [C, H, W] to [H, W, C]
        lb = bounds[:, 0].reshape(shape).transpose([1, 2, 0]).astype(dtype)
        ub = bounds[:, 1].reshape(shape).transpose([1, 2, 0]).astype(dtype)

        num_attack_pixel = (lb != ub).sum()
        print(f"\nVerifying {vnnlib_file} with {num_attack_pixel} attacked pixels")

        CSR = SparseImageStar2DCSR(lb, ub)
        del lb, ub, bounds

        rbCSR[i], vtCSR[i], _, Y = certifyRobustness(net=starvNet, inputs=CSR, labels=label,
            veriMethod='BFS', reachMethod='approx', lp_solver='gurobi', pool=None, 
            RF=0.0, DR=0, return_output=False, show=show)
        numPred[i] = Y.num_pred
    
        if rbCSR[i] == 1:
            print(f"ROBUSTNESS RESULT: ROBUST")
        elif rbCSR[i] == 2:
            print(f"ROBUSTNESS RESULT: UNKNOWN")
        elif rbCSR[i] == 0:
            print(f"ROBUSTNESS RESULT: UNROBUST")

        print(f"VERIFICATION TIME: {vtCSR[i]}")
        print(f"NUM_PRED: {numPred[i]}")
        pickle.dump([numPred, rbCSR, vtCSR, rbCOO, vtCOO], open(save_file, "wb"))
    del CSR, Y

    print(f"\n\nVerifying vggnet16 with SparseImageStar in COO format")
    for i, vnnlib_file in enumerate(vnnlib_files):
        vnnlib_file_dir = f"{vnnlib_dir}/{vnnlib_file}"

        with open(vnnlib_file_dir) as f:
            first_line = f.readline().strip('\n')
        label = int(re.findall(r'\b\d+\b', first_line)[0])

        print(f"\n Loading a VNNLIB file")
        vnnlib_rv = read_vnnlib_simple(vnnlib_file_dir, num_inputs, num_outputs)

        box, spec_list = vnnlib_rv[0]
        bounds = np.array(box, dtype=inp_dtype)
        # transpose from [C, H, W] to [H, W, C]
        lb = bounds[:, 0].reshape(shape).transpose([1, 2, 0]).astype(dtype)
        ub = bounds[:, 1].reshape(shape).transpose([1, 2, 0]).astype(dtype)

        num_attack_pixel = (lb != ub).sum()
        print(f"\nVerifying {vnnlib_file} with {num_attack_pixel} attacked pixels")

        COO = SparseImageStar2DCOO(lb, ub)
        del lb, ub, bounds

        rbCOO[i], vtCOO[i], _, _ = certifyRobustness(net=starvNet, inputs=COO, labels=label,
            veriMethod='BFS', reachMethod='approx', lp_solver='gurobi', pool=None, 
            RF=0.0, DR=0, return_output=False, show=show)
        
        if rbCOO[i] == 1:
            print(f"ROBUSTNESS RESULT: ROBUST")
        elif rbCOO[i] == 2:
            print(f"ROBUSTNESS RESULT: UNKNOWN")
        elif rbCOO[i] == 0:
            print(f"ROBUSTNESS RESULT: UNROBUST")

        print(f"VERIFICATION TIME: {vtCOO[i]}")
        pickle.dump([numPred, rbCSR, vtCSR, rbCOO, vtCOO], open(save_file, "wb"))

    pickle.dump([numPred, rbCSR, vtCSR, rbCOO, vtCOO], open(save_file, "wb"))

    headers = [f"SIM_csr, SIM_coo"]

    # Robustness Resluts
    print('-----------------------------------------------------')
    print('Robustness')
    print('-----------------------------------------------------')
    print(tabulate([np.arange(N), rbCSR, rbCOO], headers=headers))
    print()

    # Verification Time Results
    print('-----------------------------------------------------')
    print('Verification Time')
    print('-----------------------------------------------------')
    print(tabulate([np.arange(N), vtCSR, vtCOO], headers=headers))
    print()

    print('=====================================================')
    print('DONE!')
    print('=====================================================')


def plot_table_vgg16_network():
    folder_dir = f"artifacts/HSCC2025_SparseImageStar/results"
    file_dir = folder_dir + 'vggnet16_vnncomp23_results.pkl'
    with open(file_dir, 'rb') as f:
        rbIM, vtIM, rbCSR, vtCSR, rbCOO, vtCOO, num_pred = pickle.load(f)
    file_dir = folder_dir + 'vggnet16_vnncomp23_converted_results.pkl'
    with open(file_dir, 'rb') as f:
        rbIMc, vtIMc = pickle.load(f)

    nnv_dir = 'StarV/util/data/nets/vggnet16/nnv'
    mat_file = scipy.io.loadmat(f"{nnv_dir}/nnv_vggnet16_results.mat")
    rbNNV = mat_file['rb_im'].ravel()
    vtNNV = mat_file['vt_im'].ravel()

    mat_file = scipy.io.loadmat(f"{nnv_dir}/nnv_vggnet16_converted_results.mat")
    rbNNVc = mat_file['rb_im'].ravel()
    vtNNVc = mat_file['vt_im'].ravel()

    file_dir = folder_dir + 'vggnet16_vnncomp23_spec_cn_results.pkl'
    with open(file_dir, 'rb') as f:
        numPred_cn, rbCSR_cn, vtCSR_cn, rbCOO_cn, vtCOO_cn = pickle.load(f)

    f_dir = f"StarV/util/data/nets/vggnet16"
    net_dir = f"{f_dir}/onnx/vgg16-7.onnx"
    num_inputs, num_outputs, inp_dtype = get_num_inputs_outputs(net_dir)
    vnnlib_dir = f"{f_dir}/vnnlib"
    vnnlib_files = [f for f in os.listdir(vnnlib_dir) if f.endswith('.vnnlib')]
    vnnlib_files.sort(key = natural_keys)
    shape = (3, 224, 224)

    num_attack_pixel = []
    for i, vnnlib_file in enumerate(vnnlib_files):
        vnnlib_file_dir = f"{vnnlib_dir}/{vnnlib_file}"

        with open(vnnlib_file_dir) as f:
            first_line = f.readline().strip('\n')
        label = int(re.findall(r'\b\d+\b', first_line)[0])

        vnnlib_rv = read_vnnlib_simple(vnnlib_file_dir, num_inputs, num_outputs)

        box, spec_list = vnnlib_rv[0]
        bounds = np.array(box, dtype='float32')
        # transpose from [C, H, W] to [H, W, C]
        lb = bounds[:, 0].reshape(shape).transpose([1, 2, 0])
        ub = bounds[:, 1].reshape(shape).transpose([1, 2, 0])
        num_attack_pixel.append(int((lb != ub).sum())) 


    N = 15
    vt_NNENUM = [3.5, 3.4, 9.3, 4.8, 18.1, 35.7, 6.5, 18.3, 133.8, 10.6, 40.9, 57.6, 'T/O', 236.5, 746.6]
    vt_DP = 'O/M'
    vt_marabou = 'T/O'
    vt_bcrown = [7.355725526809692, 8.868661165237427, 8.908552885055542, 9.075981855392456, 8.986030578613281, 8.999144315719604, 8.916476249694824, 9.294207572937012, 10.620023727416992, 9.017800092697144, 9.108751058578491, 9.2491958141326, 594.9671733379364, 17.784186124801636, 34.14556264877319]
    vt_bcrown = np.array(vt_bcrown, dtype='float64')
    vt_abcrown = [302.8435814380646, 243.49199199676514, 174.6395332813263, 622.3142883777618, 430.933091878891, 622.221896648407, 664.8663415908813, 709.2889895439148, 708.833279132843, 893.600474357605, 897.9993720054626, 860.9506402015686, 945.6725194454193, 1077.005056142807, 1191.9225597381592]

    headers = ['Specs', 'e', 'Result', 'm', 'IM', 'SIM_csr', 'SIM_coo', 'NNV', 'DeepPoly',  'Marabou', 'IM', 'NNV', 'NNENUM', 'ab-CROWN', 'b-CROWN']

    result = 'UNSAT'
    
    data = []
    for i in range(N):
        vt_im = 'O/M' if np.isnan(vtIM[i]) else f"{vtIM[i]:0.1f}"
        vt_imc = 'O/M' if np.isnan(vtIMc[i]) else f"{vtIMc[i]:0.1f}"
        vt_nnv = 'O/M' if vtNNV[i] < 0 else f"{vtNNV[i]:0.1f}"
        vt_nnvc = 'O/M' if vtNNVc[i] < 0 else f"{vtNNVc[i]:0.1f}"
        
        nPred = 'NA' if np.isnan(vtCSR[i]) else f"{num_pred[i]}"
        data.append([i, num_attack_pixel[i], result, nPred,  vt_im, f"{vtCSR[i]:0.1f}", f"{vtCOO[i]:0.1f}", vt_nnv, vt_DP, vt_marabou, vt_imc, vt_nnvc, vt_NNENUM[i], f"{vt_abcrown[i]:0.1f}", f"{vt_bcrown[i]:0.1f}"])

    num_attack_pixel_cn = [200, 300, 400, 500, 1000, 2000, 3000]
    N_cn = len(numPred_cn)
    vt_NNENUM_cn = [744.02, 1060.96, 1354.75, 1781.26, 'T/O', 'T/O', 'O/M']
    vt_bcrown_cn = [26782.327130317688, 37052.68477010727, 'T/O', 'T/O', 'T/O', 'T/O', 'T/O']
    vt_abcrown_cn = 'T/O'
    for i in range(N_cn):
        vt_im = 'O/M' 
        vt_imc = 'O/M'
        vt_nnv = 'O/M'
        vt_nnvc = 'O/M'
        vt_bcrown_cd = vt_bcrown_cn[i] if vt_bcrown_cn[i] == 'T/O' else f"{np.array(vt_bcrown_cn[i], dtype='float64'):0.1f}"

        nPred = 'NA' if np.isnan(vtCSR_cn[i]) else f"{numPred_cn[i]}"
        data.append([f"c_{i}", num_attack_pixel_cn[i], result, nPred,  vt_im, f"{vtCSR_cn[i]:0.1f}", f"{vtCOO_cn[i]:0.1f}", vt_nnv, vt_DP, vt_marabou, vt_imc, vt_nnvc, vt_NNENUM_cn[i], vt_abcrown_cn, vt_bcrown_cd])

    print(tabulate(data, headers=headers))

    Tlatex = tabulate(data, headers=headers, tablefmt='latex')
    with open(folder_dir+f"Table_4_vggnet16_vnncomp23_results.tex", "w") as f:
        print(Tlatex, file=f)


if __name__ == "__main__":
    verify_vgg16_network(dtype='float64')
    verify_vgg16_converted_network(dtype='float64')
    verify_vgg16_network_spec_cn()
    plot_table_vgg16_network()