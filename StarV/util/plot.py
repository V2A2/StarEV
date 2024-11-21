"""
Plot module, contains methods for plotting
Dung Tran, 9/11/2022
"""

from StarV.set.probstar import ProbStar
from StarV.set.star import Star
import numpy as np
import matplotlib.pyplot as plt
import pypoman
import warnings


def getVertices(I):
    """Get all vertices of a star"""
    
    assert isinstance(I, ProbStar) or isinstance(I, Star), 'error: input should be a ProbStar or a Star'
    if len(I.C) == 0:
        lb = I.pred_lb
        ub = I.pred_ub
        A = np.eye(I.nVars)
        C = np.vstack((A, -A))
        d = np.concatenate([ub, -lb])
    else:
        if len(I.pred_lb) == 0:
            [lb, ub] = I.estimateRanges()
        else:
            lb = I.pred_lb
            ub = I.pred_ub
        A = np.eye(I.nVars)
        C1 = np.vstack((A, -A))
        d1 = np.concatenate([ub, -lb])
        C = np.vstack((I.C, C1))
        d = np.concatenate([I.d, d1])

    c = I.V[:,0]
    V = I.V[:,1:I.nVars+1]

    proj = (V, c)
    ineq = (C, d)
    verts = pypoman.projection.project_polytope(proj, ineq)

    return verts


def plot_2D_Star(I, show=True):

    if I.dim != 2:
        raise Exception('Input set is not 2D star')
    verts = getVertices(I)
    try:
        pypoman.plot_polygon(verts)
    except Exception:
        warnings.warn(message='Potential floating-point error')
    if show:
        plt.show()

    

def plot_probstar(I,dir_mat=None,safety_value=None, dir_vec=None, show_prob=True, label=('$x$', '$y$'), show=True):
    """Plot a star set in a specific direction
       y = dir_mat*x + dir_vec, x in I
    """

    if isinstance(I, ProbStar):
        I1 = I.affineMap(dir_mat, dir_vec)
        if I1.dim > 2:
            raise Exception('error: only 2D plot is supported')
        prob = I1.estimateProbability()
        plot_2D_Star(I, show=False)
        l, u = I1.getRanges()
        if show_prob:
            ax = plt.gca()
            ax.text(0.5*(l[0] + u[0]), 0.5*(l[1] + u[1]), str(prob),fontsize = 18)
            ax.set_xlim(l[0], u[1])
            ax.set_ylim(l[0], u[1])

    elif isinstance(I, list):
        L = []
        U = []
        for i in range(0,len(I)):
            I2 = I[i].affineMap(dir_mat, dir_vec)
            if I2.dim > 2:
                raise Exception('error: only 2D plot is supported')
            prob = I2.estimateProbability()
            plot_2D_Star(I2, show=False)
            l, u = I2.getRanges()
            if i==0:
                L = l
                U = u
            else:
                L = np.vstack((L, l))
                U = np.vstack([U, u])
            if show_prob:
                ax = plt.gca()
                ax.text(0.5*(l[0] + u[0]), 0.5*(l[1] + u[1]), str(prob),fontsize=18)

        Lm = L.min(axis=0)
        Um = U.max(axis=0)
        ax = plt.gca()
        ax.set_xlim(Lm[0], Um[0])
        ax.set_ylim(Lm[1], Um[1])
    else:
        raise Exception('error: first input should be a ProbStar or a list of ProbStar')

    plt.xlabel(label[0], fontsize=13)
    plt.ylabel(label[1], fontsize=13)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    if show:
        if safety_value is not None:
            plt.axvline(x = safety_value, color = 'r', linestyle = '-',linewidth=1) 
            plt.text(4.05, plt.gca().get_ylim()[1]*0.3, 'unsafe condition', color='r', fontsize=18)
            # plt.axvline(x = safety_value + 0.04, color = 'r', linestyle = '-',linewidth=1) 
            # plt.axhline(y = safety_value+0.1, color = 'r', linestyle = '-',linewidth=1) 
            # plt.axhline(y = safety_value + 0.25, color = 'r', linestyle = '-',linewidth=1) 
        plt.show()

def plot_star(I, dir_mat=None,safety_value=None, dir_vec=None, label=('$y_1$', '$y_2$'), show=True):
    """Plot a star set in a specific direction

       y = dir_mat*x + dir_vec, x in I
    """

    if isinstance(I, Star):
        I1 = I.affineMap(dir_mat, dir_vec)
        # if I1.dim > 2:
            # raise Exception('error: only 2D plot is supported')
        plot_2D_Star(I, show=False)
        l, u = I1.getRanges()
        
    elif isinstance(I, list):
        L = []
        U = []
        for i in range(0,len(I)):
            I2 = I[i].affineMap(dir_mat, dir_vec)
            if I2.dim > 2:
                raise Exception('error: only 2D plot is supported')
            plot_2D_Star(I2, show=False)
            l, u = I2.getRanges()
            if i==0:
                L = l
                U = u
            else:
                L = np.vstack((L, l))
                U = np.vstack([U, u])
            
        Lm = L.min(axis=0)
        Um = U.max(axis=0)
        ax = plt.gca()
        ax.set_xlim(Lm[0], Um[0])
        ax.set_ylim(Lm[1], Um[1])
    else:
        raise Exception('error: first input should be a ProbStar or a list of ProbStar')

    plt.xlabel(label[0], fontsize=13)
    plt.ylabel(label[1], fontsize=13)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    if safety_value is not None:
        plt.axvline(x = safety_value, color = 'r', linestyle = '--',linewidth=1) 
        plt.text(4, plt.gca().get_ylim()[1] * 0.9, 'unsafe condition', color='r', fontsize=12, verticalalignment='center')
    if show:
        plt.show()



def plot_1D_Star(I, safety_value=None,show=True, color='g'):
    """Plot a 1D star set
    Yuntao Li"""

    if isinstance(I, ProbStar) or isinstance(I, Star):
        if I.dim != 1:
            raise Exception('error: input set is not 1D star')
        # [lb, ub] = getVertices(I)
        [lb, ub] = I.getRanges()
        plt.plot([lb, ub], [0, 0], color=color)
        if show:
            plt.show()
    elif isinstance(I, list) and len(I) > 1:
        for i in range(0, len(I)):
            if I[i].dim != 1:
                raise Exception('error: input set is not 1D star')
            [lb, ub] = I[i].getRanges()
            plt.plot([lb, ub], [0, 0], color=color)
        if safety_value is not None:
                plt.axvline(x = safety_value, color = 'r', linestyle = '--',linewidth=1) 
        if show:
            plt.show()
    elif isinstance(I, list) and len(I) == 1:
        if I[0].dim != 1:
            raise Exception('error: input set is not 1D star')
        [lb, ub] = I[0].getRanges()
        plt.plot([lb, ub], [0, 0], color=color)
        if show:
            plt.show()


def plot_1D_Star_time(I, time_bound,step_size,safety_value=None,show=True, color='g'):
   
    """Plot series 1D star set over time"""

    times = np.arange(0, time_bound + 2*step_size, step_size)
    
    if isinstance(I, list) and len(I) > 1:
        for i in range(0, len(I)):
            if I[i].dim != 1:
                raise Exception('error: input set is not 1D star')
            [lb, ub] = I[i].getRanges()
            plt.plot([times[i], times[i]],[lb, ub], color=color)
            plt.xlabel("Time")
            plt.ylabel("y1")
           
        if show:
            if safety_value is not None:
                plt.axhline(y = safety_value, color = 'r', linestyle = '--',linewidth=1) 
                # plt.text(4, plt.gca().get_ylim()[1] * 0.9, 'unsafe condition', color='r', fontsize=12, verticalalignment='center')

            plt.show()
 
