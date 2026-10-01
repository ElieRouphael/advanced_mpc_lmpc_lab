import numpy as np
import matplotlib.pyplot as plt


def plot_polytope(ax, P, **kwargs):
    V=P.vertices_2d()
    if len(V)==0: return ax
    Vc=np.vstack([V,V[0]])
    ax.plot(Vc[:,0],Vc[:,1],**kwargs)
    return ax


def style_axes(ax, xlabel=None, ylabel=None, title=None):
    if xlabel: ax.set_xlabel(xlabel)
    if ylabel: ax.set_ylabel(ylabel)
    if title: ax.set_title(title)
    ax.grid(True, alpha=0.3)
    return ax
