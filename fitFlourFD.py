import numpy as np
import pandas as pd
from skfda.representation.basis import BSplineBasis
from skfda.representation.grid import FDataGrid
from matplotlib import pyplot as plt
import skewfunHDDC as tfun
from sklearn import metrics as met
import time

def fitFlourFD(cutoffx = 20):
    """
    Description
    -----------
    This function fits a B-spline basis to the flour data, creating functional data objects
    for three different ranges.

    Parameters
    ----------
    cutoffx: int, default=20
        The cutoff point for the first range. Must be an even number.


    Returns
    -------
    dict
        A dictionary containing three functional data objects (`fdx`, `fdy`, `fdfull`)
        and the grouping information (`groupd`).
    """

    x = pd.read_csv("flourx.csv")
    y = pd.read_csv("floury.csv")
    full = pd.read_csv("flourfull.csv")

    labels = pd.read_csv("flourgroupd.csv")
    ncurves = x.shape[0]
    nsplines = y.shape[1]

    bbasis_x = BSplineBasis(n_basis=nsplines, domain_range = (0, cutoffx))
    bbasis_y = BSplineBasis(n_basis=nsplines, domain_range=(cutoffx+2, 480))
    bbasis_fullx = BSplineBasis(n_basis=nsplines, domain_range=(0, 480))

    argvals_x = np.linspace(0, cutoffx)
    argvals_y = np.linspace(cutoffx+2, 480)
    argvals_full = np.linspace(0, 480)

    evalx = bbasis_x(argvals_x)[:,:,0]
    evaly = bbasis_y(argvals_y)[:,:,0]
    evalfull = bbasis_fullx(argvals_full)[:,:,0]


    final_x = x @ evalx
    final_y = y @ evaly
    final_full = full @ evalfull


    fdx = FDataGrid(final_x, argvals_x)
    fdy = FDataGrid(final_y, argvals_y)
    fdfull = FDataGrid(final_full, argvals_full)

    return{'fdx': fdx.to_basis(bbasis_x),
           'fdy': fdy.to_basis(bbasis_y),
           'fdfull': fdfull.to_basis(bbasis_fullx),
           'groupd': ((labels - 1).to_numpy()).T
    }

import matplotlib.pyplot as plt
import numpy as np

def plotFlourFD(fd):
    """
    Description
    -----------
    This function plots the functional data fd. Replica of R code.

    Parameters
    ----------
    fd : dict
        A dictionary containing:
        - 'fdfull': Functional data object for the full basis.
        - 'groupd': Array indicating the group membership (Sunday or Tuesday).
        - Other keys as necessary for your application.
    """
    # Define the unique clusters and their colors
    clusters = np.unique(fd['groupd'])
    colors = plt.cm.viridis(np.linspace(0, 1, len(clusters)))

    # Plot all curves with colors based on their labels
    fd['fdfull'].plot(group=fd['groupd'])

    # Add axis labels and a legend
    plt.xlabel('Domain')
    plt.ylabel('Curve')
    plt.title('All Curves by Cluster')
    plt.show()

    # Plot the curves belonging to each cluster separately
    for cluster, color in zip(clusters, colors):
        indices = np.where(fd['groupd'] == cluster)[0]
        fd['fdfull'][indices].plot(color=color, label=cluster)
        
        plt.xlabel('Domain')
        plt.ylabel('Curve')
        plt.title(f'Curves for Cluster: {cluster + 1}')
        plt.show()

if __name__ == "__main__":
    fd = fitFlourFD()
    models = ["AKJBKQKDK","AKJBQKDK", "AKBKQKDK", "ABKQKDK", "AKBQKDK", "ABQKDK"]
    modelsy = ["EII", "VII", "EEI", "VEI", "EVI", "VVI", "EEE", "VEE", "EVE","EEV", "VVE", "VEV","EVV","VVV"]
    modelsys = ["EII", "VII", "EEI", "VEI"]
    labels = fd['groupd']
    res = tfun.tfunHDDC(fd['fdx'], fd['fdy'], K=3, model=models, modely=modelsys, threshold=0.1, init="kmeans", nb_rep=1)
    print(met.confusion_matrix(res.cl, labels))  


