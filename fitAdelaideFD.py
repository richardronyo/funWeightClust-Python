import numpy as np
import pandas as pd
from skfda.representation.basis import BSplineBasis, FDataBasis
from skfda.preprocessing.smoothing import BasisSmoother
from skfda.representation.grid import FDataGrid
from matplotlib import pyplot as plt
import skewfunHDDC as tfun
from sklearn import metrics as met



def fitAdelaideFD():
    demand = pd.read_csv("de_combined.csv")
    full = demand.values[:, :48]
    full = full.astype(float)
    x = full[:508, :]
    y = full[508:, :]

    ncurves = full.shape[0]
    nsplines = full.shape[1]

    cls = demand.values[:, 48]

    bbasis_x = BSplineBasis(n_basis=nsplines, domain_range=(0, 24))
    bbasis_y = BSplineBasis(n_basis=nsplines, domain_range=(0, 24))
    bbasis_fullx = BSplineBasis(n_basis=nsplines, domain_range=(0, 24))

    argvals = np.linspace(0, 24, 48)


    evalx = bbasis_x.evaluate(argvals)[:,:,0]
    evaly = bbasis_y.evaluate(argvals)[:,:,0]
    evalfull = bbasis_fullx.evaluate(argvals)[:,:,0]

    final_x = x @ evalx
    final_y = y @ evaly
    final_full = full @ evalfull


    fdx = FDataGrid(final_x, argvals)
    fdy = FDataGrid(final_y, argvals)
    fdfull = FDataGrid(final_full, argvals)

    return {
        'fdx': fdx.to_basis(bbasis_x),
        'fdy': fdy.to_basis(bbasis_y),
        'fdfull': fdfull.to_basis(bbasis_fullx),
        'groupd': cls,
    }

def plotAdelaideFD(fd):
    cls = fd['groupd']  # Replace with your actual group labels

    fd['fdfull'].plot(group=cls)
    plt.show()

if __name__ == "__main__":
    ans = fitAdelaideFD()
    labels = ans['groupd']

    plotAdelaideFD(ans)

    models = ["AKJBKQKDK","AKJBQKDK", "AKBKQKDK", "ABKQKDK", "AKBQKDK", "ABQKDK"]
    modelsys = ["EII", "VII", "EEI", "VEI"]

    res = tfun.tfunHDDC(ans['fdx'], ans['fdy'], K=2, model=models, modely=modelsys, init="kmeans", nb_rep=1, threshold=0.1)