import numpy as np
import pandas as pd
from skfda.representation.basis import BSplineBasis, FDataBasis
from skfda.preprocessing.smoothing import BasisSmoother
from skfda.representation.grid import FDataGrid
from matplotlib import pyplot as plt
import skewfunHDDC as tfun


def fitAdelaideFD():
    demand = pd.read_csv("de_combined.csv")
    ans = demand.values[:, :48]
    ans = ans.astype(float)

    cls = demand.values[:, 48]
    bbasis_x = BSplineBasis([1, 12], n_basis=6)
    bbasis_y = BSplineBasis([13, 24], n_basis=6)
    bbasis_fullx = BSplineBasis([1, 24], n_basis=6)

    argvals_x = np.linspace(1, 12, 24)
    argvals_y = np.linspace(13, 24, 24)
    argvals_fullx = np.concatenate((argvals_x, argvals_y))

    smoother_x = BasisSmoother(bbasis_x)
    smoother_y = BasisSmoother(bbasis_y)
    smoother_fullx = BasisSmoother(bbasis_fullx)

    fd_demand_x = smoother_x.fit_transform(FDataGrid(ans[:, :24], argvals_x))
    fd_demand_y = smoother_y.fit_transform(FDataGrid(ans[:, 24:48], argvals_y))
    fd_demand_fullx = smoother_fullx.fit_transform(FDataGrid(ans, argvals_fullx))

    return {
        'fdx': fd_demand_x.to_basis(bbasis_x),
        'fdy': fd_demand_y.to_basis(bbasis_y),
        'fdfull': fd_demand_fullx.to_basis(bbasis_fullx),
        'groupd': cls
    }

def plotAdelaideFD(fd):
    cls = fd['groupd']  # Replace with your actual group labels

    fd['fdfull'].plot(group=cls)
    plt.show()

if __name__ == "__main__":
    ans = fitAdelaideFD()
    plotAdelaideFD(ans)

    tfun.tfunHDDC(ans['fdx'], ans['fdy'], K=2)