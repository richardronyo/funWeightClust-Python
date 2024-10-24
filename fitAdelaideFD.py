import numpy as np
import pandas as pd
from process_data import create_functional_data, plot_fd

def fitAdelaide():
    """
    This function creates functional data from the Adelaide dataset
    """

    x = pd.read_csv("data/Adelaide/de_x.csv").to_numpy()
    y = pd.read_csv("data/Adelaide/de_y.csv").to_numpy()

    fdx = create_functional_data(x, "BSPLINE", y.shape[1])
    fdy = create_functional_data(y, "BSPLINE", y.shape[1])


    labels = np.repeat(["Sunday", "Tuesday"], [x.shape[0], x.shape[0]])
    clm = np.repeat([0, 1], [int(x.shape[0]/2), int(x.shape[0]/2)])

    return {
        "fdx": fdx,
        "fdy": fdy,
        "labels": labels,
        "groupd": clm
    }


