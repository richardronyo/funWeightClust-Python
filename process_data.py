import numpy as np
import pandas as pd

from skfda.representation.basis import BSplineBasis, FourierBasis
from skfda.representation.grid import FDataGrid

import funweightclust as fwc
from sklearn import metrics as met


pd.set_option('future.no_silent_downcasting', True)

def create_functional_data(values, basis_type = "FOURIER", n_basis=50):
    """
    This function creates functional data from a 2D NumPy array. It will use a Fourier Basis with 50 basis functions by default, and return a FDataGrid object that can be used in the FunWeightClust Model.
    """
    domain = values.shape[1]

    x_grid_points = np.arange(1, domain + 1)

    if basis_type != "FOURIER":
        fd = FDataGrid(data_matrix=values, grid_points=x_grid_points).to_basis(FourierBasis(domain_range=(1, domain + 1), n_basis=n_basis))
    else:
        fd = FDataGrid(data_matrix=values, grid_points=x_grid_points).to_basis(BSplineBasis(domain_range=(1, domain + 1), n_basis=n_basis))

    return fd

def constant_functional_data(values, domain = 200):
    """
    This function generates functional data from a 1D NumPy array. It does so by repeating the values in the column to create a matrix, and then using this matrix to create a FDataGrid object that can be used in the FunWeightClust model.

    This is useful if trying to use the FunWeightClust Model when you want the model to predict a constant value.
    """

    values = values.reshape(-1, 1)
    values = np.tile(values, (1, domain))

    x_grid_points = np.arange(1, domain + 1)

    fd = FDataGrid(data_matrix=values, grid_points=x_grid_points).to_basis(FourierBasis(domain_range=(1, domain + 1), n_basis=50))

    return fd
    

if __name__ == "__main__":
    raw_cingulum = pd.read_csv("data/ADNI/ADNI_Cingulum_ADCN.csv")
    cingulum_voxelwise_data_column_names = [column for column in raw_cingulum.columns if "Var" in column]
    cingulum_voxelwise_data = raw_cingulum[cingulum_voxelwise_data_column_names]

    
    
    cingulum_labels = raw_cingulum["Research.Group"].replace({'AD':0, 'CN':1}).astype(int).to_numpy()

    cingulum_fd = create_functional_data(cingulum_voxelwise_data)
    cingulum_fd.plot()

    raw_cingulum_y_data = raw_cingulum["MMSE.Total.Score"].to_numpy()
    cingulum_fdy = constant_functional_data(raw_cingulum_y_data)

    res = fwc.funweightclust(cingulum_fd, cingulum_fdy, K=2, model="all", modely="all", init="kmeans", nb_rep=1, threshold=0.001)
    print("ARI Score:\t", met.adjusted_rand_score(res.cl, cingulum_labels))
    print("Confusion Matrix:\n", met.confusion_matrix(res.cl, cingulum_labels))
