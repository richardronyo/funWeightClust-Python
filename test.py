# test_mixture.py
import skfda
import numpy as np
import modelSimulation as sim
from sklearn import metrics as met
from skewfunHDDC import _T_funhddt_m_step1, _T_mypcat_fd1_Uni
from scipy import linalg as scil



if __name__ == "__main__":
    data = sim.genModelFD(ncurves=1000, nsplines=35, alpha=[0.9, 0.9, 0.9], eta=[10, 5, 15])

    fdobj = data['data'][0]  # Assuming data is a list with fdobj at index 0
    fdobjy = data['data'][1]  # Assuming fdobjy is at index 1
    bigDATA = np.random.rand(35, 1000)  # Example bigDATA, adjust dimensions as needed
    Wlist = {'W_m': np.eye(35)}  # Example Wlist, adjust as needed
    K = 3
    t = np.random.rand(1000, K)  # Example t, adjust dimensions as needed
    model = 'AKJBKQKDK'  # Example model, adjust as needed
    modely = 'VII'  # Example modely, adjust as needed
    threshold = 0.5  # Example threshold, adjust as needed
    method = 'bic'  # Example method, adjust as needed
    noise_ctrl = False  # Example noise_ctrl, adjust as needed
    com_dim = 5  # Example com_dim, adjust as needed
    d_max = 10  # Example d_max, adjust as needed
    d_set = np.array([1, 2, 3])  # Example d_set, adjust as needed

    corX = t
    eigenvalues, cov, bj = _T_mypcat_fd1_Uni(fdobj.coefficients, Wlist['W_m'], np.atleast_2d(t[:,0]), np.atleast_2d(corX[:,0]))
    print("eigenvalues: ", eigenvalues)
    print("Covariance Matrix: ", cov)
    print("bj: ", bj)

    result = _T_funhddt_m_step1(fdobj, bigDATA, fdobjy, Wlist, K, t, model, modely, threshold, method, noise_ctrl, com_dim, d_max, d_set)
    print(result)