# test_mixture.py
import skfda
import numpy as np
import modelSimulation as sim
from sklearn import metrics as met
from skewfunHDDC import _T_funhddt_m_step1, _T_funhddt_e_step1
from scipy import linalg as scil


"""
Coefficient Matrix Size: (1 x Number of Splines) -> (NCurves x Number of Splines/NCurves)
    - N = Number of Curves
    - p = Number of Splines / NCurves
"""


if __name__ == "__main__":
    datax = sim.genModelFD(ncurves=5, nsplines=35, alpha=[0.9, 0.9, 0.9], eta=[10, 5, 15])
    datay = sim.genModelFD(ncurves=5, nsplines=35, alpha=[0.9, 0.9, 0.9], eta=[10, 5, 15])
    fdobj = datax['data'][0]  # Assuming data is a list with fdobj at index 0
    fdobjy = datay['data'][1]  # Assuming fdobjy is at index 1
    N = 5
    p = 7
    q = p
    fdobj_coefficients = np.reshape(fdobj.coefficients, (N, p))
    fdobjy_coefficients = np.reshape(fdobjy.coefficients, (N, p))
    bigDATA = np.random.rand(p+1, N)  # Example bigDATA, adjust dimensions as needed
    #W should be symmetric
    W= np.eye(7)
    W_m = scil.cholesky(W)
    dety = scil.det(W)
    Wlist = {'W': W, 'W_m': W_m, 'dety':dety}

    K = 4
    t = np.random.rand(N, K)  # Example t, adjust dimensions as needed
    model = 'AKJBKQKDK'  # Example model, adjust as needed
    modely = 'VII'  # Example modely, adjust as needed
    threshold = 0.5  # Example threshold, adjust as needed
    method = 'bic'  # Example method, adjust as needed
    noise_ctrl = False  # Example noise_ctrl, adjust as needed
    com_dim = 5  # Example com_dim, adjust as needed
    d_max = 10  # Example d_max, adjust as needed
    d_set = np.array([1, 2, 3])  # Example d_set, adjust as needed

    corX = t
    ti = np.atleast_2d(t[:, 0])
    
    coefmean = np.sum(np.transpose(ti) @ (np.ones((1, fdobj_coefficients.shape[1]))) * fdobj_coefficients, axis=0) / np.sum(ti)
    print("-----------------------------------------------------------------------------------------------------")
    ans = _T_funhddt_m_step1(fdobj, bigDATA, fdobjy, Wlist, N, p, q, 4, t, model, modely, threshold, method, noise_ctrl, com_dim, d_max, d_set)
    print(list(ans.keys()))
    print("-----------------------------------------------------------------------------------------------------")
    e_step = _T_funhddt_e_step1(fdobj, bigDATA, fdobjy, Wlist, N, p, q, ans, clas=0, known=None, kno=None)
    print(e_step)