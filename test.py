# test_mixture.py
import skfda
import numpy as np
import modelSimulation as sim
import skewfunHDDC as tfun
from sklearn import metrics as met
from skewfunHDDC import _T_funhddt_m_step1, _T_funhddt_e_step1, _T_hdc_getComplexityt, _T_hdclassift_bic, _T_funhddc_main1
from scipy import linalg as scil


from py_mixture import C_mstep

if __name__ == "__main__":
    datax = sim.genModelFD(ncurves=9, nsplines=7, alpha=[0.9, 0.9, 0.9], eta=[10, 5, 15])
    datay = sim.genModelFD(ncurves=9, nsplines=8, alpha=[0.9, 0.9, 0.9], eta=[10, 5, 15])
    fdobj = datax['data']  # Assuming data is a list with fdobj at index 0
    fdobjy = datay['data']  # Assuming fdobjy is at index 1
    N = 9
    p = 7
    q = 8
    K = 4

    """
                  [,1]      [,2]       [,3]      [,4]
        [1,] 0.1386920 0.4736886 0.09821989 0.2893995
        [2,] 0.2634635 0.2358751 0.28971799 0.2109434
        [3,] 0.1864560 0.3335034 0.26260827 0.2174324
        [4,] 0.4009334 0.1485603 0.07463583 0.3758704
        [5,] 0.2652814 0.1247790 0.37873804 0.2312015
        [6,] 0.2634635 0.2358751 0.28971799 0.2109434
        [7,] 0.1864560 0.3335034 0.26260827 0.2174324
        [8,] 0.4009334 0.1485603 0.07463583 0.3758704
        [9,] 0.2652814 0.1247790 0.37873804 0.2312015
    """

    t = np.array([[0.1386920, 0.4736886, 0.09821989, 0.2893995], 
                  [0.2634635, 0.2358751, 0.28971799, 0.2109434],
                  [0.1864560, 0.3335034, 0.26260827, 0.2174324], 
                  [0.4009334, 0.1485603, 0.07463583, 0.3758704],
                  [0.2652814, 0.1247790, 0.37873804, 0.2312015],
                  [0.2634635, 0.2358751, 0.28971799, 0.2109434],
                  [0.1864560, 0.3335034, 0.26260827, 0.2174324], 
                  [0.4009334, 0.1485603, 0.07463583, 0.3758704],
                  [0.2652814, 0.1247790, 0.37873804, 0.2312015]]) 
    t = t / np.sum(t, axis=1, keepdims=True)

    #bigDATA = np.random.rand(p+1, N)  # Example bigDATA, adjust dimensions as needed
    #W should be symmetric
    W = np.array([[3.57142857, 2.187510807, 0.461282867,  0.02976911, 0.000000000, 0.000000000, 0.00000000],
                    [2.18751081, 5.535671059, 3.906319524,  0.86307362, 0.007442277, 0.000000000, 0.00000000],
                    [0.46128287, 3.906319524, 8.169562887,  5.61510914, 0.590268892, 0.007442277, 0.00000000],
                    [0.02976911, 0.863073625, 5.615109144, 11.98410777, 5.615109144, 0.863073625, 0.02976911],
                    [0.00000000, 0.007442277, 0.590268892,  5.61510914, 8.169562887, 3.906319524, 0.46128287],
                    [0.00000000, 0.000000000, 0.007442277,  0.86307362, 3.906319524, 5.535671059, 2.18751081],
                    [0.00000000, 0.000000000, 0.000000000,  0.02976911, 0.461282867, 2.187510807, 3.57142857]])

    W_m = scil.cholesky(W)
    dety = scil.det(W)
    Wlist = {'W': W, 'W_m': W_m, 'dety':dety}

    fdobj_coefficients = np.array([
        [15.179215, 12.2853302, 0.1706342, 5.9525660, -1.85376983, -2.489193, -3.6606809, 1.115633, -3.2780230],
        [19.522077, -1.8987559, -4.9134485, -5.6769222, 2.11583787, 3.516996, 19.8439407, 23.548688, 23.9715003],
        [49.024911, 51.5248197, 53.0146325, 78.6522744, 80.83581270, 80.996987, -1.5083613, -5.477779, -0.2653915],
        [104.111753, 100.8021604, 98.3981979, 2.6491477, 0.06517189, 1.888955, 77.7580021, 80.152517, 73.4006103],
        [-0.117254, -0.9624216, 1.8260055, 39.0505829, 40.62584847, 43.127121, 1.6229681, 7.063851, -3.3246895],
        [-1.673156, -1.6419503, 2.5976759, -0.1201364, 1.55263213, 4.474678, 0.8027561, -7.335495, 5.4182071],
        [-4.036739, -0.7964045, -3.5410879, -3.4811767, 1.63984854, -3.420437, 102.7369060, 97.748196, 91.5799685]
    ])
    
    fdobjy_coefficients = np.array([
        [11.4001637, 4.5154316, 1.7397484, 1.5434480, -4.1241413, -4.3976390, 2.2354556, 4.1885579, 7.349027],
        [-14.5813800, 18.2146662, -14.9583650, 0.1588885, -0.9466329, 1.7702740, -6.5615520, 1.1021552, 6.162376],
        [46.7555476, 49.9110952, 49.1468820, 78.1050064, 78.7403193, 79.0529752, 22.6000279, 17.0953475, 22.080442],
        [98.7477271, 97.8715514, 97.0447163, -1.0608164, 0.6595792, -1.4585182, 3.4110204, -0.3643439, -1.067301],
        [0.9801432, 0.5597144, -0.7128167, 40.6680433, 45.7456131, 38.9485551, 81.4397818, 81.6511184, 82.548493],
        [0.5700323, -2.0627910, -0.7397165, 1.5665028, 4.2125860, -0.1636002, 1.9342918, 1.0229902, 2.776774],
        [-0.4195140, 0.1075383, 0.2273986, -1.1553105, 7.1781329, 2.0139017, 0.0654272, -2.2920262, -3.845197],
        [2.0089707, -2.8761052, 0.8156604, 0.9993770, 0.7680416, -3.5606855, 96.0266593, 96.0125727, 101.047078]
    ])
    
    fdobj.coefficients = fdobj_coefficients.T
    fdobjy.coefficients = fdobjy_coefficients.T

    DATA = fdobj_coefficients
    intermediate_bigDATA = W@(DATA)

    ones_row = np.ones((1, N))
    bigDATA = np.vstack((intermediate_bigDATA, ones_row))
    

    model = 'ABKQKDK'  # Example model, adjust as needed
    modely = 'EII'  # Example modely, adjust as needed
    threshold = 0.5  # Example threshold, adjust as needed
    method = 'cattell'  # Example method, adjust as needed
    noise_ctrl = False  # Example noise_ctrl, adjust as needed
    com_dim = 4 # Example com_dim, adjust as needed
    d_max = 100  # Example d_max, adjust as needed
    d_set = np.array([2, 2, 2, 2])  # Example d_set, adjust as needed

    corX = t
    print("-----------------------------------------------------------------------------------------------------")
    par = _T_funhddt_m_step1(fdobj, bigDATA, fdobjy, Wlist, N, p, q, K, t, model, modely, threshold, method, noise_ctrl, d_set, com_dim, d_max)
    print("model: ", par['model'])
    print("modely: ", par['modely'])
    print("ev:\t", par['ev'].shape, "\n", par['ev'])
    print("a:\n", par['a'])
    print("b:\n", par['b'])
    print("d:\n", par['d'])
    print("mu:\n", par['mu'])
    print("prop:\n", par['prop'])
    print("gami:\t", par['gam'].shape, "\n", par['gam'])
    print("covy\t", par['covy'].shape, "\n:", par['covy'])
    print("icovyi:\t", par['icovy'].shape, "\n", par['icovy'])
    print("logi:\n", par['logi'])
    print("-----------------------------------------------------------------------------------------------------")
    e = _T_funhddt_e_step1(fdobj, bigDATA, fdobjy, Wlist, N, p, q, par)
    print("t:\t", e['t'].shape, "\n", e['t'])
    print("L:\n", e['L'])
    print("mah_pen:\n", e['mah_pen'])
    print("K_pen:\n", e['K_pen'])
    print("mah_pen1:\n", e['mah_pen1'])
    print("-----------------------------------------------------------------------------------------------------")
    complexity = _T_hdc_getComplexityt(par, p, q)
    print(complexity)
    print("-----------------------------------------------------------------------------------------------------")
    likely = []
    likely.append(e['L'])
    par['loglik'] = likely[-1]
    par['posterior'] = e['t']
    
    bic = _T_hdclassift_bic(par, p, q)
    print(bic)
    print("-----------------------------------------------------------------------------------------------------")
    dfstart = 50
    dfupdate = 'approx'
    dfconstr = 'no'
    itermax=200
    eps = 1.e-6
    init = 'random'
    init_vector = None
    mini_nb = [5,10]
    min_individuals = 4
    kmeans_control = {'n_init':1, 'max_iter':10, 'algorithm':'lloyd'}
    known = None
    ans = _T_funhddc_main1(fdobj, fdobjy, Wlist, K, dfstart, dfupdate, dfconstr, model, modely, itermax, threshold, method, eps, init, init_vector,mini_nb, min_individuals, noise_ctrl, com_dim,kmeans_control, d_max, d_set, known)
    print(ans)
    print("-----------------------------------------------------------------------------------------------------")
    res = tfun.tfunHDDC(fdobj, fdobjy, model='all', modely='all', K=1)
