#Required Libraries------------------------------------------------------------#
import skfda
from sklearn import cluster as clust
from sklearn.model_selection import ParameterGrid
import numpy as np
import warnings
import pandas as pd
import multiprocessing as multi
import time
from scipy import linalg as scil
from scipy.special import digamma
from scipy.special import loggamma
from scipy.special import binom
from scipy.optimize import brentq
from shutil import get_terminal_size
import numba as nb
from numba import complex128
import math
#------------------------------------------------------------------------------#

#GLOBALS
LIST_TYPES = (list, np.ndarray)
UNKNOWNS = (np.NaN, np.inf, -np.inf, None)
INT_TYPES = (int, np.integer)
FLOAT_TYPES = (float, np.floating)
NUMERIC_TYPES = (INT_TYPES, FLOAT_TYPES)

def check_symmetric(a, tol=1e-8):
    '''
    Check if matrix is symmetric to a given tolerance.

    a -- the matrix to be checked
    tol -- the tolerance being checked
    '''
    return np.all(np.abs(a-a.T) < tol)


class _Table:
    """
    Store data with row names and column names
    
    Attributes
    ----------
    data : numpy array
    rownames : numpy array
    colnames : numpy array
    """
    def __init__(self, data=None, rownames=None, colnames=None):
        self.data = data
        self.rownames = rownames
        self.colnames = colnames


    def switchRow(self, ind1, ind2):
        if ind1 < ind2:
            self.data[[ind1, ind2]] = self.data[[ind2, ind1]]
            self.rownames[[ind1, ind2]] = self.rownames[[ind1, ind2]]
        else:
            self.data[[ind2, ind1]] = self.data[[ind1, ind2]]
            self.rownames[[ind2, ind1]] = self.rownames[[ind1, ind2]]

    def switchCol(self, ind1, ind2):
        if ind1 < ind2:
            self.data.T[[ind1, ind2]] = self.data.T[[ind2, ind1]]
            self.colnames[[ind1, ind2]] = self.colnames[[ind2, ind1]]
        else:
            self.data.T[[ind2, ind1]] = self.data.T[[ind1, ind2]]
            self.colnames[[ind2, ind1]] = self.colnames[[ind1, ind2]]

    def __str__(self):
        string = f'\t{self.colnames}\n'
        for i in range(len(self.data)):
            try:
                string += f'{self.rownames[i]}\t{self.data[i]}\n'
            except:
                string += f'\t{self.data[i]}\n'

        return string

    def __len__(self):
        return len(self.data)
    
class FunWeightClust:
    '''
    Description
    -----------
    Object containing the parameters and result of clustering functional data
    via FunWeightClust. Can predict clusters on functional data on the same size basis
    as the one originally clustered.

    Attributes
    ----------
    Wlist : `dict`
        list containing the following: the inner product matrix of the 
        functional data's bases with themselves, the cholesky decomposition of 
        the previous matrix, and the determinant of the first matrix

    model : `str` 
        model chosen for best clustering of original data
    K : `int` 
        number of clusters chosen for best clustering of original data
    d : `np.ndarray` of `ints`
        number of intrinsic dimensions found for best clustering of original
        data
    a : `np.ndarray` of `floats` 
        a matrix for best clustering of original data
    b : `np.ndarray` of `floats` 
        b matrix for best clustering of original data
    mu : `np.ndarray` of `floats`
        :`mu_k` is a vector of means found in the m-step
    prop : `np.ndarray` of `floats`
        vector of proportions used for the mixture distributions
    nux : `np.ndarray` of `floats`
        vector of degrees of freedom
    ev : `np.ndarray` of `floats`
        matrix containing eigenvectors
    Q : `np.ndarray` of `floats`
        ---NOT USED---
    Q1 : `np.ndarray` of `floats`
        matrix of coefficients of eigenfunctions
    fpca : `dict`
        ---NOT USED---
    loglik : `float`
        log likelihood calculated for best clustering of original data
    loglik_all : `np.ndarray` of `floats`
        log likelihood calculated for all parameter combinations and
        repetitions during clustering of original data
    posterior : `np.ndarray` of `floats`
        posterior probability distribution calculated
    cl : `np.ndarray` of `ints` 
        chosen clustering of data
    com_ev : `int`
        ---NOT USED---
    N : `int`
        number of rows in coefficients of basis functions
    complexity : `int`
        number of free parameters to be estimated
    threshold : `float`
        threshold used for best clustering of original data
    d_select : `str`
        method used to decided intrinsic dimensions
    converged : `bool`
        a boolean indicating whether the FunWeightClust algorithm converged
        to a clustering or not
    index -- 
    bic : `float`
        Bayseian information criterion
    icl : `float`
        bic added with the addition of the estimated mean entropy
    basis : `Basis` 
        set of basis functions that the data was fit on
    '''

    def __init__(self, Wlist, model, modely, K, d, a, b, mu, prop, nux, ev, Q, Q1,
                 fpca, loglik, loglik_all, posterior, cl, com_ev, N,
                 complexity, threshold, d_select, converged, index, bic, icl, basis, gam, covy, icovy, logi, first_t):
        self.Wlist = Wlist
        self.model = model
        self.modely = modely
        self.K = K
        self.d = d
        self.a = a
        self.b = b
        self.mu = mu
        self.prop = prop
        self.nux = nux
        self.ev = ev
        self.Q=Q
        self.Q1 = Q1
        self.fpca = fpca
        self.loglik = loglik
        self.loglik_all = loglik_all
        self.posterior = posterior
        self.cl = cl
        self.com_ev = com_ev
        self.N = N
        self.complexity = complexity
        self.threshold=threshold
        self.d_select=d_select
        self.converged=converged
        self.index = index
        self.bic = bic
        self.icl = icl
        self.criterion = None
        self.complexity_all = None
        self.allCriteria = None
        self.allRes = None
        self.basis = basis
        self.gam = gam
        self.covy = covy
        self.icovy = icovy
        self.logi = logi
        self.first_t = first_t

    
    from imahalanobis import imahalanobis
    from py_mixture import C_rmahalanobis
    def predict(self, data, datay):
        '''
        predict takes a FDataBasis object on the same number of basis functions as
        the original clustered data and uses the result of the original
        clustering to predict a clustering for the new set of data.

        data -- predictor functional data on the same number of basis functions as
        the original clustered data.

        datay - response functional data on the same number of basis functions as the original clustered data.
        '''

        if data is None:
            raise ValueError("Cannot predict without supplying data")

        #univariate
        x = data.coefficients.copy()
        y = datay.coefficients.copy()

        Np = data.basis
        p = x.shape[1]
        N = x.shape[0]
        q = y.shape[1]
        if Np != self.basis:
            raise ValueError("New observations should be represented using the same base as for the training set")
        a = self.a
        b = self.b
        d = self.d
        mu = self.mu
        prop = self.prop
        b[b<1.e-6] = 1.e-6
        W = self.Wlist['W']
        wki = self.Wlist['W_m']
        dety = self.Wlist["dety"]
        ldetcov = self.logi
        pqp = p+1
        t = np.zeros((N, self.K))
        mah_pen = np.zeros((self.K, N))
        mah_pen1 = np.zeros((self.K, N))
        K_pen = np.zeros((N, self.K))
        num = np.zeros((N, self.K))
        s = np.repeat(0., self.K)
        pi = math.pi
        gam = self.gam
        icovy = self.covy
        ones_row = np.ones((1, N))
        DATA = x.T
        intermediate_bigDATA = W@(DATA)
        bigDATA = np.vstack((intermediate_bigDATA, ones_row))
        bigx = bigDATA.T

        match self.model:
            case 'AKJBKQKDK':
                for i in range(self.K):
                    s[i] = np.sum(np.log(a[i, 0:d[i]]))
                    Qk = self.Q1[f'{i}'].copy()
                    diag2 = np.repeat(1/b[i], p-d[i])
                    diag1 = 1/a[i, 0:d[i]]
                    aki = np.sqrt(np.diag(np.concatenate((diag1, diag2))))
                    muki = mu[i]
                    pp = x.shape[1]
                    pN = x.shape[0]
                    pdi = aki.shape[1]

                    mah_pen[i, :] = imahalanobis(x, muki, wki, Qk, aki, pp, pN, pdi, np.zeros(N))
                    delta = np.zeros(N)
                    #scipy logamma vs math lgamma?
                    mah_pen1[i, :] = C_rmahalanobis(N, pqp, q, self.K, i, bigx, y, gam[i, :, :], icovy[i, :, :], delta)
                    pi = math.pi
                    K_pen[:, i] = (-2 * np.log(prop[i])) + (p + q) * np.log(2 * pi) + s[i] - np.log(dety) + (p - d[i]) * np.log(b[i]) + mah_pen[i, :] + mah_pen1[i, :] + ldetcov[i]
            case 'AKJBQKDK':
                for i in range(self.K):
                    s[i] = np.sum(np.log(a[i, 0:d[i]]))
                    Qk =self.Q1[f'{i}']
                    diag2 = np.repeat(1/b[0], p-d[i])
                    diag1 = 1/a[i, 0:d[i]]
                    aki = np.sqrt(np.diag(np.concatenate((diag1, diag2))))
                    muki = mu[i]
                    
                    pp = x.shape[1]
                    pN = x.shape[0]
                    pdi = aki.shape[1]
                    delta = np.zeros(N)

                    mah_pen[i, :] = imahalanobis(x, muki, wki, Qk, aki, pp, pN, pdi, np.zeros(N))
                    delta = np.zeros(N)
                    #scipy logamma vs math lgamma?
                    mah_pen1[i, :] = C_rmahalanobis(N, pqp, q, self.K, i, bigx, y, gam[i, :, :], icovy[i, :, :], delta)
                    pi = math.pi
                    K_pen[:, i] = (-2 * np.log(prop[i])) + (p + q) * np.log(2 * pi) + s[i] - np.log(dety) + (p - d[i]) * np.log(b[0]) + mah_pen[i, :] + mah_pen1[i, :] + ldetcov[i]
            case 'AKBKQKDK':
                for i in range(self.K):
                    s[i] = d[i]*np.log(a[i])
                    Qk = self.Q1[f'{i}']
                    diag2 = np.repeat(1/b[i], p-d[i])
                    diag1 = np.repeat(1/a[i], d[i])
                    aki = np.sqrt(np.diag(np.concatenate((diag1, diag2))))
                    muki = mu[i]
                    pp = x.shape[1]
                    pN = x.shape[0]
                    pdi = aki.shape[1]
                    delta = np.zeros(N)

                    mah_pen[i, :] = imahalanobis(x, muki, wki, Qk, aki, pp, pN, pdi, np.zeros(N))
                    delta = np.zeros(N)
                    #scipy logamma vs math lgamma?
                    mah_pen1[i, :] = C_rmahalanobis(N, pqp, q, self.K, i, bigx, y, gam[i, :, :], icovy[i, :, :], delta)
                    pi = math.pi
                    K_pen[:, i] = (-2 * np.log(prop[i])) + (p + q) * np.log(2 * pi) + s[i] - np.log(dety) + (p - d[i]) * np.log(b[i]) + mah_pen[i, :] + mah_pen1[i, :] + ldetcov[i]

            case 'ABKQKDK':
                for i in range(self.K):
                    s[i] = d[i]*np.log(a[0])
                    Qk = self.Q1[f'{i}']
                    diag2 = np.repeat(1/b[i], p-d[i])
                    diag1 = np.repeat(1/a[0], d[i])
                    aki = np.sqrt(np.diag(np.concatenate((diag1, diag2))))
                    muki = mu[i]
                    delta = np.zeros(N)

                    pp = x.shape[1]
                    pN = x.shape[0]
                    pdi = aki.shape[1]
                    mah_pen[i, :] = imahalanobis(x, muki, wki, Qk, aki, pp, pN, pdi, np.zeros(N))
                    delta = np.zeros(N)
                    #scipy logamma vs math lgamma?
                    mah_pen1[i, :] = C_rmahalanobis(N, pqp, q, self.K, i, bigx, y, gam[i, :, :], icovy[i, :, :], delta)
                    pi = math.pi
                    K_pen[:, i] = (-2 * np.log(prop[i])) + (p + q) * np.log(2 * pi) + s[i] - np.log(dety) + (p - d[i]) * np.log(b[i]) + mah_pen[i, :] + mah_pen1[i, :] + ldetcov[i]

            case 'AKBQKDK':
                for i in range(self.K):
                    s[i] = d[i]*np.log(a[i])
                    Qk = self.Q1[f'{i}']
                    diag2 = np.repeat(1/b[0], p-d[i])
                    diag1 = np.repeat(1/a[i], d[i])
                    aki = np.sqrt(np.diag(np.concatenate((diag1, diag2))))
                    muki = mu[i]
                    pp = x.shape[1]
                    pN = x.shape[0]
                    pdi = aki.shape[1]
                    mah_pen[i, :] = imahalanobis(x, muki, wki, Qk, aki, pp, pN, pdi, np.zeros(N))
                    delta = np.zeros(N)
                    mah_pen1[i, :] = C_rmahalanobis(N, pqp, q, self.K, i, bigx, y, gam[i, :, :], icovy[i, :, :], delta)
                    pi = math.pi
                    K_pen[:, i] = (-2 * np.log(prop[i])) + (p + q) * np.log(2 * pi) + s[i] - np.log(dety) + (p - d[i]) * np.log(b[0]) + mah_pen[i, :] + mah_pen1[i, :] + ldetcov[i]

            case 'ABQKDK':
                for i in range(self.K):
                    s[i] = d[i]*np.log(a[0])
                    Qk = self.Q1[f'{i}']
                    diag2 = np.repeat(1/b[0], p-d[i])
                    diag1 = np.repeat(1/a[0], d[i])
                    aki = np.sqrt(np.diag(np.concatenate((diag1, diag2))))
                    muki = mu[i]

                    pp = x.shape[1]
                    pN = x.shape[0]
                    pdi = aki.shape[1]
                    mah_pen[i, :] = imahalanobis(x, muki, wki, Qk, aki, pp, pN, pdi, np.zeros(N))
                    delta = np.zeros(N)
                    mah_pen1[i, :] = C_rmahalanobis(N, pqp, q, self.K, i, bigx, y, gam[i, :, :], icovy[i, :, :], delta)
                    pi = math.pi

                    K_pen[:, i] = (-2 * np.log(prop[i])) + (p + q) * np.log(2 * pi) + s[i] - np.log(dety) + (p - d[i]) * np.log(b[0]) + mah_pen[i, :] + mah_pen1[i, :] + ldetcov[i]

        kcon = -np.apply_along_axis(np.max, 1, K_pen)
        K_pen += np.atleast_2d(kcon).T
        num = np.exp(K_pen)
        
        t = num / np.atleast_2d(np.sum(num, axis=1)).T
        cl = np.argmax(t, axis=1)

        return {'t': t, 'class': cl}


def funweightclust(datax, datay, K=np.arange(1,11), model='AKJBKQKDK', modely = "VVV", known=None, threshold=0.1, itermax=200, dfstart=50., eps=1.e-6,init='random',
            criterion='bic', d_select='cattell', init_vector=None, 
            show=True, mini_nb=[5,10], min_individuals=4, mc_cores=1, nb_rep=2,
            keepAllRes=False,kmeans_control={'n_init':1, 'max_iter':10, 'algorithm':'lloyd'},  d_max=100, d_range=2, cmtol=1.e-10, cmmax=10, verbose=True,
            dfupdate='approx', dfconstr='no',   
            Numba=True, testing=False, t = None):
    '''
    Description
    -----------
    FunWeightClust is a clustering algorithm for functional data. It is based on the work developed by Dr. Cristina Anton and Iain Smith.

    Parameters
    ----------
    data : `FDataBasis` or `list` of `FDataBasis`
        a `FDataBasis` object that contains functional data fit to a set of
        basis functions, or a list of these
    K : `int` or `list` of `int`, default=np.arange(1,11)
        number of clusters to run the algorithm with. If given as a `list` or
        list-like, the algorithm will run with each unique number of clusters.
    model : `str` or `list` of `str`, default='AKJBKQKDK'
        the type of model to be used. `FunWeightClust` supports the following
        model names: `'AKJBKQKDK'`, `'AKJBQKDK'`, `'AKBKQKDK'`, `'AKBQKDK'`, 
        `'ABKQKDK'`, `'ABQKDK'`. Can be given with any capitilization.
    known : `list` of `ints` or `np.NaNs`, default=None
        a vector given known clustering of data. Values that are not known
        should be given as `np.NaN`. When not None, FunWeightClust will perform
        classification. If all values are given in known, then FunWeightClust will 
        perform parameter estimation.
    dfstart : `int`, default=50
        the degrees of freedom given as an `int` used to initialize the 
        t-distribution.
    dfupdate : {'approx', 'numeric'}, default='approx'
        given as either `'numeric'` or `'approx'`. Approx is the default and results
        in using a closed form approximation. Numeric makes use of the `scipy` 
        function `brentq`.
    dfcontr : {'yes', 'no'}, default='yes'
        given as either `'yes'` or `'no'`. When yes, the degrees of freedom
        between clusters remains the same. If no, they can be different.
    threshold : `float` or `list` of `floats`, default=0.1
        the threshold of the Cattell scree-test used for selecting the
        group specific intrinsic dimensions
    itermax: `int`, default=200
        the number of iterations that the algorithm is allowed to perform
        before returning that it diverged.
    eps: `float`, default=1.e^-6
        threshold for convergence of the algorithm
    init: {'kmeans', 'random', 'mini-em', 'vector'}, default='kmeans'
        the method of initializing the clusters. Options are: `'kmeans'`,
        `'random'`, `'mini-em'` and `'vector'`.
    criterion : {'bic', 'icl'}, default='bic'
        criterion used for model selection. Default is `'bic'` but `'icl'`
        can be used
    d_select: {'cattell', 'bic', 'grid'}, default='cattell'
        methods of selecting intrinsic dimensions of each group. Default
        is `'cattell'`, but `'bic'` and `'grid'` can also be used.
    init_vector: `list` of `ints`, default=None
        vector containing user-supplied cluster initialization. Used
        only when `init='vector'`
    show: `bool`, default=True
        use `show=False` to turn off the information displayed when the 
        function finishes.
    mini_nb: `list` of `ints`, default=[5,10]
        list-like object of `ints` of length 2 used only when
        `init='mini-em'`. First value gives the number of times the algorithm is
        repeated, while the second gives the maximum iterations. This will give the
        initialization that maximizes the log-likelihood.
    min_individuals: `int`, default=4
        sets the minimum allowed population of a class. If a
        class contains less than the value of min_indivudals, then that run of the
        algorithm is terminated and the string `"pop<min_indiv"` is returned as the
        result of that combination of parameters.
    mc_cores: `int`, default=1
        number of CPU cores to use when using multi-processing. Information
        on algorithm run-time cannot be shown when using this.
    nb_rep: `int`, default=2
        number of times each combination of parameters will be run
    keepAllRes: `bool`, default=False
        returns results from each algorithm run
    kmeans_control: `dict`, default={'n_init':1, 'max_iter':10, 'algorithm':'lloyd'}
        parameters to be used with init='kmeans'. Must specify
        `'n_init'`, `'max_iter'`, and `'algorithm'`. Uses `scikit-learn's` 
        `kmeans` function.
    d_max: `float`, default=100
        maximum number of intrinsic dimensions that can be computed. May
        speed up algorithm if intrinsic dimensions are signifcantly large.
    d-range: `int` or `list` of `ints`, default=2
        list of values to be used for the intrinsic dimension of each
        group when `d_select='grid'`.
    verbose: bool, default=True
        setting to `True` (default) will display run-time estimation as the
        program exectutes.
    '''

    com_dim = None
    noise_ctrl = 1.e-8

    if not isinstance(datax, skfda.FDataBasis):
        MULTI = True
    
    data = datax.copy()
    _T_hddc_control(locals())

    model = _T_hdc_getTheModel(model, all2models=True)
    modely = _T_hdc_getTheModely(modely, all2models=True)

    if init == "random" and nb_rep < 20:
        nb_rep = 20

    if mc_cores > 1:
        verbose = False

    BIC = []
    ICL = []

    fdobj = datax.copy()
    fdobjy= datay.copy()

    if isinstance(fdobj, skfda.FDataBasis):
        x = fdobj.coefficients
        p = x.shape[1]

        W = np.eye(p)
        W_m = scil.cholesky(W)
        dety = scil.det(W)
        Wlist = {'W': W, 'W_m': W_m, 'dety':dety}
    else:
        x = fdobj[0].coefficients

        for i in range(1, len(fdobj)):
            x = np.c_[x, fdobj[i].coefficients]

        p = x.shape[1]

        W_fdobj = []
        for i in range(len(datax)):
            W_fdobj.append(skfda.misc.inner_product_matrix(datax[i].basis, datax[i].basis))

        prow = W_fdobj[-1].shape[0]
        pcol = len(datax)*prow
        W1 = np.c_[W_fdobj[-1], np.zeros((prow, pcol-W_fdobj[-1].shape[1]))]
        W_list = {}

        for i in range(1, len(datax)):
            W2 = np.c_[np.zeros((prow, (i)*W_fdobj[-1].shape[1])),
                    W_fdobj[i],
                    np.zeros((prow, pcol - (i+1) * W_fdobj[-1].shape[1]))]
            W_list[f'{i-1}'] = W2

        W_tot = np.concatenate((W1,W_list[f'{0}']))
        if len(datax) > 2:
            for i in range(1, len(datax)-1):
                W_tot = np.concatenate((W_tot, W_list[f'{i}']))

        W_tot[W_tot < 1.e-15] = 0
        W_m = scil.cholesky(W_tot)
        dety = scil.det(W_tot)
        Wlist = {'W':W_tot, 'W_m': W_m, 'dety': dety}


    
    if not(type(threshold) == list or type(threshold) == np.ndarray):
        threshold = np.array([threshold])

    if not(type(K) == list or type(K) == np.ndarray):
        K = np.array([K])

    if len(np.unique(K)) != len(K):
        warnings.warn("The number of clusters, K, should be unique (repeated values will be removed)")
        K = np.sort(np.unique(K))

    mkt_list = {}
    if d_select == 'grid':
        if len(K) > 1:
            raise ValueError("When using d_select='grid, K must be only one value (ie. not a list)")
        
        for i in range(K[0]):
            mkt_list[f'd{i}'] = [str(d_range)]
        mkt_list.update({'model':model, 'modely': modely, 'K':[str(a) for a in K], 'threshold':[str(a) for a in threshold]})

        

    else:
        for i in range(np.max(K)):
            mkt_list[f'd{i}'] = ['2']
        mkt_list.update({'model':model, 'modely': modely, 'K':[str(a) for a in K], 'threshold':[str(a) for a in threshold]})

        
    
    mkt_expand = ParameterGrid(mkt_list)
    mkt_expand = list(mkt_expand)
    repeat = mkt_expand.copy()
    for i in range(nb_rep-1):
        mkt_expand = np.concatenate((mkt_expand, repeat))
    
    model = [a['model'] for a in mkt_expand]
    modely = [a['modely'] for a in mkt_expand]

    K = [int(a['K']) for a in mkt_expand]
    d = {}

    for i in range(max(K)):
        d[f'{i}'] = [a[f'd{i}'] for a in mkt_expand]
    

    #Pass in dict from mkt_expand
    def hddcWrapper(mkt, verbose, start_time = 0, totmod=1):
        if verbose:
            modelNo = mkt[1]
            mkt = mkt[0]
        model = mkt['model']
        modely = mkt['modely']
        K = int(mkt['K'])
        threshold = float(mkt['threshold'])

        d_set = np.repeat(2, K)
        for i in range(K):
            d_set[i] = int(mkt[f'd{i}'])

        try:
            res = _T_funhddc_main1(fdobj=fdobj, fdobjy=fdobjy, wlist=Wlist, K=K, dfstart=dfstart, dfupdate=dfupdate, dfconstr=dfconstr, model=model, modely=modely,
                                    itermax=itermax, threshold=threshold,method=d_select, eps=eps, init=init, init_vector=init_vector,
                                    mini_nb=mini_nb, min_individuals=min_individuals, noise_ctrl=noise_ctrl,com_dim=com_dim, 
                                    kmeans_control=kmeans_control, d_max=d_max, d_set=d_set, known=known)
            if verbose:
                _T_estimateTime(stage=modelNo, start_time=start_time, totmod=totmod)

        except Exception as e:
            raise e
        
        return res
    

    nRuns = len(mkt_expand)
    if nRuns < mc_cores:
        mc_cores = nRuns

    max_cores = multi.cpu_count()
    if mc_cores > max_cores:
        warnings.warn(f"mc_cores was set to a value greater than the maximum number of cores on this system.\nmc_cores will be set to {max_cores}")
        mc_cores = max_cores

    start_time = time.process_time()

    if mc_cores == 1:
        if verbose:
            _T_estimateTime("init")
            mkt_expand = np.c_[mkt_expand, np.arange(0, len(mkt_expand))]

        res = [hddcWrapper(a, verbose, start_time, len(mkt_expand)) for a in mkt_expand]

    else:
        try:
            p = multi.Pool(mc_cores)

            models = [mkt['model'] for mkt in mkt_expand]
            modelys = [mkt['modely'] for mkt in mkt_expand]
            Ks = [int(mkt['K']) for mkt in mkt_expand]
            thresholds = [float(mkt['threshold']) for mkt in mkt_expand]
            d_sets = []
            for i in range(len(Ks)):
                d_temp = []
                for j in range(Ks[i]):
                    d_temp.append(int(mkt_expand[i][f'd{j}']))
                d_sets.append(d_temp)
            
            with p:            
                params = [(fdobj, fdobjy, Wlist, Ks[i], dfstart, dfupdate, dfconstr, models[i], modelys[i], itermax, thresholds[i], d_select, eps, init, init_vector, mini_nb, min_individuals, noise_ctrl, com_dim, kmeans_control, d_max, d_sets[i], known, testing, t) for i in range(len(models))]
                res = p.starmap_async(_T_funhddc_main1, params).get()

        except Exception as e:
            raise Exception("An error occurred while trying to use parallel. Try with mc_cores = 1 and try again").with_traceback(e.__traceback__)

    if verbose:
        mkt_expand = mkt_expand[:,0]
    res = np.array(res)
    loglik_all = np.array([x.loglik if isinstance(x, FunWeightClust) else - np.Inf for x in res])
    comment_all = np.array(["" if isinstance(x, FunWeightClust) else x for x in res])

    threshold = np.array([float(x['threshold']) for x in mkt_expand])
    if np.all(np.invert(np.isfinite(loglik_all))):
        warnings.warn("All models diverged")


        return {'model': model, 'K': K, 'threshold':threshold, 'LL':loglik_all, 'BIC': None, 'comment': comment_all}

    n = len(mkt_expand)

    modelKeep = np.arange(0, len(res))


    loglik_all = loglik_all[modelKeep]
    comment_all = comment_all[modelKeep]
    chosenRes = res[modelKeep]    
    
    bic = [res.bic if isinstance(res, FunWeightClust) else -np.Inf for res in chosenRes]
    icl = [res.icl if isinstance(res, FunWeightClust) else -np.Inf for res in chosenRes]
    allComplex = [res.complexity if isinstance(res, FunWeightClust) else -np.Inf for res in chosenRes]
    model = np.array(model)[modelKeep]
    modely = np.array(modely)[modelKeep]
    threshold = np.array(threshold)[modelKeep]
    K = np.array(K)[modelKeep]
    d_keep = {}
    for i in range(np.max([int(x) for x in K])):
        d_keep[f'{i}'] = np.array([int(x[f'd{i}']) for x in np.array(mkt_expand)[modelKeep]])

    CRIT = bic if criterion == 'bic' else icl
    resOrdering = np.argsort(CRIT)[::-1]

    qui = np.nanargmax(CRIT)
    bestCritRes = chosenRes[qui]
    bestCritRes.criterion = criterion

    bestCritRes.complexity_all = [('_'.join(mkt_expand[modelKeep[i]].values()), allComplex[i]) for i in range(len(mkt_expand))]
    if show:
        if n > 1:
            print("FunWeightClust: \n")

        printModel = np.array([x.rjust(max([len(a) for a in model])) for x in model])
        printModely = np.array([x.rjust(max([len(a) for a in modely])) for x in modely])
        printK = np.array([str(x).rjust(max([len(str(a)) for a in K])) for x in K])
        printTresh = np.array([str(x).rjust(max([len(str(a)) for a in threshold])) for x in threshold])
        resout = np.c_[printModel[resOrdering], printModely[resOrdering], printK[resOrdering], printTresh[resOrdering], _T_addCommas((np.array(bestCritRes.complexity_all)[resOrdering])[:,1].astype(float)), _T_addCommas(np.array(CRIT)[resOrdering])]
        resout = np.c_[np.arange(1,len(mkt_expand)+1), resout]

        resout = np.where(resout != "-inf", resout, 'NA')
        if np.any(np.nonzero(comment_all != '')[0]): 
            resout = np.c_[resout, comment_all[resOrdering]]
            resPrint = pd.DataFrame(data = resout[:,1:], columns=['Model', 'ModelY', 'K', 'Threshold', 'Complexity', criterion.upper(), 'Comment'], index=resout[:, 0])
        else:
            resPrint = pd.DataFrame(data = resout[:,1:], columns=['Model', 'ModelY', 'K', 'Threshold', 'Complexity', criterion.upper()], index=resout[:, 0])

        print(resPrint)
        print(f'\nSelected model {bestCritRes.model}-{bestCritRes.modely} with {bestCritRes.K} clusters')
        print(f'\nSelection Criterion: {criterion}\n')

    allCriteria = resPrint
    bestCritRes.allCriteria=allCriteria

    if keepAllRes:
        allRes = chosenRes
        bestCritRes.allRes=allRes

    return bestCritRes

def _T_funhddc_main1(fdobj, fdobjy, wlist, K, dfstart, dfupdate, dfconstr, model, modely,
                     itermax, threshold, method, eps, init, init_vector,
                     mini_nb, min_individuals, noise_ctrl, com_dim,
                     kmeans_control, d_max, d_set, known, testing=False, r_t = None):
    
    """
    Description
    -----------
        _T_funhddc_main1 completes the EM algorithm loop corresponding to model and modely, and returns the designated values

    Parameters
    ----------
    fdobj: `FDataBasis` or `list` of `FDataBasis`
        a `FDataBasis` object that contains functional data fit to a set of
        basis functions, or a list of these
    fdobjy: `FDataBasis` or `list` of `FDataBasis`
        a `FDataBasis` object that contains functional data fit to a set of
        basis functions, or a list of these
    Wlist: dict {W, W_m, dety}
        a dictionary that contains:
            W: (N, N) or (p, p)
            W_m: (N, N) or (p, p)
                The Cholesky of W
            dety: (float)
                the Determinant of W
    K: `int` or `list` of `int`, default=np.arange(1,11)
        number of clusters to run the algorithm with. If given as a `list` or
        list-like, the algorithm will run with each unique number of clusters.
    dfstart: `int`, default=50
        the degrees of freedom given as an `int` used to initialize the 
        t-distribution.
    dfupdate: {'approx', 'numeric'}, default='approx'
        given as either `'numeric'` or `'approx'`. Approx is the default and results
        in using a closed form approximation. Numeric makes use of the `scipy` 
        function `brentq`.
    dfcontr: {'yes', 'no'}, default='yes'
        given as either `'yes'` or `'no'`. When yes, the degrees of freedom
        between clusters remains the same. If no, they can be different.
    model: `str` or `list` of `str`, default='AKJBKQKDK'
        the type of model to be used. `FunWeightClust` supports the following
        model names: `'AKJBKQKDK'`, `'AKJBQKDK'`, `'AKBKQKDK'`, `'AKBQKDK'`, 
        `'ABKQKDK'`, `'ABQKDK'`. Can be given with any capitilization.
    modely: `str` or `list` of `str`, default='AKJBKQKDK'
        the type of model to be used. `FunWeightClust` supports the following
        model names: `'AKJBKQKDK'`, `'AKJBQKDK'`, `'AKBKQKDK'`, `'AKBQKDK'`, 
        `'ABKQKDK'`, `'ABQKDK'`. Can be given with any capitilization.
    itermax: `int`, default=200
        the number of iterations that the algorithm is allowed to perform
        before returning that it diverged.
    threshold : `float` or `list` of `floats`, default=0.1
        the threshold of the Cattell scree-test used for selecting the
        group specific intrinsic dimensions
    method: 'str'
        Either 'bic,' 'cattell,' or 'grid'
    eps: `float`, default=1.e^-6
        threshold for convergence of the algorithm
    init: {'kmeans', 'random', 'mini-em', 'vector'}, default='kmeans'
        the method of initializing the clusters. Options are: `'kmeans'`,
        `'random'`, `'mini-em'` and `'vector'`.
    init_vector: `list` of `ints`, default=None
        vector containing user-supplied cluster initialization. Used
        only when `init='vector'`
    mini_nb: `list` of `ints`, default=[5,10]
        list-like object of `ints` of length 2 used only when
        `init='mini-em'`. First value gives the number of times the algorithm is
        repeated, while the second gives the maximum iterations. This will give the
        initialization that maximizes the log-likelihood.
    min_individuals: `int`, default=4
        sets the minimum allowed population of a class. If a
        class contains less than the value of min_indivudals, then that run of the
        algorithm is terminated and the string `"pop<min_indiv"` is returned as the
        result of that combination of parameters.
    noise_ctrl: 'bool'
    com_dim: 'int'
    kmeans_control: `dict`, default={'n_init':1, 'max_iter':10, 'algorithm':'lloyd'}
        parameters to be used with init='kmeans'. Must specify
        `'n_init'`, `'max_iter'`, and `'algorithm'`. Uses `scikit-learn's` 
        `kmeans` function.
    d_max: `float`, default=100
        maximum number of intrinsic dimensions that can be computed. May
        speed up algorithm if intrinsic dimensions are signifcantly large.
    d_set: `int` or `list` of `ints`, default=2
        list of values to be used for the intrinsic dimension of each group when `d_select='grid'`.
    known: `list` of `ints` or `np.NaNs`, default=None
        a vector given known clustering of data. Values that are not known should be given as `np.NaN`. When not None, FunWeightClust will perform classification. If all values are given in known, then FunWeightClust will perform parameter estimation.
    
    Return
    ------
    tfun: FunWeightClust object
    """
    
    np.seterr(all='ignore')

    
    if(type(fdobj) == skfda.FDataBasis):
       MULTI = False
       x = fdobj.coefficients
       data = x

    else:
        #Multivariate
        if len(fdobj) > 1:
            MULTI = True
            data = []
            x = fdobj[0].coefficients
            for i in range(1, len(fdobj)):
                x = np.c_[x, fdobj[i].coefficients]

            
        #univariate
        else:
            x = fdobj[0].coefficients
            

    if(type(fdobjy) == skfda.FDataBasis):
        MULTI = False
        y = fdobjy.coefficients
        datay = y

    else:

        if len(fdobjy) > 1:
            MULTI = True
            datay = []
            y = fdobjy[0].coefficients
            datay.append(fdobjy[0].coefficients)

            for i in range(1, len(fdobjy)):
                y = np.c_[y, fdobj[i].coefficients]
                datay.append(fdobjy[i].coefficients)
            
            datay = np.array(datay)
        else:
            y = fdobjy[0].coefficients

    N = x.shape[0]
    p = x.shape[1]
    q = y.shape[1]

    W = wlist['W']


    ones_row = np.ones((1, N))
    DATA = x.T
    intermediate_bigDATA = W@(DATA)
    bigDATA = np.vstack((intermediate_bigDATA, ones_row))

    com_ev = None

    d_max = min(N,p,d_max)
    databig = np.hstack((x, datay))

    #classification
    n = N
    if(known is None):
        clas = 0
        kno = None
        test_index = None

    else:

        if len(known) != N:
            raise ValueError("Known classifications vector not the same length as the number of samples (see help file)")
        

        else:
            if (not np.isnan(np.sum(known))):
                test_index = np.linspace(0, n-1, n).astype(int)
                kno = np.repeat(1, n)
                unkno = (kno-1)*(-1)
                K = len(np.unique(known))
                init_vector = known.astype(int)
                init = "vector"

            else:

                training = np.where((np.invert(np.isnan(known))))
                test_index = training
                kno = np.zeros(n).astype(int)
                kno[test_index] = 1
                unkno = np.atleast_2d((kno - 1)*(-1))

            clas = 1

    if K > 1:
        t = np.zeros((n, K))
        tw = np.zeros((n, K))

        match init:
            case "vector":
                if clas > 0:
                    cn1 = len(np.unique(known[test_index]))
                    matchtab = np.repeat(-1, K*K).reshape((K,K))
                    temp = pd.crosstab(index = known, columns = init_vector).reindex(columns=np.arange(K), index=np.arange(cn1))
                    matchtab[0:cn1, 0:K] = temp.to_numpy()
                    table = _Table(data=matchtab, rownames=temp.index, colnames = temp.columns)
                    
                    rownames = np.concatenate(([table.rownames, np.nonzero(np.invert(np.isin(np.arange(0,K,1), np.unique(known[test_index]))))[0]]))
                    table.rownames = rownames
                    matchit = np.repeat(-1, K)

                    while(np.max(table.data)>0):
                        ij = int(np.nonzero(table.data == np.max(table.data))[1][0])

                        ik = np.argmax(table.data[:,ij])
                        matchit[ij] = table.rownames[ik]
                        table.data[:,ij] = np.repeat(-1, K)
                        table.data[ik] = np.repeat(-1, K)

                    matchit[np.nonzero(matchit == -1)[0]] = np.nonzero(np.invert(np.isin(np.arange(K), np.unique(matchit))))[0]
                    initnew = init_vector.copy()
                    for i in range(0, K):
                        initnew[init_vector == i] = matchit[i]

                    init_vector = initnew

                for i in range(0, K):
                    t[np.nonzero(init_vector == i)[0], i] = 1


            case "kmeans":
                kmc = kmeans_control
                km = clust.KMeans(n_clusters = K, max_iter = kmeans_control['max_iter'], n_init = kmeans_control['n_init'], algorithm=kmeans_control['algorithm'])
                cluster = km.fit_predict(databig)

                if clas > 0:
                    cn1 = len(np.unique(known[test_index]))
                    matchtab = np.repeat(-1, K*K).reshape((K,K))
                    matchtab = _Table(data = matchtab)
                    temptab = pd.crosstab(known, cluster, rownames=['known'], colnames=['init']).reindex(columns=np.arange(K), index=np.arange(cn1))
                    matchtab.data[0:cn1, 0:K] = temptab.to_numpy()
                    matchtab.rownames = np.concatenate((temptab.index, np.nonzero(np.invert(np.isin(np.arange(K), np.unique(known[test_index]))))[0]))
                    matchtab.colnames = temptab.columns
                    matchit = np.repeat(-1, K)

                    while(np.max(matchtab.data)>0):
                        ij = int(np.nonzero(matchtab.data == np.max(matchtab.data))[1][0])

                        ik = np.argmax(matchtab.data[:,ij])
                     
                        matchit[ij] = matchtab.rownames[ik]
                        matchtab.data[:,ij] = np.repeat(-1, K)
                        matchtab.data[ik] = np.repeat(-1, K)

                    matchit[np.nonzero(matchit == -1)[0]] = np.nonzero(np.invert(np.isin(np.arange(K), np.unique(matchit))))[0]
                    knew = cluster.copy()
                    for i in range(0, K):
                        knew[cluster == i] = matchit[i]
                    
                    cluster = knew

                for i in range(K):
                    t[np.nonzero(cluster == i)[0], i] = 1.

            #skip trimmed kmeans
            case "tkmeans":
                raise ValueError("tkmeans not supported")

            case "mini-em":
                prms_best = 1
                for i in range(0, mini_nb[0]):
                    prms = _T_funhddc_main1(fdobj=fdobj, fdobjy=fdobjy, wlist=wlist, K=K, known=known, dfstart=dfstart,
                                            dfupdate=dfupdate, dfconstr=dfconstr, model=model, modely=modely,
                                            threshold=threshold, method=method, eps=eps,
                                            itermax=mini_nb[1], init_vector=0, init="random",
                                            mini_nb=mini_nb, min_individuals=min_individuals,
                                            noise_ctrl=noise_ctrl, kmeans_control=kmeans_control,
                                            com_dim=com_dim, d_max=d_max, d_set=d_set)
                    if isinstance(prms, FunWeightClust):
                        if not isinstance(prms_best, FunWeightClust):
                            prms_best = prms
                        
                        elif prms_best.loglik < prms.loglik:
                            prms_best = prms

                if not isinstance(prms, FunWeightClust):
                    return "mini-em did not converge"
                
                t = prms_best.posterior

                if clas > 0:
                    cluster = np.argmax(t, axis=1)

                    cn1 = len(np.unique(known[test_index]))
                    matchtab = np.repeat(-1, K*K).reshape((K,K))
                    matchtab = _Table(data = matchtab)
                    temptab = pd.crosstab(known, cluster, rownames=['known'], colnames=['init']).reindex(columns=np.arange(K), index=np.arange(cn1))
                    matchtab.data[0:cn1, 0:K] = temptab.to_numpy()
                    matchtab.rownames = np.concatenate((temptab.index, np.nonzero(np.invert(np.isin(np.arange(K), np.unique(known[test_index]))))[0]))
                    matchtab.colnames = temptab.columns
                    matchit = np.repeat(-1, K)

                    while(np.max(matchtab.data)>0):
                        ij = int(np.nonzero(matchtab.data == np.max(matchtab.data))[1][0])

                        ik = np.argmax(matchtab.data[:,ij])
                     
                        matchit[ij] = matchtab.rownames[ik]
                        matchtab.data[:,ij] = np.repeat(-1, K)
                        matchtab.data[ik] = np.repeat(-1, K)

                    matchit[np.nonzero(matchit == -1)[0]] = np.nonzero(np.invert(np.isin(np.arange(K), np.unique(matchit))))[0]
                    knew = cluster.copy()
                    for i in range(0, K):
                        knew[cluster == i] = matchit[i]
                    
                    cluster = knew

                    for i in range(0, K):
                        t[np.nonzero(cluster == i)[0], i] = 1.

            case "random":
                rangen = np.random.default_rng()
                t = rangen.multinomial(n=1, pvals=np.repeat(1/K, K), size = n)
                compteur = 1

                #sum columns
                while(np.min(np.sum(t, axis=0)) < 1 and compteur + 1 < 5):
                    compteur += 1
                    t = rangen.multinomial(n=1, pvals=np.repeat(1/K, K), size=n)

                if(np.min(np.sum(t, axis=0)) < 1):
                    raise ValueError("Random initialization failed (n too small)")
                if clas > 0:
                    cluster = np.argmax(t, axis=1)
                    cn1 = len(np.unique(known[test_index]))
                    matchtab = np.repeat(-1, K*K).reshape((K,K))
                    matchtab = _Table(data=matchtab)
                    temptab = pd.crosstab(known, cluster, rownames=['known'], colnames=['init']).reindex(columns=np.arange(K), index=np.arange(cn1))
                    matchtab.data[0:cn1, 0:K] = temptab.to_numpy()
                    matchtab.rownames = np.concatenate((temptab.index, np.nonzero(np.invert(np.isin(np.arange(K), np.unique(known[test_index]))))[0]))
                    matchtab.colnames = temptab.columns
                    matchit = np.repeat(-1, K)

                    while(np.max(matchtab.data)>0):
                        ij = int(np.nonzero(matchtab.data == np.max(matchtab.data))[1][0])

                        ik = np.argmax(matchtab.data[:,ij])
                     
                        matchit[ij] = matchtab.rownames[ik]
                        matchtab.data[:,ij] = np.repeat(-1, K)
                        matchtab.data[ik] = np.repeat(-1, K)

                    matchit[np.nonzero(matchit == -1)[0]] = np.nonzero(np.invert(np.isin(np.arange(K), np.unique(matchit))))[0]
                    knew = cluster.copy()
                    for i in range(0, K):
                        knew[cluster == i] = matchit[i]
                    
                    cluster = knew

                    for i in range(0, K):
                        t[np.nonzero(cluster == i)[0], i] = 1.

                    
    else:
        t = np.ones(shape = (n, 1))
        tw = np.ones(shape = (n, 1))

    if clas > 0:
        t = np.atleast_2d(unkno).T*t
        
        for i in range(0, n):
            if kno[i] == 1:
                t[i, int(known[i])] = 1.

    nux = np.repeat(dfstart, K)

    I = 0
    likely = []
    test = np.Inf
    first_t = t.copy()
    if testing == True and r_t.all() != None:
        t = r_t

    while(I < itermax and test >= eps):
        if K > 1:
            if(np.isnan(np.sum(t))):
                return "t matrix contatins NaNs/Nones"
            

            if(np.any(np.sum(t>(1/K), axis=0) < min_individuals)):
                return "pop<min_individuals"

        m = _T_funhddt_m_step1(fdobj, bigDATA, fdobjy, wlist, N, p, q, K, t, model, str(modely), threshold, method, noise_ctrl, d_set, com_dim, d_max)
        to = _T_funhddt_e_step1(fdobj, bigDATA, fdobjy, wlist, N, p, q, m, clas, known, kno)
        
        
        L = to['L']
        t = to['t']

        likely.append(L)

        if(I == 1):
            test = abs(likely[I] - likely[I-1])
        elif I > 1:
            lal = (likely[I] - likely[I-1])/(likely[I-1] - likely[I-2])
            lbl = likely[I-1] + (likely[I] - likely[I-1])/(1.0/lal)
            test = abs(lbl - likely[I-1])
        
        I += 1

    #a
    if np.isin(model, np.array(["AKBKQKDK", "AKBQKDK"])):
        a = m['a'][:,0]

    
    elif np.isin(model, np.array(["ABKQKDK", "ABQKDK"])):
        a = m['a'][0]

    else:
        a = m['a']


    #b
    if np.isin(model, np.array(["AKJBQKDK", "AKBQKDK", "ABQKDK"])):
        b = np.array([m['b'][0]])
    else:
        b = m['b']
    
    #d
    d = m['d']


    #mu
    mu = m['mu']

    prop = m['prop']
    gam = m['gam']
    covy = m['covy']
    icovy = m['icovy']
    logi = m['logi']
    complexity = _T_hdc_getComplexityt(m, p, q, dfconstr)
    cl = np.argmax(t, axis=1)
    converged = test < eps
    nux = "NUX"
    params = {'wlist': wlist, 'model':model, 'modely': modely, 'K':K, 'd':d,
                'a':a, 'b':b, 'mu':mu, 'prop':prop, 'nux':nux, 'ev': m['ev'],
                'Q': m['Q'], 'Q1':m['Q1'], 'fpca': m['fpcaobj'], 
                'loglik':likely[-1], 'loglik_all': likely, 'posterior': t,
                'class': cl, 'com_ev': com_ev, 'N':n, 'complexity':complexity,
                'threshold': threshold, 'd_select': method, 
                'converged': converged, "index": test_index, 'gam': gam, 'covy': covy, 'icovy': icovy, 'logi':logi}

    bic_icl = _T_hdclassift_bic(params, p, q, dfconstr)
    params['BIC'] = bic_icl["bic"]
    params["ICL"] = bic_icl['icl']

    try:
        base = fdobj.basis
    except:
        base = fdobj[0].basis
    tfunobj = FunWeightClust(Wlist=params['wlist'], model=params['model'], modely=params['modely'], K=params['K'], d=params['d'], 
                        a=params['a'], b=params['b'], mu=params['mu'], prop=params['prop'], nux=params['nux'],
                        ev=params['ev'], Q=params['Q'], Q1=params['Q1'], fpca=params['fpca'],
                        loglik=params['loglik'], loglik_all=params['loglik_all'], posterior=params['posterior'],
                        cl=params['class'], com_ev=params['com_ev'], N=params['N'], complexity=params['complexity'],
                        threshold=params['threshold'], d_select=params['d_select'], converged=params['converged'], 
                        index=params['index'], bic=params['BIC'], icl=params['ICL'], basis=base, gam=params['gam'], covy=params['covy'], icovy=params['icovy'], logi=params['logi'], first_t = first_t)
    return tfunobj



from py_mixture import C_rmahalanobis
from imahalanobis import imahalanobis
def _T_funhddt_e_step1(fdobj, bigDATA, fdobjy, Wlist, N, p, q, par, clas=0, known=None, kno=None):
    """
    Description
    -----------
    This function completes the expectation section of the EM algorithm

    Parameters
    ----------
    fdobj: `FDataBasis` or `list` of `FDataBasis`
        a `FDataBasis` object that contains functional data fit to a set of
        basis functions, or a list of these
    bigDATA: np.ndarray((p+1, N), dtype=np.float64)
        a product of the matrix W found in Wlist, and the coefficient matrix found in fdobj with a row of ones at the bottom
    fdobjy: `FDataBasis` or `list` of `FDataBasis`
        a `FDataBasis` object that contains functional data fit to a set of
        basis functions, or a list of these
    Wlist: dict {W, W_m, dety}
        a dictionary that contains:
            W: (N, N) or (p, p)
            W_m: (N, N) or (p, p)
                The Cholesky of W
            dety: (float)
                the Determinant of W
    N: 'int'
        Number of rows in the coefficient matrix in fdobj
    p: 'int'
        Number of columns in the coefficient matrix in fdobj
    q: 'int'
        Number of columns in the coefficient matrix in fdobjy
    par: 'dict'
        Output from the M step
    
    Returns
    -------
    ans: dict {t, mah_pen, mah_pen1, K_pen}
        t: np.ndarray[np.float64_t, ndim=2], (N, K)
            Matrix that contains the probabilities of a curve belonging to one of K groups
        mah_pen: np.ndarray[np.float64_t, ndim=2], (N, p)
            Matrix that contains imahalanobis distances
        mah_pen1: np.ndarray[np.float64_t, ndim=2], (N, p)
            Matrix that contains rmahalanobis distances
        K_pen: np.ndarray[np.float64_t, ndim=2], (p, N)
    """

    if(type(fdobj) == skfda.FDataBasis):
       MULTI = False
       x = fdobj.coefficients

    else:
        #Multivariate
        if len(fdobj) > 1:
            MULTI = True
            data = []
            x = fdobj[0].coefficients
            data.append(fdobj[0].coefficients)
            for i in range(1, len(fdobj)):
                x = np.c_[x, fdobj[i].coefficients]
                data.append(fdobj[i].coefficients)

            data = np.array(data)
        #univariate
        else:
            x = fdobj[0].coefficients
            x = np.reshape(x, (N, p))
            

    if(type(fdobjy) == skfda.FDataBasis):
        MULTI = False
        y = fdobjy.coefficients

    else:

        if len(fdobjy) > 1:
            MULTI = True
            datay = []
            y = fdobjy[0].coefficients
            datay.append(fdobjy[0].coefficients)

            for i in range(1, len(fdobjy)):
                y = np.c_[y, fdobj[i].coefficients]
                data.append(fdobjy[i].coefficients)
            
            datay = np.array(datay)
        else:
            y = fdobjy[0].coefficients
            y = np.reshape(y, (N, q))

    pqp = p+1
    bigx = (bigDATA.T)
    K = par["K"]
    a = par["a"]
    b = par["b"]
    mu = par["mu"]
    d = par["d"]
    prop = par["prop"]
    Q = par["Q"]
    Q1 = par["Q1"].copy()
    icovy = par["icovy"]
    ldetcov = par["logi"]
    gam = par["gam"]
    dety = Wlist["dety"]
    
    b[b<1e-6] = 1e-6
    
    if clas > 0:
        unkno = np.atleast_2d((kno-1)*(-1)).T

    tw = np.zeros((N, K)) 
    mah_pen = np.zeros((K, N))
    mah_pen1 = np.zeros((K, N))
    K_pen = np.zeros((N, K))
    ft = np.zeros((N, K))

    s = np.zeros(K)

    for i in range(0,K):
        
        s[i] = np.sum(np.log(a[i, 0:int(d[i])]))
        Qk = Q1[f"{i}"]
        
        aki = np.sqrt(np.diag(np.concatenate((1/a[i, 0:int(d[i])],np.repeat(1/b[i], p-int(d[i])) ))))
        muki = mu[i]
        Wki = Wlist["W_m"]

        pp = x.shape[1]
        pN = x.shape[0]
        pdi = aki.shape[1]
        new_x = x.copy()


        ans = imahalanobis(new_x, muki, Wki, Qk, aki, pp, pN, pdi, np.zeros(N))
        mah_pen[i, :] = ans

        dety = Wlist["dety"]
        delta = np.zeros(N)
        mah_pen1[i, :] = C_rmahalanobis(N, pqp, q, K, i, bigx, y, gam[i, :, :], icovy[i, :, :], delta)
        
        pi = math.pi
        K_pen[:, i] = (-2 * np.log(prop[i])) + (p + q) * np.log(2 * np.pi) + s[i] - np.log(dety) + (p - d[i]) * np.log(b[i]) + mah_pen[i, :] + mah_pen1[i, :] + ldetcov[i]

    A = (-1/2)*K_pen
    L = np.sum(np.log(np.sum(np.exp(A - np.max(A, axis=1).reshape(-1, 1)), axis=1)) + np.max(A, axis=1))

    t = np.zeros((N, K))
    for j in range(0, K):
        new_K = K_pen.copy()
        new_K -= new_K[:, j][:, np.newaxis]
        new_K *= -1
        t[:, j] = 1 / np.sum((np.exp(new_K/2)), axis=1)

    if (clas > 0):
        t = unkno * t
        for i in range(N):
            if kno[i] == 1:
                t[i, known[i]] = 1
    return {'t': t, 'L': L, 'mah_pen': mah_pen.T, 'mah_pen1': mah_pen1.T, 'K_pen':K_pen}



from py_mixture import C_mstep
def _T_funhddt_m_step1(fdobj, bigDATA, fdobjy, Wlist, N, p, q, K, t, model, modely, threshold, method, noise_ctrl, d_set, com_dim=None, d_max=100):
    """
    Description
    -----------
    This function completes the maximization section of the EM algorithm

    Parameters
    ----------
        fdobj: `FDataBasis` or `list` of `FDataBasis`
            a `FDataBasis` object that contains functional data fit to a set of
            basis functions, or a list of these
        bigDATA: np.ndarray((p+1, N), dtype=np.float64)
            a product of the matrix W found in Wlist, and the coefficient matrix found in fdobj with a row of ones at the bottom
        fdobjy: `FDataBasis` or `list` of `FDataBasis`
            a `FDataBasis` object that contains functional data fit to a set of
            basis functions, or a list of these
        Wlist: dict {W, W_m, dety}
            a dictionary that contains:
                W: (N, N) or (p, p)
                W_m: (N, N) or (p, p)
                    The Cholesky of W
                dety: (float)
                    the Determinant of W
        N: 'int'
            Number of rows in the coefficient matrix in fdobj
        p: 'int'
            Number of columns in the coefficient matrix in fdobj
        q: 'int'
            Number of columns in the coefficient matrix in fdobjy
        K: 'int'
          number of clusters
        t: np.ndarray[np.float64_t, ndim=2], (N, K)
            Matrix that contains the probabilities of a curve belonging to one of K groups
        model: `str` or `list` of `str`, default='AKJBKQKDK'
            the type of model to be used. `FunWeightClust` supports the following
            model names: `'AKJBKQKDK'`, `'AKJBQKDK'`, `'AKBKQKDK'`, `'AKBQKDK'`, 
            `'ABKQKDK'`, `'ABQKDK'`. Can be given with any capitilization.
        modely: `str` or `list` of `str`, default='AKJBKQKDK'
            the type of model to be used. `FunWeightClust` supports the following
            model names: `'AKJBKQKDK'`, `'AKJBQKDK'`, `'AKBKQKDK'`, `'AKBQKDK'`, 
            `'ABKQKDK'`, `'ABQKDK'`. Can be given with any capitilization.
        threshold : `float` or `list` of `floats`, default=0.1
            the threshold of the Cattell scree-test used for selecting the
            group specific intrinsic dimensions
        method: 'str'
            Either 'bic,' 'cattell,' or 'grid'

    Returns
    -------
    result: dict
    """
    t = (np.reshape(t, (N, K)))

    if(type(fdobj) == skfda.FDataBasis):
       MULTI = False
       x = fdobj.coefficients

    else:
        #Multivariate
        if len(fdobj) > 1:
            MULTI = True
            data = []
            x = fdobj[0].coefficients
            data.append(fdobj[0].coefficients)
            for i in range(1, len(fdobj)):
                x = np.c_[x, fdobj[i].coefficients]
                data.append(fdobj[i].coefficients)

            data = np.array(data)
        #univariate
        else:
            x = fdobj[0].coefficients
            x = np.reshape(x, (N, p))
            

    if(type(fdobjy) == skfda.FDataBasis):
        MULTI = False
        y = fdobjy.coefficients

    else:

        if len(fdobjy) > 1:
            MULTI = True
            datay = []
            y = fdobjy[0].coefficients
            datay.append(fdobjy[0].coefficients)

            for i in range(1, len(fdobjy)):
                y = np.c_[y, fdobj[i].coefficients]
                data.append(fdobjy[i].coefficients)
            
            datay = np.array(datay)
        else:
            y = fdobjy[0].coefficients
            y = np.reshape(y, (N, q))

    bigx = np.transpose(bigDATA)
    bigx = np.reshape(bigx, (N, p+1))

    #Formula 24, nk
    n = np.sum(t, axis=0)

    d_max = min(N, p, d_max)
    #Formula 24, pik
    prop = n/N

    #Formula 25, muk
    mu = np.zeros((K, p))
    for i in range(0, K):
        mu[i, :] = np.sum(x*((t[:, i]).reshape(-1, 1)), axis=0) / n[i]
    

    ind = [np.where(row > 0)[0] for row in t]
    n_bis = np.arange(0,K)

    for i in range(0,K):
        n_bis[i] = len(ind[i])

    traceVect = np.zeros(K)

    ev = np.zeros((K, p))

    Q = {}
    fpcaobj = {}

    for i in range(0, K):
        if MULTI:
            valeurs_propres, cov, U = _T_mypcat_fd1_Multi(data, Wlist['W_m'], np.atleast_2d(t[:,i]))
        else:
            valeurs_propres, cov, U = _T_mypcat_fd1_Uni(x, Wlist['W_m'], np.atleast_2d(t[:,i]))

        
        traceVect[i] = np.sum(np.diag(valeurs_propres))
        ev[i] = np.sort(valeurs_propres)[::-1]
        Q[f'{i}'] = U
        fpcaobj[f'{i}'] = {'valeurs_propres': valeurs_propres, 'cov': cov, 'U':U}


    if model in ["AJBQD", "ABQD"]:
        d = np.repeat(com_dim, K)
    elif model in ["AKJBKQKD", "AKBKQKD", "ABKQKD", "AKJBQKD", "AKBQKD", "ABQKD"]:
        threshold_exceeded = (ev > noise_ctrl)
    
        repeated_indices = np.repeat(np.arange(1, ev.shape[1] + 1), K).reshape(ev.shape[1], K).T        
        max_indices = np.apply_along_axis(lambda x: np.argmax(x) + 1, 1, threshold_exceeded * repeated_indices)        
        dmax = np.min(max_indices) - 1
        
        if com_dim > dmax:
            com_dim = max(dmax, 1)
        
        d = np.full(K, com_dim)
    else:
        n = np.sum(t, axis=0)
        d = _T_hdclassif_dim_choice(ev, n, method, threshold, False, noise_ctrl, d_set)


    Q1 = Q.copy()

    for i in range(0, K):
        Q[f'{i}'] = Q[f'{i}'][:,0:d[i]]


    #Parameter a
    ai = np.full((K, max(d)), np.nan)
    if model in ['AKJBKQKDK', 'AKJBQKDK', 'AKJBKQKD', 'AKJBQKD']:
        for i in range(K):
            ai[i, :d[i]] = ev[i, :d[i]]
    elif model in ['AKBKQKDK', 'AKBQKDK', 'AKBKQKD', 'AKBQKD']:
        for i in range(K):
            ave_val = np.sum(ev[i, :d[i]]) / d[i]
            ai[i, :] = np.full(max(d), ave_val)
    elif model == "AJBQD":
        for i in range(K):
            ai[i, :] = ev[:d[0], 0]
    elif model == "ABQD":
        ai[:] = np.sum(ev[:d[0], 0]) / d[0]
    else:
        a = 0
        eps = np.sum(prop * d)
        for i in range(K):
            a += np.sum(ev[i, :d[i]]) * prop[i]
        ai = np.full((K, max(d)), a / eps)

    #Parameter b
    bi = np.repeat(np.NaN, K)
    denom = min(N, p)

    if model in ['AKJBKQKDK', 'AKBKQKDK', 'ABKQKDK']:
        for i in range(K):
            remainEV = traceVect[i] - np.sum(ev[i, 0:d[i]])
            bi[i] = remainEV/(p-d[i])
    elif model in ["ABQD", "AJBQD"]:
        remainEV = traceVect - np.sum(ev[:d[0], 0])
        # Compute bi
        bi[:K] = remainEV / (denom - d[0])
    else:
        b = 0
        eps = np.sum(prop*d)
        for i in range(K):
            remainEV = traceVect[i] - np.sum(ev[i, 0:d[i]])
            b = b+remainEV*prop[i]
        bi[0:K] = b/(min(N,p)-eps)

    gami = np.zeros((K, q*(p+1)), dtype=np.float64)
    covyi = np.zeros((K, q*q), dtype=np.float64)
    icovyi = np.zeros((K, q*q), dtype=np.float64)
    logi = np.zeros(K, dtype=np.float64)
    pqp = p+1


    gami, covyi, icovyi, logi = C_mstep(modely, N, pqp, q, K, prop, bigx, y, t, gami, covyi, icovyi, logi, mtol=1e-10, mmax=10)
    result = {'model':model, 'modely': modely, "K": K, "d":d, "a":ai, "b": bi, "mu":mu, "prop": prop, "ev":ev, "Q":Q, "fpcaobj":fpcaobj, "Q1":Q1, "gam": gami, "covy": covyi, "icovy": icovyi, "logi": logi, "x": x, "y": y, "ev": ev, "N": N}    
    
    return result        



def _T_mypcat_fd1_Uni(fdobj_coefficients, W_m, Ti):
    """
    Description
    -----------
    This function completes Principal Component Analysis on Univariate Functional Data

    Parameters
    ----------
        fdobj_coefficients: (N, p)
            a `FDataBasis` object that contains functional data fit to a set of
            basis functions
        W_m: (N, N) or (p, p)
            The Cholesky of W
        Ti: (N, 1)
            A column of the probability matrix t

    Returns
    -------
        valeurs_propres: (p, 1)
            Array containing the eigenvalues
        cov: (p, p)
            Covariance Matrix
        bj: (p, p)
            Principal Component Scores

    """

    coefmean = np.sum(np.transpose(Ti) @ (np.ones((1, fdobj_coefficients.shape[1]))) * fdobj_coefficients, axis=0) / np.sum(Ti)
    swept_fdobj_coefficients = fdobj_coefficients.copy()
    swept_fdobj_coefficients = swept_fdobj_coefficients.astype(np.float64)

    swept_fdobj_coefficients -= coefmean
    n = swept_fdobj_coefficients.shape[1]
    v = np.sqrt(Ti)
    M = np.repeat(1., n).reshape((n, 1))@(v)
    rep = (M * swept_fdobj_coefficients.T).T
    mat_cov = (rep.T@rep) / np.sum(Ti)
    cov = ((W_m@ mat_cov)@(W_m.T))
    
    if not np.all(np.abs(cov-cov.T) < 1.e-12):
        ind = np.nonzero(cov - cov.T > 1.e-12)
        for i in ind:
            cov[i] = cov.T[i]
    
    valeurs_propres, vecteurs_propres = np.linalg.eig(cov)

    sorted_indices = np.argsort(-np.abs(valeurs_propres))
    valeurs_propres = valeurs_propres[sorted_indices]
    vecteurs_propres = vecteurs_propres[:, sorted_indices]


    for i in range(len(valeurs_propres)):
        if np.imag(i) > 0:
            valeurs_propres[i] = 0
    bj = np.linalg.solve(W_m, np.eye(W_m.shape[0]))@np.ascontiguousarray(np.real(vecteurs_propres))

    

    return np.real(valeurs_propres), cov, bj   


@nb.njit
def _T_mypcat_fd1_Multi(data, W_m, Ti, corI):
    """
    Description
    -----------
    This function completes Principal Component Analysis on Multivariate Functional Data

    Parameters
    ----------
        data: list of 'FDataBasis objects' (N, p)
            a list of`FDataBasis` object that contains functional data fit to a set of
            basis functions
        W_m: (N, N) or (p, p)
            The Cholesky of W
        Ti: (N, 1)
            A column of the probability matrix t

    Returns
    -------
        valeurs_propres: (p, 1)
            Array containing the eigenvalues
        cov: (p, p)
            Covariance Matrix
        bj: (p, p)
            Principal Component Scores

    This function is not included in the R code, thus I could not describe corI
    """


    coefficients = data.reshape(data.shape[1], data.shape[-1]*data.shape[0])
    coefmean = np.zeros((coefficients.shape))

    for i in range(len(data)):
        for j in range(data[i].shape[-1]):

            coefmean[:, j] = np.sum(((np.ascontiguousarray(corI.TW)@np.atleast_2d(np.repeat(1., data[i].shape[-1]))).T * data[i].T)[:, i])/np.sum(corI)


    n = coefficients.shape[1]
    v = np.sqrt(corI)
    M = np.repeat(1., n).reshape((n, 1))@(v)
    rep = (M * coefficients.T).T
    mat_cov = (rep.T@rep) / np.sum(Ti)
    cov = (W_m@ mat_cov)@(W_m.T)

    if not np.all(np.abs(cov-cov.T) < 1.e-12):
        ind = np.nonzero(cov - cov.T > 1.e-12)
        for i in ind:
            cov[i] = cov.T[i]

    valeurs_propres, vecteurs_propres = np.linalg.eig(cov.astype(complex128))
    for i in range(len(valeurs_propres)):
        if np.imag(i) > 0:
            valeurs_propres[i] = 0
    bj = np.linalg.solve(W_m, np.eye(W_m.shape[0]))@np.ascontiguousarray(np.real(vecteurs_propres))

    return np.real(valeurs_propres), cov, bj


def _T_hdclassif_dim_choice(ev, n, method, threshold, graph, noise_ctrl, d_set):
    """
    Description
    -----------
    This function determines the optimal amount of dimensions for the model

    Parameters
    ----------
    ev: (K, p)
        A matrix containing the eigenvalues from Principal Component Analysis for each cluster
    n: (p, p)
        A column sum of the t matrix
    method: 'str'
        Either 'bic,' 'cattell,' or 'grid'
    threshold : `float` or `list` of `floats`, default=0.1
        the threshold of the Cattell scree-test used for selecting the group specific intrinsic dimensions
    graph: True or False
    noise_ctrl: 'bool'
    d_set: `int` or `list` of `ints`, default=2
        list of values to be used for the intrinsic dimension of each group when `d_select='grid'`.

    Returns
    -------
    d: (1, K)
        Matrix containing the dimensions per cluster (?)
    """
    N = np.sum(n)
    prop = n/N
    K = len(ev) if ev.ndim > 1 else 1

    if (ev.ndim > 1 and K > 1):
        p = len(ev[0])

        if (method == "cattell"):
            dev = np.abs(np.apply_along_axis(np.diff, 1, ev))
            max_dev = np.apply_along_axis(np.nanmax, 1, dev)
            dev = (dev / np.repeat(max_dev, p-1).reshape(dev.shape)).T
            transpose_compare = (ev[:, 1:] > noise_ctrl).T 
            range_vector = np.arange(1, p).reshape(-1, 1)

            result_matrix =  (dev > threshold) *range_vector * transpose_compare
            d = np.apply_along_axis(lambda col: np.argmax(col) + 1, 0, result_matrix)
            old_d = np.apply_along_axis(np.argmax, 1, (dev > threshold).T*(np.arange(0, p-1))*((ev[:,1:] > noise_ctrl)))
        elif (method == "bic"):

            d = np.repeat(0, K)
            
            for i in range(K):
                Nmax = np.max(np.nonzero(ev[i] > noise_ctrl)[0]) - 1
                B = np.empty((Nmax,1))
                p2 = np.sum(np.invert(np.isnan(ev[i])))
                Bmax = -np.inf

                for kdim in range(0, Nmax):
                    if d[i] != 0 and kdim > d[i] + 10: break

                    a = np.sum(ev[i, 0:(kdim+1)])/(kdim+1)           
                    b = np.sum(ev[i, (kdim + 1):p2])/(p2-(kdim+1))
                 

                    if b < 0 or a < 0:
                        B[kdim] = -np.inf

                    else:
                        L2 = -1/2*((kdim+1)*np.log(a) + (p2 - (kdim + 1))*np.log(b) - 2*np.log(prop[i]) +p2*(1+1/2*np.log(2*np.pi))) * n[i]
                        B[kdim] = 2*L2 - (p2+(kdim+1)*(p2-(kdim+2)/2)+1) * np.log(n[i])
                       
                    if B[kdim] > Bmax:
                        Bmax = B[kdim]
                        d[i] = kdim

            if graph:
                None

        elif method == "grid":
            d = d_set.copy()

    else:
        ev = ev.flatten()
        p = len(ev)

        if method == "cattell":
            dvp = np.abs(np.diff(ev))
            Nmax = np.max(np.nonzero(ev>noise_ctrl)[0]) - 1
            if p ==2:
                d = 0
            else:
                d = np.max(np.nonzero(dvp[0:Nmax] >= threshold*np.max(dvp[0:Nmax]))[0])
            diff_max = np.max(dvp[0:Nmax])

        elif method == "bic":
            d = 0
            Nmax = np.max(np.nonzero(ev > noise_ctrl)[0]) - 1
            B = np.empty((1, Nmax))
            Bmax = -np.inf

            for kdim in range(Nmax):
                if d != 0 and kdim > d+10:
                    break
                a = np.sum(ev[0:kdim])/kdim
                b = np.sum(ev[(kdim+1):p])/(p-kdim)
                if(b <= 0 or a <= 0):
                    B[kdim] = -np.inf
                else:
                    L2 = -1/2*(kdim*np.log(a) + (p-kdim)*np.log(b)+p*(1+1/2*np.log(2*np.pi)))*N
                    B[kdim] = 2*L2 - (p+kdim * (p-(kdim + 1)/2)+1)*np.log(N)

                if B[kdim] > Bmax:
                    Bmax = B[kdim]
                    d = kdim
    if type(d) != np.ndarray:
        d=np.array([d])
    
    return d

def _T_hdclassift_bic(par, p, q, dfconstr='yes'):
    """
    Processes the input model parameter to validate, standardize, and map it to predefined model names for modely. 
    Ensures the input is valid, converts it to the appropriate format, and returns the corresponding modely names.

    Parameters
    ----------
    model : list[str] or str
        The input modely(s) to be processed. Can be a list or array of modely names or numbers, 
        or a single modely name or number.
    all2models : bool, optional
        If True, and the input modely is 'ALL', returns an array of all modely names. 
        Otherwise, returns 'ALL'. By default False.

    Returns
    -------
    new_model : list[str]
        An array of validated and standardized modely names.
    """
    model = par['model']
    modely = par['modely']
    K = par['K']
    d = par['d']
    b = par['b']
    a = par['a']
    N = par['N']
    prop = par['prop']


    if np.isscalar(b) or len(np.atleast_1d(b)) == 1:
        eps = np.sum(prop*d)

        if model in ["ABQD", "AJBQD"]:
            n_max = len(par['ev'])
        else:
            n_max = par['ev'].shape[1] if len(par['ev'].shape) > 1 else len(par['ev'])
        
        b = b * (n_max - eps) / (p - eps)
        b = np.repeat(b, K)
    
    if np.isscalar(a) or (len(np.atleast_1d(a)) == 1 or len(np.atleast_1d(a)) == K):
        a = np.repeat(np.max(d), K)
    if np.nanmin(a) <= 0 or np.any(b) < 0:
        return -np.Inf
    
    if par['loglik'] == None:
        som_a = np.zeros(K)

        for i in range(K):
            som_a[i] = np.sum(np.log(a[i, d[i]]))
        L = (-1/2) * np.sum(prop * (som_a + (p - d)*(np.log(b)) - 2*np.log(prop) + (p + q)*(1 + np.log(2*np.pi)))) * N
    else:
        L = par['loglik']

    ro = K*p+K - 1
    tot = np.sum(d*(p-(d+1)/2))
    D = np.sum(d)
    d = d[0]

    if model == 'AKJBKQKDK':
        m = ro + tot + D + K
    else:
        if model == "AKBKQKDK":
            m = ro + tot + 2*K
        elif model == "ABKQKDK":
            m = ro + tot + K + 1
        elif model == "AKJBQKDK":
            m = ro + tot + D + 1
        elif model == "AKBQKDK":
            m = ro + tot + K + 1
        elif model == "ABQKDK":
            m = ro + tot + 2

    if modely == 'EII':
        m += 1
    else:
        if modely == 'VII':
            m += K
        elif modely == 'EEI':
            m += q
        elif modely == 'VEI':
            m += K + q - 1
        elif modely == 'EVI':
            m += 1 + K * (q - 1)
        elif modely == 'VVI':
            m += K * q
        elif modely == 'EEE':
            m += q * (q + 1) / 2
        elif modely == 'VEE':
            m += K + q - 1 + q * (q - 1) / 2
        elif modely == 'EVE':
            m += 1 + K * (q - 1) + q * (q - 1) / 2
        elif modely == 'EEV':
            m += q + K * q * (q - 1) / 2
        elif modely == 'VVE':
            m += q * K + q * (q - 1) / 2
        elif modely == 'VEV':
            m += K + q - 1 + K * q * (q - 1) / 2
        elif modely == 'EVV':
            m += 1 + K * (q - 1) + K * q * (q - 1) / 2
        elif modely == 'VVV':
            m += K * q * (q + 1) / 2

    bic = - (-2*L + m * np.log(N))
    
    t = par['posterior']
    
    Z = ( (t - np.atleast_2d(np.apply_along_axis(np.max, 1, t)).T) == 0. ) + 0.
    
    icl = bic - 2*np.sum(Z*np.log(t + 1.e-15))
    
    return {'bic': bic, 'icl': icl}

def _T_hdc_getComplexityt(par, p, q, dfconstr='yes'):
    model = par['model']
    modely = par['modely']

    K = par['K']
    d = par['d']



    ro = K*p + K - 1
    tot = np.sum(d*(p-(d+1)/2))
    D = np.sum(d)
    d = d[0]
  
    if model == 'AKJBKQKDK':
        m = ro + tot + D + K
    elif model == "AKBKQKDK":
        m = ro + tot + 2*K
    elif model == "ABKQKDK":
        m = ro + tot + K + 1
    elif model == "AKJBQKDK":
        m = ro + tot + D + 1
    elif model == "AKBQKDK":
        m = ro + tot + K + 1
    elif model == "ABQKDK":
        m = ro + tot + 2

    if (modely == 'EII'):
        m += 1
    elif (modely == 'VII'):
        m += K
    elif (modely == 'EEI'):
        m += q
    elif (modely == 'VEI'):
        m += K + q - 1
    elif (modely == 'EVI'):
        m += 1 + K*(q - 1)
    elif (modely == 'VVI'):
        m += + K*q
    elif (modely == 'EEE'):
        m += q*(q + 1)/2
    elif (modely == 'VEE'):
        m += K + q - 1 + q*(q - 1)/2
    elif (modely == 'EVE'):
        m += 1 + K*(q - 1) + q*(q - 1)/2
    elif (modely == 'EEV'):
        m += q + K*q*(q - 1)/2
    elif (modely == 'VVE'):
        m += q*K+q*(q - 1)/2
    elif (modely == 'VEV'):
        m += K + q - 1 + K*q*(q - 1)/2
    elif (modely == 'EVV'):
        m += 1 + K*(q - 1) + K*q*(q - 1)/2
    elif (modely == 'VVV'):
        m += K*q
    return m

def _T_hdc_getTheModel(model, all2models = False):
    """
    Processes the input model parameter to validate, standardize, and map it to predefined model names.
    Ensures the input is valid, converts it to the appropriate format, and returns the corresponding model names.

    Parameters
    ----------
    model : list[str] or str
        The input model(s) to be processed. Can be a list or array of model names or numbers,
        or a single model name or number.
    all2models : bool, optional
        If True, and the input model is 'ALL', returns an array of all model names.
        Otherwise, returns 'ALL'. By default False.

    Returns
    -------
    new_model : list[str]
        An array of validated and standardized model names.
    """
    model_in = model
    try:
        if type(model) == np.ndarray or type(model) == list:
            new_model = np.array(model,dtype='<U9')
            model = np.array(model)
        else:
            new_model = np.array([model],dtype='<U9')
            model = np.array([model])
    except:
        raise ValueError("Model needs to be an array or list")

    if(model.ndim > 1):
        raise ValueError("The argument 'model' must be 1-dimensional")
    #check for invalid values
    if type(model[0]) != np.str_:
        if np.any(np.apply_along_axis(np.isnan, 0, model)):
            raise ValueError("The argument 'model' cannot contain any Nan")

    #List of model names accepted
    ModelNames = np.array(["AKJBKQKDK", "AKBKQKDK", "ABKQKDK", "AKJBQKDK", "AKBQKDK", "ABQKDK"])
    #numbers between 0 and 5 inclusive are accepted, so check if numbers are sent in as a string before capitalizing
    if type(model[0]) == np.str_:
        if model[0].isnumeric():
            model = model.astype(np.int_)
            
        else:
            new_model = [np.char.upper(m) for m in model]

    #shortcut for all the models
    if len(model) == 1 and new_model[0] == "ALL":
        if all2models:
            new_model = np.zeros(6, dtype='<U9')
            model = np.arange(0,6)
        else:
            return "ALL"
        
    #are the models numbers?
    if type(model[0]) == np.int_:
        qui = np.nonzero(np.isin(model, np.arange(0, 6)))[0]
        if len(qui) > 0:
            new_model[qui] = ModelNames[model[qui]]
            new_model = new_model[qui]

    #find model names that are incorrect    
    qui = np.nonzero( np.invert(np.isin(new_model, ModelNames)))[0]
    if len(qui) > 0:
        if len(qui) == 1:
            msg = f'(e.g. {model_in[qui[0]]} is incorrect.)'

        else:
            msg = f'(e.g. {model_in[qui[0]]} or {model_in[qui[1]]} are incorrect.)'

        raise ValueError("Invalid model name " + msg)
    
    #Warn user that the models *should* be unique
    if np.max(np.unique(model, return_counts=True)[1]) > 1:
        warnings.warn("Values in 'model' argument should be unique.", UserWarning)

    mod_num = []
    for i in range(len(new_model)):
        mod_num.append(np.nonzero(new_model[i] == ModelNames)[0])
    mod_num = np.sort(np.unique(mod_num))
    new_model = ModelNames[mod_num]

    return new_model


def _T_hdc_getTheModely(model, all2models = False):
    """
    Processes the input model parameter to validate, standardize, and map it to predefined model names for modely.
    Ensures the input is valid, converts it to the appropriate format, and returns the corresponding modely names.

    Parameters
    ----------
    model : list[str] or str
        Input modely(s) to process. Can be a list/array of modely names/numbers,
        or a single modely name/number.
    all2models : bool, optional
        If True and input is 'ALL', returns all modely names.
        Otherwise returns 'ALL'. Default is False.

    Returns
    -------
    list[str]
        Array of validated and standardized modely names.
    """
    model_in = model
    #is the model a list or array?
    try:
        if type(model) == np.ndarray or type(model) == list:
            new_model = np.array(model,dtype='<U9')
            model = np.array(model)
        else:
            new_model = np.array([model],dtype='<U9')
            model = np.array([model])
    except:
        raise ValueError("Model needs to be an array or list")

    #one-dimensional please
    if(model.ndim > 1):
        raise ValueError("The argument 'model' must be 1-dimensional")
    #check for invalid values
    if type(model[0]) != np.str_:
        if np.any(np.apply_along_axis(np.isnan, 0, model)):
            raise ValueError("The argument 'model' cannot contain any Nan")

    #List of model names accepted
    ModelNames = np.array(["EII", "VII", "EEI", "VEI", "EVI", "VVI", "EEE", "VEE", "EVE", "EEV", "VVE", "VEV","EVV","VVV"])
    #numbers between 0 and 5 inclusive are accepted, so check if numbers are
    #sent in as a string before capitalizing
    if type(model[0]) == np.str_:
        if model[0].isnumeric():
            model = model.astype(np.int_)
            
        else:
            new_model = [np.char.upper(m) for m in model]

    #shortcut for all the models
    if len(model) == 1 and new_model[0] == "ALL":
        if all2models:
            new_model = np.zeros(6, dtype='<U9')
            model = np.arange(0,6)
        else:
            return "ALL"
        
    #are the models numbers?
    if type(model[0]) == np.int_:
        qui = np.nonzero(np.isin(model, np.arange(0, 6)))[0]
        if len(qui) > 0:
            new_model[qui] = ModelNames[model[qui]]
            new_model = new_model[qui]

    #find model names that are incorrect    
    qui = np.nonzero( np.invert(np.isin(new_model, ModelNames)))[0]
    if len(qui) > 0:
        if len(qui) == 1:
            msg = f'(e.g. {model_in[qui[0]]} is incorrect.)'

        else:
            msg = f'(e.g. {model_in[qui[0]]} or {model_in[qui[1]]} are incorrect.)'

        raise ValueError("Invalid model name " + msg)
    
    #Warn user that the models *should* be unique
    if np.max(np.unique(model, return_counts=True)[1]) > 1:
        warnings.warn("Values in 'model' argument should be unique.", UserWarning)

    mod_num = []
    for i in range(len(new_model)):
        mod_num.append(np.nonzero(new_model[i] == ModelNames)[0])
    mod_num = np.sort(np.unique(mod_num))
    new_model = ModelNames[mod_num]

    return new_model


def _T_addCommas(x):
    """
    Vectorized function that formats numeric values with comma separators.
    Applies comma formatting to each element in an array or to a single value.

    Parameters
    ----------
    x : numeric or array-like
        Input number(s) to be formatted. Can be a single value or array of numbers.

    Returns
    -------
    str or numpy.ndarray
        Formatted string(s) with comma separators and 2 decimal places.
        Returns array if input is array-like, single string otherwise.

    """
    vfunc = np.vectorize(_T_addCommas_single)
    return vfunc(x)

def _T_addCommas_single(x):
    """
    Formats a single numeric value with comma separators and fixed decimal places.

    Parameters
    ----------
    x : numeric
        Input number to be formatted.

    Returns
    -------
    str
        String representation of the number
    """
    return "{:,.2f}".format(x)


def _T_estimateTime(stage, start_time=0, totmod=0):
    curwidth = get_terminal_size()[0]
    outputString = ""

    if stage == 'init':
        medstring = "????"
        longstring = "????"
        shortstring = "0"
        modelCount = None
        unitsRun = ""
        unitsRemain=""

    else:

        modelCount = stage + 1
        modsleft = totmod - modelCount
        timerun = time.process_time() - start_time
        timeremain = (timerun/modelCount)*modsleft
        
        if timeremain > 60 and timeremain <=3600:
            unitsRemain = 'mins'
            timeremain = timeremain/60
        elif timeremain > 3600 and timeremain <= 86400:
            unitsRemain = 'hours'
            timeremain = timeremain/3600
        elif timeremain > 86400 and timeremain <= 604800:
            unitsRemain = 'days'
            timeremain = timeremain/86400
        elif timeremain > 604800:
            unitsRemain = 'weeks'
            timeremain = timeremain/604800
        else:
            unitsRemain = 'secs'

        if timerun > 60 and timerun <=3600:
            unitsRun = 'mins'
            timerun = timerun/60
        elif timerun > 3600 and timerun <= 86400:
            unitsRun = 'hours '
            timerun = timerun/3600
        elif timerun > 86400 and timerun <= 604800:
            unitsRun = 'days'
            timerun = timerun/86400
        elif timerun > 604800:
            unitsRun = 'weeks '
            timerun = timerun/604800
        else:
            unitsRun = 'secs'


        shortstring = round((1-modsleft/totmod)*100)
        medstring = round(timeremain, 1)
        longstring = round(timerun,1)


    if curwidth >=15:
        shortstring = str(shortstring).rjust(5)
        outputString = f'{shortstring}% complete'

        if curwidth >=48:
            medstring = str(medstring).rjust(10)
            outputString = f'Approx. remaining:{medstring} {unitsRemain}  |  {outputString}'

            if curwidth >=74:
                longstring = str(longstring).rjust(10)
                outputString = f'Time taken:{longstring} {unitsRun}  |  {outputString}'

    print(outputString,'\r', flush=True, end='')

def _T_hddc_control(params):

    K = ('K',params['K'])
    checkMissing(K)
    checkType(K, (INT_TYPES))
    checkRange(K, lower=1)

    data = ('data', params['data'])
    checkMissing(data)
    checkType(data, (skfda.FDataBasis, dict))

    if isinstance(data[1], skfda.FDataBasis):
        checkType(('data', data[1].coefficients), (LIST_TYPES, (INT_TYPES, FLOAT_TYPES, LIST_TYPES)))
        naCheck=np.sum(data[1].coefficients)
        if naCheck in UNKNOWNS or pd.isna(naCheck):
            raise ValueError(f"'data' parameter contains unsupported values. Please remove NaNs, NAs, infs, etc. if they are present")
        if np.any(np.array(K[1])>2*data[1].coefficients.shape[1]):
            raise ValueError("The number of observations in the data must be at least twice the number of clusters")
        row_length = data[1].coefficients.shape[0]
    else:
        data_length = 0
        row_length = 0
        for i in range(len(data[1])):
            checkType((f'data', data[1][i].coefficients), (LIST_TYPES, (INT_TYPES, FLOAT_TYPES, LIST_TYPES)))
            naCheck=np.sum(data[1][i].coefficients)
            if naCheck in UNKNOWNS or pd.isna(naCheck):
                raise ValueError(f"'data' parameter contains unsupported values. Please remove NaNs, NAs, infs, etc. if they are present")
            data_length += data[1][i].coefficients.shape[1]
            row_length += data[1][i].coefficients.shape[0]

        if np.any(np.array(K[1])>2*data_length):
            raise ValueError("The number of observations in the data must be at least twice the number of clusters")

    model = ("model", params['model'])
    checkMissing(model)
    checkType(model, (str, INT_TYPES))

    known = ('known', params['known'])
    if not (known[1] is None):
        checkType(known, (LIST_TYPES, (INT_TYPES, FLOAT_TYPES)))

        if isinstance(K[1], LIST_TYPES):
            k_temp = K[1][0]
            if len(K[1]) > 1:
                raise ValueError("K should not use multiple values when using 'known' parameter")
        else:
            k_temp = K[1]

        if np.all(np.isnan(known[1])) or np.all(pd.isna(known[1])):
            raise ValueError("'known' should have values from each class (should not all be unknown)")
        
        if len(known[1]) != row_length:
            raise ValueError("length of 'known' parameter must match number of observations from data")
        knownTemp = np.where(np.any(pd.isna(known[1])) or np.any(np.isnan(known[1])), 0, known[1])
        if len(np.unique(knownTemp)) > k_temp:
            raise ValueError("at most K different classes can be present in the 'known' parameter")
        
        if np.max(knownTemp) > k_temp-1:
            raise ValueError("group numbers in 'known' parameter must come from integers up to K (ie. for K=3, 0,1,2 are acceptable)")

    dfstart = ('dfstart', params['dfstart'])
    checkMissing(dfstart)
    checkType(dfstart, (INT_TYPES, FLOAT_TYPES))
    checkRange(dfstart, lower=2)

    threshold = ('threshold', params['threshold'])
    checkMissing(threshold)
    checkType(threshold, (INT_TYPES, FLOAT_TYPES))
    checkRange(threshold, upper=1, lower=0)
    
    dfupdate = ('dfupdate', params['dfupdate'])
    checkMissing(dfupdate)
    checkType(dfupdate, (str))
    if dfupdate[1] not in ['approx', 'numeric']:
        raise ValueError("'dfupdate' parameter should be either 'approx' or 'numeric'")
    
    dfconstr = ('dfconstr', params['dfconstr'])
    checkMissing(dfconstr)
    checkType(dfconstr, (str))
    if dfconstr[1] not in ['no', 'yes']:
        raise ValueError("'dfconstr' parameter should be either 'no' or 'yes'")
    
    itermax = ('itermax', params['itermax'])
    checkMissing(itermax)
    checkType(itermax, (INT_TYPES))
    checkRange(itermax, lower=2)

    eps = ('eps', params['eps'])
    checkMissing(eps)
    checkType(eps, (INT_TYPES, FLOAT_TYPES))
    checkRange(eps, lower=0)

    init = ('init', params['init'])
    checkMissing(init)
    checkType(init, (str))

    match init[1]:

        case "vector":
            vec = ('init_vector', params['init_vector'])

            checkMissing(vec)
                
            checkType(vec, (LIST_TYPES, (INT_TYPES)))

            if isinstance(K[1], LIST_TYPES):
                k_temp = K[1][0]
                if len(K[1]) > 1:
                    raise ValueError("K should not use multiple values when using init = 'vector'")
            else:
                k_temp = K[1]
            if len(np.unique(vec[1])) < k_temp:
                raise ValueError(f"'init_vector' lacks representation from all K classes (K={K})")

            if len(vec[1]) != row_length:
                raise ValueError("Size 'init_vector' is different from size of data")

        case "mini-em":
            mini = ('mini_nb', params['mini_nb'])

            checkMissing(mini)
            checkType(mini, (LIST_TYPES, (INT_TYPES)))
            checkRange(mini, lower=1)

            if len(mini[1]) != 2:
                raise ValueError(f"Parameter 'mini_nb' should be of length 2, not length {len(mini[1])}")
            
        case "kmeans":
            kmc = ('kmeans_control', params['kmeans_control'])

            if kmc[1] is None:
                pass
            else:
                checkType(kmc, [dict])
                checkKMC(kmc[1])

    criterion = ('criterion', params['criterion'])
    checkMissing(criterion)
    checkType(criterion, (str))

    if criterion[1] not in ['bic', 'icl']:
        raise ValueError("'Criterion' parameter should be either 'bic' or 'icl'")
    
    d_select = ('d_select', params['d_select'])
    checkMissing(d_select)
    checkType(d_select, (str))

    if d_select[1] == 'grid':
        d_range = ('d_range', params['d_range'])
        checkMissing(d_range)
        checkType(d_range, (INT_TYPES))
        checkRange(d_range, lower=1)

        if np.max(d_range) > data_length:
            raise ValueError("Intrinsic dimension 'd' can't be larger than number of input parameters. Please set lower max")

    if d_select[1] not in ['cattell', 'bic']:
        raise ValueError("'d_select' parameter should be 'cattell' 'bic', or 'grid'")
    
    show = ('show', params['show'])
    checkMissing(show)
    checkType(show, [bool])

    min_indiv = ('min_individuals', params['min_individuals'])
    checkMissing(min_indiv)
    checkType(min_indiv, (INT_TYPES))
    checkRange(min_indiv, lower=2)

    cores = ('mc_cores', params['mc_cores'])
    checkMissing(cores)
    checkType(cores, (INT_TYPES))
    checkRange(cores, lower=1)

    rep = ('nb_rep', params['nb_rep'])
    checkMissing(rep)
    checkType(rep, (INT_TYPES))
    checkRange(rep, lower=1)

    keep = ('keepAllRes', params['keepAllRes'])
    checkMissing(keep)
    checkType(keep, [bool])
    
    d_max = ('d_max', params['d_max'])
    checkMissing(d_max)
    checkType(d_max, (INT_TYPES))
    checkRange(d_max, lower=1)

    verbose = ('verbose', params['verbose'])
    checkMissing(verbose)
    checkType(verbose, [bool])


def checkType(param, check):
    if not isinstance(check, type):
        if check[0] is LIST_TYPES:
            result = isinstance(param[1], LIST_TYPES)
            if not np.all(result):
                raise ValueError(f"Parameter {param[0]} is of wrong type (should be of type {LIST_TYPES[0]} for example)")
            
            result = np.array([isinstance(val, check[1]) for val in param[1]])

            if param[0] == "data":
                for i in param[1]:
                    checkType((param[0], i), (INT_TYPES, FLOAT_TYPES))

            if not np.all(result):
                
                result = np.nonzero(result == False)[0]
                raise ValueError(f"Parameter {param[0]} contains data of an incorrect type (cannot contain elements of type {type(param[1][result][0])} for example)")

    else:

        if isinstance(param[1], LIST_TYPES):
            result = np.array([isinstance(val, check) for val in param[1]])

        else:
            result = isinstance(param[1], check)

        if not np.all(result):

            result = np.nonzero(result == False)[0]
            if isinstance(param[1], LIST_TYPES):
                result = param[1][result][0]

            else:
                result = param[1]

            raise ValueError(f"Parameter {param[0]} is of an incorrect type (cannot be of type {type(result)}) for example")
            

def checkMissing(param):

    if param is None:
        raise ValueError(f"Missing required '{param[0]}' parameter")
    
def checkRange(param, upper=None, lower=None):

    result = True
    if isinstance(param[1], LIST_TYPES):
        if not (lower is None):

            result = np.min(param[1]) < lower

        elif not (upper is None):
            
            result = result and (np.max(param[1]) > upper)

    else:

        if not (lower is None):
            result = param[1] < lower
        
        elif not (upper is None):
            result = result and (param[1] > upper)

    if result:
        msg = ""
        if lower != False:
            msg = f' greater than or equal to {lower}'

        elif upper != False:
            if len(msg > 0):
                msg = f'{msg} and'
            msg = f'{msg} less than or equal to {upper}'

        raise ValueError(f"Parameter '{param[0]}' must be {msg}")

def checkKMC(kmc):
    settingNames = ['n_init', 'max_iter', 'algorithm']

    result = [name in kmc for name in settingNames]
    
    if not np.all(result):
        result = np.nonzero(result == False)[0][0]
        raise ValueError(f"Missing setting {result} in parameter 'kmeans_control'")
    
    checkType(('n_init',kmc['n_init']), ((INT_TYPES)))
    checkRange(('n_init',kmc['n_init']), lower=1)
    checkType(('max_iter', kmc['max_iter']), (INT_TYPES))
    checkRange(('max_iter',kmc['max_iter']), lower=1)
    checkType(kmc['algorithm'], (str))

def deafaultKMC(kmc):
    return {}