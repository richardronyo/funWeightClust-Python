import pandas as pd
import numpy as np

from process_data import create_functional_data

def fitCorpus(basis = 30):
    """
    This function creates functional dataset for the values that are found in data/ADNI folder
    """

    if basis == 30:
        x = pd.read_csv("data/Corpus/corpus_f30.csv").to_numpy()
        basis_type = "FOURIER"
    elif basis == 40:
        x = pd.read_csv("data/Corpus/corpus_bs40.csv").to_numpy()
        basis_type = "BSPLINE"
    elif basis == 20:
        x = pd.read_csv("data/Corpus/corpus_f20.csv").to_numpy()
        basis_type = "FOURIER"

    y = pd.read_csv("data/Corpus/corpus_y.csv").to_numpy()
    
    clm = pd.read_csv("data/Corpus/corpus_labels.csv")
    clm = clm - 1
    clm = clm.to_numpy().flatten()

    fdx = create_functional_data(values=x, basis_type=basis_type, n_basis=x.shape[1])
    fdy = create_functional_data(values=y, basis_type=basis_type, n_basis=y.shape[1] + 1)

    return{
        "fdx": fdx,
        "fdy": fdy,
        "groupd": clm
    }

def fitCingulum(basis=30):
    if basis == 30:
        x = pd.read_csv("data/Cingulum/x.csv")
        basis_type = "FOURIER"
    elif basis == 40:
        x = pd.read_csv("data/Cingulum/cingulum_x_bs40.csv")
        basis_type = "BSPLINE"
    elif basis == 20:
        x = pd.read_csv("data/Cingulum/cingulum_xf20.csv")
        basis_type = "FOURIER"

    y = pd.read_csv("data/Cingulum/y.csv")

    clm = pd.read_csv("data/Cingulum/cingulum_labels.csv")
    clm = clm - 1
    clm = clm.to_numpy().flatten()

    nsplines = x.shape[1]

    fdx = create_functional_data(values=x, basis_type=basis_type, n_basis=nsplines)
    fdy = create_functional_data(values=y, basis_type=basis_type, n_basis=y.shape[1] + 1)

    return{
        'fdx': fdx,
        'fdy': fdy,
        'groupd': clm
    }

