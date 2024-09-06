import numpy as np
import pandas as pd
from skfda.representation.basis import BSplineBasis, FourierBasis
from skfda.representation.grid import FDataGrid
from matplotlib import pyplot as plt
import funweightclust as funweight
from sklearn import metrics as met


def fitCorpusFD(basis = 30):
    if basis == 30:
        x = pd.read_csv("data/corpus_f30.csv")
    elif basis == 40:
        x = pd.read_csv("data/corpus_bs40.csv")
    elif basis == 20:
        x = pd.read_csv("data/corpus_f20.csv")

    y = pd.read_csv("data/corpus_y.csv")
    clm = pd.read_csv("data/corpus_labels.csv")

    ncurves = x.shape[0]
    nsplines = x.shape[1]

    if basis == 30 or basis == 20:
        fbasis_x = FourierBasis(n_basis=nsplines, domain_range=(0, 5587))
        fbasis_y = FourierBasis(n_basis=y.shape[1], domain_range=(0, 5587))
    elif basis == 40:
        bbasis_x = BSplineBasis(n_basis=nsplines, domain_range=(0, 5587))
        bbasis_y = BSplineBasis(n_basis=y.shape[1], domain_range=(0, 5587))

    argvals_x = np.linspace(0, 5587)
    argvals_y = np.linspace(0, 5587)

    if basis == 30 or basis == 20:
        evalx = fbasis_x(argvals_x)[:, :, 0]
        evaly = fbasis_y(argvals_y)[:, :, 0]
    elif basis == 40:
        evalx = bbasis_x(argvals_x)[:, :, 0]
        evaly = bbasis_y(argvals_y)[:, :, 0]

    final_x = x @ evalx
    final_y = y @ evaly

    fdx = FDataGrid(final_x, argvals_x)
    fdy = FDataGrid(final_y, argvals_y)

    if basis == 30 or basis == 20:
        fd_basis_x = fbasis_x
        fd_basis_y = fbasis_y
    elif basis == 40:
        fd_basis_x = bbasis_x
        fd_basis_y = bbasis_y

    return {
        'fdx': fdx.to_basis(fd_basis_x),
        'fdy': fdy.to_basis(fd_basis_y),
        'groupd': (clm - 1).to_numpy().flatten()
    }

def fitAlzheimerFD(basis = 30):
    if basis == 30:
        x = pd.read_csv("data/x.csv")
    elif basis == 40:
        x = pd.read_csv("data/cingulum_x_bs40.csv")
    elif basis == 20:
        x = pd.read_csv("data/cingulum_xf20.csv")

    y = pd.read_csv("data/y.csv")
    clm = pd.read_csv("data/cingulum_labels.csv")
    

    ncurves = x.shape[0]
    nsplines = x.shape[1]

    if basis == 30 or basis == 20:
        fbasis_x = FourierBasis(n_basis=nsplines, domain_range=(0, 2231))
        fbasis_y = FourierBasis(n_basis=y.shape[1], domain_range=(0, 2231))
    elif basis == 40:
        bbasis_x = BSplineBasis(n_basis=nsplines, domain_range=(0, 2231))
        bbasis_y = BSplineBasis(n_basis=y.shape[1], domain_range=(0, 2231))

    argvals_x = np.linspace(0, 2231)
    argvals_y = np.linspace(0, 2231)

    if basis == 30 or basis == 20:
        evalx = fbasis_x(argvals_x)[:, :, 0]
        evaly = fbasis_y(argvals_y)[:, :, 0]
    elif basis == 40:
        evalx = bbasis_x(argvals_x)[:, :, 0]
        evaly = bbasis_y(argvals_y)[:, :, 0]

    final_x = x.to_numpy().astype(np.float64) @ evalx
    final_y = y.to_numpy().astype(np.float64) @ evaly

    fdx = FDataGrid(final_x, argvals_x)
    fdy = FDataGrid(final_y, argvals_y)

    if basis == 30 or basis == 20:
        fd_basis_x = fbasis_x
        fd_basis_y = fbasis_y
    elif basis == 40:
        fd_basis_x = bbasis_x
        fd_basis_y = bbasis_y

    return {
        'fdx': fdx.to_basis(fd_basis_x),
        'fdy': fdy.to_basis(fd_basis_y),
        'groupd': (clm - 1).to_numpy().flatten()
    }

def split_fda_chunks(fd1, fd2, num_chunks, labels):
    n_samples = fd1.n_samples
    indices = np.arange(n_samples)
    np.random.shuffle(indices)  # Shuffle indices to make the splits random

    chunk_size = n_samples // num_chunks
    chunks_fd1 = []
    chunks_fd2 = []
    label_chunks = []

    for i in range(num_chunks):
        start_index = i * chunk_size
        if i < num_chunks - 1:
            end_index = start_index + chunk_size
        else:
            end_index = n_samples  # Handle the last chunk to include any remaining samples

        chunk_indices = indices[start_index:end_index]
        chunks_fd1.append(fd1[chunk_indices])
        chunks_fd2.append(fd2[chunk_indices])
        label_chunks.append(labels[chunk_indices])

    return chunks_fd1, chunks_fd2, label_chunks

def select_and_combine_chunks(fd_chunks, selected_chunk_index):
    # Select the specified chunk
    selected_chunk = fd_chunks[selected_chunk_index]
    
    # Combine the remaining chunks
    remaining_chunks = [chunk for i, chunk in enumerate(fd_chunks) if i != selected_chunk_index]
    combined_remaining_chunks = remaining_chunks[0]
    for chunk in remaining_chunks[1:]:
        combined_remaining_chunks = combined_remaining_chunks.concatenate(chunk)
    
    return selected_chunk, combined_remaining_chunks

def test_predict(number_of_chunks, threshold, region="cingulum"):
    if region == "corpus":
        data = fitCorpusFD()
    else:
        data = fitAlzheimerFD()
    
    fdobj = data['fdx']
    fdobjy = data['fdy']
    labels = data['groupd']

    split_data, split_datay, split_labels = split_fda_chunks(fdobj, fdobjy, number_of_chunks, labels)

    confusion_matrices = []
    ari_scores = []

    for i in range(len(split_data)):
        test_labels = np.array(split_labels[i])

        test_data, training_data = select_and_combine_chunks(split_data, i)
        test_datay, training_datay = select_and_combine_chunks(split_datay, i)

        training_labels = np.concatenate([split_labels[j] for j in range(len(split_labels)) if j != i])
    
        print("Training ", i+1, ":")
        res = funweight.funweightclust(training_data, training_datay, K=2, model="all", modely="all", init="kmeans", nb_rep=1, threshold=threshold, known=training_labels, verbose=False)
        p = res.predict(test_data, test_datay)
        if (p['class'] is not None):
            confusion_matrices.append(met.confusion_matrix(test_labels, p['class']))
            ari_scores.append(met.adjusted_rand_score(test_labels, p['class'])) 

    return ari_scores, confusion_matrices


#Split the clusters, get 10% from each, combine them, use the new combined block format for testing and training.

    
if __name__ == "__main__":
    data = fitAlzheimerFD()
    fdobj = data['fdx']
    fdobjy = data['fdy']
    clm = data['groupd']

    model = "AkjBkQkDk"
    modely = "EII"
    model=model.upper()
    res = funweight.funweightclust(fdobj, fdobjy, K=2, model = model, modely = modely, known = None, init="kmeans", nb_rep=1, verbose=False, threshold=0.01)

