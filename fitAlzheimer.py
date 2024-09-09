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
    """
    Description
    -----------
    This function splits functional data into chunks that will be used when testing the model.

    Parameters
    ----------
    fd1: `FDataBasis` or `list` of `FDataBasis`
        a `FDataBasis` object that contains functional data fit to a set of
        basis functions, or a list of these

    fd2: `FDataBasis` or `list` of `FDataBasis`
        a `FDataBasis` object that contains functional data fit to a set of
        basis functions, or a list of these

    num_chunks: 'int'
        The number of chunks you want the data split into. For example splitting the data into 4 smaller chunks of data.

    labels: list of 'int'
        The labels for each entry that notify which cluster it belongs to.
    """
    # Get indices for each cluster
    label_cluster1_indices = np.random.permutation(np.where(labels == 0)[0])
    label_cluster2_indices = np.random.permutation(np.where(labels == 1)[0])
    
    # Split indices into chunks
    label_cluster1_chunks = np.array_split(label_cluster1_indices, num_chunks)
    label_cluster2_chunks = np.array_split(label_cluster2_indices, num_chunks)
    
    # Combine chunks from both clusters
    label_chunks = [labels[np.concatenate((label_cluster1_chunks[i], label_cluster2_chunks[i]))] for i in range(num_chunks)]
    
    # Create empty lists for functional data chunks
    chunks_fd1 = []
    chunks_fd2 = []
    
    # Create chunks for fd1 and fd2 based on the label_chunks
    for chunk in label_chunks:
        chunks_fd1.append(fd1[chunk])
        chunks_fd2.append(fd2[chunk])
    
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

    for i in range(number_of_chunks):
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

if __name__ == "__main__":
    ari_scores, confusion_matrices = test_predict(2, 0.001)

    for i in range(10):
        print(f"Testing Set {i}:\nARI Score: {ari_scores[i]}\nConfusion Matrix:\n{confusion_matrices[i]}")