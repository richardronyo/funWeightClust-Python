import pandas as pd
import numpy as np

from process_data import create_functional_data, constant_functional_data
import funweightclust as fwc
import sklearn.metrics as met


def fitADNICingulum():
    raw_cingulum = pd.read_csv("data/ADNI/ADNI_Cingulum_ADCN.csv")
    
    cingulum_voxelwise_data_column_names = [column for column in raw_cingulum.columns if "Var" in column]
    cingulum_voxelwise_data = raw_cingulum[cingulum_voxelwise_data_column_names]

    clm = raw_cingulum["Research.Group"].replace({'AD':0, 'CN':1}).astype(int).to_numpy()

    raw_cingulum_y_data = raw_cingulum["MMSE.Total.Score"].to_numpy()

    fdx = create_functional_data(cingulum_voxelwise_data)
    fdy = constant_functional_data(raw_cingulum_y_data)

    return {
        'fdx': fdx,
        'fdy': fdy,
        'groupd': clm
    }

def fitADNICorpus():
    raw_corpus = pd.read_csv("data/ADNI/ADNI_Corpus_ADCN.csv")

    corpus_voxelwise_data_column_names = [column for column in raw_corpus.columns if "Var" in column]
    corpus_voxelwise_data = raw_corpus[corpus_voxelwise_data_column_names]

    clm = raw_corpus["Research.Group"].replace({'AD':0, 'CN':1}).astype(int).to_numpy()

    raw_corpus_y_data = raw_corpus["MMSE.Total.Score"].to_numpy()

    fdx = create_functional_data(corpus_voxelwise_data)
    fdy = constant_functional_data(raw_corpus_y_data)

    return {
        'fdx': fdx,
        'fdy': fdy,
        'groupd': clm
    }

if __name__ == "__main__":
    corpus_data = fitADNICorpus()
    corpus_fdx = corpus_data['fdx']
    corpus_fdy = corpus_data['fdy']
    corpus_labels = corpus_data['groupd']

    res = fwc.funweightclust(corpus_fdx, corpus_fdy, K=2, model="all", modely="all", init="kmeans", nb_rep=1, threshold=0.001)
    print("ARI Score:\t", met.adjusted_rand_score(res.cl, corpus_labels))
    print("Confusion Matrix:\n", met.confusion_matrix(res.cl, corpus_labels))