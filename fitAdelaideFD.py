import numpy as np
import pandas as pd
from process_data import create_functional_data, constant_functional_data
from testing_functions import test_predict


def fitAdelaide():
    """
    This function creates functional data from the Adelaide dataset. It will choose the measurements from sunday and monday as predictors, and the day of the week as response variables.
    """

    raw_adelaide = pd.read_csv("data/Adelaide/adelaide.csv")
    X = raw_adelaide["X"]
    Y = raw_adelaide.filter(like="Y")
    
    Y = Y.filter(regex="Sun|Tue")
    predictor_fourier = create_functional_data(Y.values.T, "FOURIER")
    predictor_bspline = create_functional_data(Y.values.T, "BSPLINE")
    
    response_raw = np.repeat([0, 1], [508, 508])
    response = constant_functional_data(response_raw)

    return {
        "fdx": predictor_bspline,
        "fdy": response,
        "groupd": response_raw
    }

if __name__ == "__main__":
    ans = fitAdelaide()
    num_chunks = 4
    accuracy_rates, ari_scores, confusion_matrices = test_predict(ans, num_chunks, 0.001)


    for i in range(num_chunks):
        print("Accuracy Rate for Chunk {}: {}".format(i + 1, accuracy_rates[i]))
        print("ARI Score for Chunk {}: {}".format(i + 1, ari_scores[i]))
        print("Confusion Matrix for Chunk {}: \n{}".format(i + 1, confusion_matrices[i]))
        print("\n")

