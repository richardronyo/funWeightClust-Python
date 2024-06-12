# test_mixture.py
import numba
import numpy as np
from py_mixture import determinant, inverse, eigen, svd, Gam1, CovarianceY, C_mstep
import timeit

A = np.array([[1, 1, 1], [3.0, 4.0, 5]], dtype=np.float64)  # Example 2x2 matrix in row-major order

def custom():
    return eigen(A)

def numpy():
    return np.linalg.inv(A)
if __name__ == "__main__":

    A = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)  # Example 2x2 matrix in row-major order
    k = 2  # Number of rows/columns of the matrix
    lda = 2  # Leading dimension of the matrix

    print(A)
    print(f"Determinant: {determinant(A)}")
    
    print(f"Inverse: {inverse(A.copy())}")

    wr, vr = eigen(A.copy())
    print(f"Eigenvalues: {wr}")
    print(f"Eigenvectors: {vr}")

    s, u, vtt = svd(A.copy())
    print("Singular values:")
    print(s)
    
    print("Left singular vectors:")
    print(u)
    
    print("Right singular vectors:")
    print(vtt)

    # Number of repetitions for timeit
    num_repeats = 1000

    # Time custom determinant function
    custom_time = timeit.timeit('custom()', globals=globals(), number=num_repeats)
    print(f"Custom inverse function time: {custom_time / num_repeats} seconds per call")

    # Time numpy determinant function
    numpy_time = timeit.timeit('numpy()', globals=globals(), number=num_repeats)
    print(f"NumPy inverse function time: {numpy_time / num_repeats} seconds per call")

    # Number of observations, predictors, responses, and groups
    NN = np.array([10], dtype=np.int32)
    pp = np.array([3], dtype=np.int32)
    qq = np.array([2], dtype=np.int32)
    GG = np.array([2], dtype=np.int32)

    # Predictor matrix (N x p)
    x = np.array([
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
        [0.7, 0.8, 0.9],
        [1.0, 1.1, 1.2],
        [1.3, 1.4, 1.5],
        [1.6, 1.7, 1.8],
        [1.9, 2.0, 2.1],
        [2.2, 2.3, 2.4],
        [2.5, 2.6, 2.7],
        [2.8, 2.9, 3.0]
    ], dtype=np.float64)

    # Response matrix (N x q)
    y = np.array([
        [0.1, 0.2],
        [0.3, 0.4],
        [0.5, 0.6],
        [0.7, 0.8],
        [0.9, 1.0],
        [1.1, 1.2],
        [1.3, 1.4],
        [1.5, 1.6],
        [1.7, 1.8],
        [1.9, 2.0]
    ], dtype=np.float64)

    # Weights matrix (N x G)
    z = np.array([
        [0.1, 0.2],
        [0.3, 0.4],
        [0.5, 0.6],
        [0.7, 0.8],
        [0.9, 1.0],
        [1.1, 1.2],
        [1.3, 1.4],
        [1.5, 1.6],
        [1.7, 1.8],
        [1.9, 2.0]
    ], dtype=np.float64)

    # Current group index
    gg = np.array([1], dtype=np.int32)

    # Result array (q * p)
    gam = np.zeros(qq[0] * pp[0], dtype=np.float64)

    # Call the function
    Gam1(NN, pp, qq, GG, x, y, z, gg, gam)

    # Output the result
    print("gam:", gam)