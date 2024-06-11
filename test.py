# test_mixture.py
import numba
import numpy as np
from py_mixture import determinant, inverse, eigen, svd
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