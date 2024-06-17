import numpy as np
from py_mixture import C_rmahalanobis

if __name__ == "__main__":
    NN = 3
    pp = 2
    qq = 2
    GG = 1
    gg = 0

    x = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float64)
    y = np.array([[2.0, 1.0], [4.0, 3.0], [6.0, 5.0]], dtype=np.float64)
    gam = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    cov = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    delta = np.zeros(NN, dtype=np.float64)

    result = C_rmahalanobis(NN, pp, qq, GG, gg, x, y, gam, cov, delta)
    ans = np.zeros((NN, NN))
    ans[:, 0] = result
    print(ans)