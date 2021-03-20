import numpy as np
from numpy import linalg
import random

def cramer(M, b):
    """
    using cramer's rule to compute x where Mx = b
    M is a square matrix and b is a vector(numpy array), both in the same vector space
    Returns: x
    """
    res = []
    for i in range(M.shape[0]):
        tmp = [M[x][y] if y!=i else b[x] for x in range(M.shape[0]) for y in range(M.shape[0])]
        # print(np.array(tmp).reshape(M.shape))
        res.append(np.linalg.det(np.array(tmp).reshape(M.shape[0], M.shape[0])))
    res /= np.linalg.det(M)
    return res

if __name__ == "__main__":
    n = 4
    M = np.array([int(random.random() * 20 - 10) for x in range(n**2)]).reshape(n,n)
    b = np.array([int(random.random() * 20 - 10) for x in range(n)])
    
    # print("M: \n", M)
    # print("b: \n", b)
    
    x = cramer(M, b)
    print("x using cramer's rule: \n", x)
    print("x using reversed matrix: \n", np.matmul(np.linalg.inv(M), b.T))
    


