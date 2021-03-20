import numpy as np
from numpy import linalg
import random

def adjugate(M):
    """
    M is a numpy matrix with a square shape greater or equal than 2x2
    Returns: the adjugate matrix of M
    """
    res = []
    tmp = []
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            tmp = [M[x][y] for x in range(M.shape[0]) for y in range(M.shape[1]) if x!=i and y!=j]
            res.append((-1)**(i+j) * np.linalg.det(np.array(tmp).reshape(M.shape[0]-1, M.shape[1]-1)))
    adj = np.array(res).reshape(M.shape[0], M.shape[1])
    return adj

if __name__ == "__main__":
    n = 4
    M = np.array([int(random.random() * 20 - 10) for x in range(n**2)]).reshape(n,n)
    adj = adjugate(M)
    
    print("adj = \n", adj)
    print("det M = \n", np.linalg.det(M))
    print("calculated M**-1 = \n", adj / np.linalg.det(M))
    print("M**-1 = \n", np.linalg.inv(M))
    