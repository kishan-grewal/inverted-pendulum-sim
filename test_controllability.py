from src.dynamics import linearise
import numpy as np

A, B = linearise()
print('A matrix:')
print(A)
print()
print('B matrix:')
print(B)
print()

# Check controllability
from numpy.linalg import matrix_rank
n = A.shape[0]
C = np.hstack([B, A @ B, A @ A @ B, A @ A @ A @ B])
print(f'Controllability matrix rank: {matrix_rank(C)} (need {n} for full controllability)')