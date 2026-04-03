import scipy.io
import scipy.sparse.linalg
import numpy as np

# Read the sparse matrix in Matrix Market format
A = scipy.io.mmread("data/Pres_Poisson/Pres_Poisson.mtx").tocsr()

# Read the right-hand side vector in Matrix Market format
b = scipy.io.mmread("data/Pres_Poisson/Pres_Poisson_b.mtx")

# Ensure b is a dense vector
if scipy.sparse.issparse(b):
    b = b.toarray().flatten()

# Solve the linear system A * x = b
x = scipy.sparse.linalg.spsolve(A, b)

# Print the solution vector (first 10 elements for brevity)
print("Solution vector (first 10 elements):")
print(x[:10])

# Optionally, save the full solution to a file
np.savetxt("data/Pres_Poisson/solution.txt", x)
