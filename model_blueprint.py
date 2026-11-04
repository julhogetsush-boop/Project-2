import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

# BLOCK 1: Parameters & Coordinate Transformation
A, B, P, A0, LZ, MU = .05, .051, 4, 1e-3, .1, 4*np.pi*1e-7
DELTA = np.log(B / A)
BETA = P * DELTA
KAPPA = BETA / (2 * np.pi)

# Analytical Torque Coefficient
T0 = np.pi * LZ * A0**2 * P**2 / (MU * np.sinh(BETA))

# Nominal field study controls (q_d, q_q)
QFIELD = (.25, .5)

# BLOCK 2: Analytical & Manufactured Solutions
def physical(x, y, q=QFIELD):
    """
    Exact physical solution (Eq. 7). 
    Returns (u, u_x, u_y) as per the implementation's `physical` function.
    """
    s = np.sinh(BETA * y) / np.sinh(BETA)
    p = np.sinh(BETA * (1 - y)) / np.sinh(BETA)
    sp = BETA * np.cosh(BETA * y) / np.sinh(BETA)
    pp = -BETA * np.cosh(BETA * (1 - y)) / np.sinh(BETA)
    c, t = np.cos(2 * np.pi * x), np.sin(2 * np.pi * x)
    
    d, e = q
    # Note: c0 = 1/cosh(BETA) is used in the implementation
    f = p + (1/np.cosh(BETA) + d) * s
    g = e * s
    
    u = f * c + g * t
    ux = 2 * np.pi * (-f * t + g * c)
    uy = (pp + (1/np.cosh(BETA) + d) * sp) * c + e * sp * t
    return u, ux, uy

def manufactured(x, y):
    """
    Manufactured solution and forcing (Eq. 40, 41).
    Returns (u, f).
    """
    c = (1 + y) * np.cos(2 * np.pi * x)
    s = np.sin(2 * np.pi * x) * np.sin(np.pi * y)
    t = .2 * np.cos(6 * np.pi * x) * np.sin(2 * np.pi * y)
    
    u = c + s + t
    f = BETA**2 * c + (BETA**2 + np.pi**2) * s + (9 * BETA**2 + 4 * np.pi**2) * t
    return u, f

# BLOCK 3: Matrix Assembly (Kronecker Product)
def matrix(n):
    """
    Builds the 2D matrix exactly as implemented in fd_experiments.py.
    Note: U[j,i]: radial index first, periodic angular index second.
    """
    i = np.arange(n)
    # Cyclic shift matrix for periodic angular boundary
    shift = sparse.csr_matrix((np.ones(n), (i, (i + 1) % n)), shape=(n, n))
    
    # C_n: 1D periodic Laplacian (x-direction)
    c = 2 * sparse.eye(n) - shift - shift.T
    
    # T_{n-1}: 1D Dirichlet Laplacian (y-direction)
    t = sparse.diags([-1, 2, -1], [-1, 0, 1], shape=(n - 1, n - 1), dtype=float)
    
    # Assemble Kh (Notice: n*n is equivalent to 1/h**2)
    Kh = n * n * (sparse.kron(t, sparse.eye(n)) + 
                  KAPPA**2 * sparse.kron(sparse.eye(n - 1), c))
    return Kh.tocsr()

# BLOCK 4: RHS and Boundary Elimination
def data(n, case, q=QFIELD):
    """
    Generates grid, exact solution, and RHS vector exactly as implemented.
    """
    # x has n nodes (periodic), y has n+1 nodes (boundaries included)
    x, y = np.arange(n)[None, :] / n, np.arange(n + 1)[:, None] / n
    
    if case == 'physical':
        u, _, _ = physical(x, y, q)
        f = np.zeros_like(u)
    else:
        u, f = manufactured(x, y)
        
    # Extract interior nodes for RHS
    b = f[1:-1].copy()
    
    # Move Dirichlet boundary values to RHS (multiply by 1/h^2 = n^2)
    b[0] += n * n * u[0]
    b[-1] += n * n * u[-1]
    
    return x, y, u, b.ravel()

# BLOCK 5: Field Derivatives and Discrete Torque
def field(k_u, exact, n):
    """
    Calculates physical flux densities and torque, exactly as implemented.
    """
    u = exact.copy()
    u[1:-1] = k_u.reshape(n - 1, n)
    
    # Central differences
    ux = n / 2 * (np.roll(u, -1, axis=1) - np.roll(u, 1, axis=1))[1:-1]
    uy = n / 2 * (u[2:] - u[:-2])
    
    # Torque calculation (Eq. 23)
    torque = -LZ * P * A0**2 / (MU * DELTA) * np.mean(ux * uy, axis=1)
    return ux, uy, torque

# BLOCK 6: Execution & Verification
if __name__ == "__main__":
    n = 64
    
    # 1. Test Matrix Assembly
    Kh = matrix(n)
    
    # 2. Test Physical Problem
    x, y, exact_phys, b_phys = data(n, 'physical')
    u_phys = spsolve(Kh, b_phys)
    ux, uy, T_phys = field(u_phys, exact_phys, n)
    
    # Check torque at mid-radius (as done in implementation)
    mid_idx = n//2 - 1
    print(f"Physical Torque (n={n}): {T_phys[mid_idx]:.6f} Nm")
    print(f"Expected Analytical Torque: {T0 * QFIELD[1]:.6f} Nm")
    
    # 3. Test Manufactured Problem (MMS)
    x, y, exact_mms, b_mms = data(n, 'manufactured')
    u_mms = spsolve(Kh, b_mms)
    
    # Calculate discrete L2 error exactly as implemented
    e = u_mms - exact_mms[1:-1].ravel()
    err_L2 = np.linalg.norm(e) / n
    print(f"MMS L2 Error (n={n}): {err_L2:.4e}")