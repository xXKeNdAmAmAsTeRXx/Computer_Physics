import numpy as np
import matplotlib.pyplot as plt
from numba import jit

N = 128
D_x = 0.2

x_max = N * D_x
y_max = N * D_x
sigma = 0.1 * x_max

x = np.arange(0, N + 1, dtype=float) * D_x
y = np.arange(0, N + 1, dtype=float) * D_x
V_grid = np.zeros((N+1, N+1))

rho = np.zeros((N+1, N+1))

for i in range(N+1):
    for j in range(N+1):
        term1 = np.exp(
            -((x[i] - 0.35 * x_max) ** 2 + (y[j] - 0.5 * y_max) ** 2) / sigma ** 2
        )

        term2 = np.exp(
            -((x[i] - 0.65 * x_max) ** 2 + (y[j] - 0.5 * y_max) ** 2) / sigma ** 2
        )

        rho[i][j] = term1 - term2


@jit(nopython=True)
def calc_w(k):
    #Calac W
    w = 0
    for i in range(-k//2, k//2+1):
        c_a = 1 if np.abs(i) == k//2 else 0
        for j in range(-k//2, k//2+1):
            c_b = 1 if np.abs(j) == k // 2 else 0

            w+=c_a * c_b
    return w


def avg_rho(k,rho, N, w):
    new_rho = np.zeros((N + 1, N + 1), dtype=float)
    for i in range(k//2, N - k//2):
        for j in range(k//2, N - k//2):

            for a in range(-k // 2, k // 2 + 1):
                c_a = 1 if np.abs(a) == k // 2 else 0
                for b in range(-k // 2, k // 2 + 1):
                    c_b = 1 if np.abs(b) == k // 2 else 0

                    new_rho[i, j] += c_a*c_b*rho[i+a, j+b]

            new_rho[i,j] = (1/w)*new_rho[i,j]

    return new_rho


w = calc_w(16)
rho = avg_rho(16, rho,N,w)

plt.imshow(rho, cmap='gray')
plt.show()

@jit(nopython=True)
def relax(V, k, rho, dx, w):
    factor = ((k*dx)**2)
    for i in range(k, N,k):
        for j in range(k, N, k):
            V[i,j] = 0.25*(V[i+k, j] + V[i-k, j] + V[i, j+k] + V[i, j+k] + factor*rho[i,j])

    return V

@jit(nopython=True)
def interpolate(V, k):
    half_k = k//2
    for i in range(0, N+1):
        for j in range(0, N+1):
            V[i + half_k, j+half_k] = 0.25*(V[i,j] + V[i+k, j] + V[i, j+k] + V[i+k, j+k])
            V[i+k, j+half_k] = 0.5*(V[i+k, j] + V[i+k, i+k])
            V[i+half_k, j+k] = 0.5*(V[i,j+k] + V[i+k, j+k])
            V[i+half_k, j] = 0.5*(V[i,j] + V[i+k, j])
            V[i,j+half_k] = 0.5*(V[i,j] + V[i, j+k])

    ##Warunki brzegowe
    
