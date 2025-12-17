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




@jit(nopython=True)
def relax(V, k, rho, dx, w):
    factor = ((k*dx)**2)
    for i in range(k, N,k):
        for j in range(k, N, k):
            V[i,j] = 0.25*(V[i+k, j] + V[i-k, j] + V[i, j+k] + V[i, j+k] + factor*rho[i,j])

    return V


@jit(nopython=True)
def S_integral(V, N,k, dx, rho):

    List_of_sums = []
    Sum = 0
    factor = ((k*dx)**2)
    denom = 2*k*dx


    for it in range(2000):
        for i in range(0, N-k+1):
            for j in range(0, N-k+1):
                term1 = ((V[i+k, i,j] - V[i,j])/denom + (V[i+k, j+k] - V[i, j+k])/denom)**2
                term2 = ((V[i,j+k] - V[i,j+k])/denom + (V[i+k, j+k] - V[i+k, j])/denom)**2
                Sum += (factor/2)*(term1 + term2) - factor*(rho[i,j]*V[i,j])



        sum_prev = List_of_sums[:-1]
        List_of_sums.append(Sum)
        if it > 0 and np.abs((Sum - sum_prev)/sum_prev) < 1e-8:
            return List_of_sums, it




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
    for i in range(0, N+1):
        V[i,0] = 0
        V[i, N] = 0

    for j in range(0, N+1):
        V[0,j] = 0
        V[N,j] = 0

    return V

K = [16, 8,4,2,1]


k_sums = {}
k_it = {}

it_total=0
it = 0

fig, axs = plt.subplots(2, 3, figsize=(14, 10))
sum_ax = ax[5]

for k in K:
    w = calc_w(k)
    rho = avg_rho(k, rho, N, w)
    V_grid = relax(V_grid,k,rho,w)

    k_sums[str(k)], it = S_integral(V_grid, N, k, rho)

    it_total += it
    k_it[str(k)] = it_total

    V_grid = interpolate(V_grid, k)