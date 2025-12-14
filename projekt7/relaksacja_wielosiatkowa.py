import numpy as np
import matplotlib.pyplot as plt
from numba import jit


N = 128
D_x = 0.2

x_max = N * D_x
y_max = N * D_x
sigma = 0.1 * x_max


def generate_starting_rho() -> np.ndarray:
    global x_max, y_max, sigma, N

    x = np.arange(0, N+1, dtype=float) * D_x
    y = np.arange(0, N+1, dtype=float) * D_x

    Y, X = np.meshgrid(y, x, indexing='ij')


    term1 = np.exp(
        -((X - 0.35 * x_max) ** 2 + (Y - 0.5 * y_max) ** 2) / sigma ** 2
    )

    term2 = np.exp(
        -((X - 0.65 * x_max) ** 2 + (Y - 0.5 * y_max) ** 2) / sigma ** 2
    )

    rho = term1 - term2

    return rho


@jit(nopython=True)
def calculate_w(k: int) -> float:
    # Oblicza sumę wag w jednym wymiarze: Sum_c = c_{-k/2} + ... + c_{k/2}
    # a nastepnie w = (Sum_c)^2
    half_k = k // 2
    sum_c_1d = 0.0

    # Sumowanie po indeksach alfa od -k/2 do k/2
    for alpha_idx in range(-half_k, half_k + 1):
        c_alpha = 1.0
        if abs(alpha_idx) == half_k:
            c_alpha = 0.5

        sum_c_1d += c_alpha

    # Normalizacja w jest kwadratem sumy wag w jednym wymiarze
    w = sum_c_1d * sum_c_1d
    return w


@jit(nopython=True)
def calculate_rho_tilde(rho: np.ndarray, k: int) -> np.ndarray:
    global N
    rho_tilde = np.zeros((N + 1, N + 1))

    c = np.ones(k + 1)
    c[0] = 0.5
    c[-1] = 0.5
    half_k = k//2


    w = calculate_w(k) # k^2
    c_alpha, c_beta = 0.0, 0.0
    for i in range(k, N, k):
        for j in range(k, N, k):
            weighted_sum = 0.0

            for alpha_idx in range(-half_k, half_k + 1):
                c_alpha = 1.0
                if abs(alpha_idx) == half_k:
                    c_alpha = 0.5

                for beta_idx in range(-half_k, half_k + 1):
                    c_beta = 1.0
                    if abs(beta_idx) == half_k:
                        c_beta = 0.5

                    i_prime = i + alpha_idx
                    j_prime = j + beta_idx



                    if 0 <= i_prime <= N and 0 <= j_prime <= N:
                        weighted_sum += c_alpha * c_beta * rho[i_prime, j_prime]



            rho_tilde[i, j] = weighted_sum/w

    return rho_tilde

@jit(nopython=True)
def calculate_S(V: np.ndarray, rho_tilde: np.ndarray, k: int) -> float:
    global D_x

    sum = 0.0

    facot = (k * D_x)**2 / 2.0
    denom_factor = 1.0 / (2.0 * k * D_x)

    for i in range(0, N - k, k):
        for j in range(0, N - k, k):
            dV_dx_avg = (
                (V[i + k, j] - V[i, j]) * denom_factor +
                (V[i + k, j + k] - V[i, j + k]) * denom_factor
            )

            dV_dy_avg = (
                (V[i, j + k] - V[i, j]) * denom_factor +
                (V[i + k, j + k] - V[i + k, j]) * denom_factor
            )

            term1 = dV_dx_avg**2 + dV_dy_avg**2

            term2 = -rho_tilde[i, j] * V[i, j]

            sum += (term1 + term2)

    sum *= facot

    return sum


@jit(nopython=True)
def numba_iter(V: np.ndarray, rho_tilde: np.ndarray, k: int):
    factor = (k * D_x)

    for j in range(k, N, k):
        for i in range(k, N, k):
            Vs = (
                V[j-k, i] +
                V[j+k, i] +
                V[j, i-k] +
                V[j, i+k]
            )

            V[j, i] = 0.25 * (Vs + factor**2*rho_tilde[j, i])

def relax_local(V: np.ndarray, k: int):
    epsilon = 1e-8

    rho_tilde = calculate_rho_tilde(rho, k)
    prev_S = calculate_S(V, rho_tilde, k)

    sums = []

    MAX_ITER = 2_000
    for it in range(MAX_ITER):
        numba_iter(V, rho_tilde, k)

        S = calculate_S(V, rho_tilde, k)
        sums.append(S)

        if it > 1 and np.abs((S - prev_S)/prev_S) < epsilon:
            break

        prev_S = S

    return V, sums

@jit(nopython=True)
def interpolate(V: np.ndarray, k: int) -> np.ndarray:


    for i in range(0, N, k):
        for j in range(0, N, k):
            V[i+k//2, j+k//2] = 0.25 * (V[i, j] + V[i+k, j] + V[i, j+k] + V[i+k, j+k])

    for i in range(0, N + 1, k):
        for j in range(0, N, k):
            V[i, j+k//2] = 0.5 * (V[i, j] + V[i, j+k])

    for i in range(0, N, k):
        for j in range(0, N + 1, k):
            V[i+k//2, j] = 0.5 * (V[i, j] + V[i+k, j])

    return V


fig, axs = plt.subplots(2, 3, figsize=(14, 10))
axs = axs.flatten()

rho = generate_starting_rho()
V = np.zeros((N+1, N+1))

it_offset: int = 1
for idx, k in enumerate([16, 8, 4, 2, 1]):
    V, sums = relax_local(V, k)


    y, x = np.meshgrid(np.arange(0, N, k), np.arange(0, N, k), indexing='ij')
    V_lowres = V[y, x]

    ax = axs[idx]
    ax.set_title(f"k={k}, iter: {len(sums)}")
    im = ax.imshow(V_lowres)
    cbar = fig.colorbar(im, ax=ax)

    ax = axs[5]
    ax.plot(np.arange(len(sums)) + it_offset, sums, label=f"k={k}")
    it_offset += len(sums)

    if k > 1:
        V = interpolate(V, k)

ax = axs[5]
ax.set_title(f"suma w danej iteracji")
ax.legend()
plt.show()