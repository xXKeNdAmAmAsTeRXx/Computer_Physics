import numpy as np
import matplotlib.pyplot as plt
from numba import njit
import matplotlib.colors as colors

# --- Ustawienia początkowe i Definicje (Takie jak w poprzedniej odpowiedzi) ---

# Stałe projektu
N = 128
Delta_x = 0.2
Epsilon = 1e-8
# Wymiary siatki
X_max = N * Delta_x
Y_max = N * Delta_x
Sigma = 0.1 * X_max


# Definicja gęstości ładunku rho(x, y)
@njit
def rho_func(x, y):
    term1 = np.exp(-((x - 0.35 * X_max) ** 2 + (y - 0.5 * Y_max) ** 2) / (Sigma ** 2))
    term2 = np.exp(-((x - 0.65 * X_max) ** 2 + (y - 0.5 * Y_max) ** 2) / (Sigma ** 2))
    return term1 - term2


# Funkcja wag c_alpha
@njit
def c_alpha_func(alpha, k):
    if alpha == k / 2 or alpha == -k / 2:
        return 0.5
    elif -k / 2 < alpha < k / 2:
        return 1.0
    return 0.0


# Obliczanie uśrednionej gęstości ładunku tilde_rho
@njit
def calculate_tilde_rho(rho_grid, i, j, k):
    w = 0.0
    sum_rho = 0.0

    # Sumowanie po komórce
    for alpha_idx in range(-k, k + 1):
        for beta_idx in range(-k, k + 1):
            alpha = alpha_idx / 2.0
            beta = beta_idx / 2.0

            c_alpha = c_alpha_func(alpha, k)
            c_beta = c_alpha_func(beta, k)

            # Indeksy węzłów do uśredniania
            idx_i = i + alpha_idx
            idx_j = j + beta_idx

            if 0 <= idx_i <= N and 0 <= idx_j <= N:
                weight = c_alpha * c_beta
                w += weight
                sum_rho += weight * rho_grid[idx_i, idx_j]

    if w == 0:
        return 0.0
    return sum_rho / w


# Funkcja relaksacji (Gauss-Seidel)
@njit
def relax(V, Rho, k):
    V_new = V.copy()


    for i in range(k, N - k + 1):
        for j in range(k, N - k + 1):

            # Obliczenie uśrednionej gęstości w punkcie (i, j)
            tilde_rho = calculate_tilde_rho(Rho, i, j, k)

            # Wzór na relaksację
            term = V[i + k, j] + V[i - k, j] + V[i, j + k] + V[i, j - k]

            V_new[i, j] = 0.25 * (term + (k * Delta_x) ** 2 * tilde_rho)
    V[:] = V_new[:]
    return V


# Obliczanie całki funkcjonalnej S_k
@njit
def calculate_S_k(V, Rho, k):
    S_k = 0.0

    for i in range(0, N - k + 1):
        for j in range(0, N - k + 1):
            # Obliczenie uśrednionej gęstości w punkcie (i, j)
            tilde_rho = calculate_tilde_rho(Rho, i, j, k)

            # dV/dx (uśrednione)
            dv_dx_term = (V[i + k, j] - V[i, j]) / (2 * k * Delta_x) + \
                         (V[i + k, j + k] - V[i, j + k]) / (2 * k * Delta_x)

            # dV/dy (uśrednione)
            dv_dy_term = (V[i, j + k] - V[i, j]) / (2 * k * Delta_x) + \
                         (V[i + k, j + k] - V[i + k, j]) / (2 * k * Delta_x)

            # Wkład do całki funkcjonalnej
            integrand = 0.5 * (dv_dx_term ** 2 + dv_dy_term ** 2 - tilde_rho * V[i, j])

            S_k += (k * Delta_x) ** 2 * integrand

    return S_k


# Zagęszczanie siatki z kroku k do kroku k/2
@njit
def interpolate(V, k):
    V_new = V.copy()
    k_half = k // 2

    if k_half < 1:
        return V_new

    # Iteracja po węzłach "starej" siatki
    for i in range(0, N + 1 - k):
        for j in range(0, N + 1 - k):
            if i + k <= N and j + k <= N:
                i_center = i + k_half
                j_center = j + k_half

                # Interpolacja w środku kwadratu
                V_new[i_center, j_center] = 0.25 * (V[i, j] + V[i + k, j] + V[i, j + k] + V[i + k, j + k])

                # Interpolacja na środkach krawędzi (pionowe i poziome)
                V_new[i, j_center] = 0.5 * (V[i, j] + V[i, j + k])
                V_new[i + k, j_center] = 0.5 * (V[i + k, j] + V[i + k, j + k])
                V_new[i_center, j] = 0.5 * (V[i, j] + V[i + k, j])
                V_new[i_center, j + k] = 0.5 * (V[i, j + k] + V[i + k, j + k])

    return V_new


# --- Główna Pętla Obliczeniowa i Wizualizacja ---
V_grid = np.zeros((N + 1, N + 1), dtype=np.float64)
Rho_grid = np.zeros((N + 1, N + 1), dtype=np.float64)

# Wypełnianie siatki gęstością ładunku
for i in range(N + 1):
    for j in range(N + 1):
        x = Delta_x * i
        y = Delta_x * j
        Rho_grid[i, j] = rho_func(x, y)


k_values = [16, 8, 4, 2, 1]
S_history = {}
V_current = V_grid.copy()
V_maps = []  # Lista do przechowywania końcowych map potencjału

# Pętla po kolejnych krokach siatki k
for k in k_values:
    print(f"Rozpoczynanie dla k={k}")

    # Zagęszczanie siatki przed zmniejszeniem kroku
    V_current = interpolate(V_current, k * 2)

    S_history[k] = []

    S_old = calculate_S_k(V_current, Rho_grid, k)
    S_history[k].append(S_old)

    it = 0
    # Główna pętla relaksacji
    while True:
        it += 1

        # Wykonanie relaksacji
        V_current = relax(V_current, Rho_grid, k)

        # Obliczenie nowej wartości całki funkcjonalnej
        S_new = calculate_S_k(V_current, Rho_grid, k)
        S_history[k].append(S_new)

        # Warunek zatrzymania iteracji
        if it > 1 and abs((S_new - S_old) / S_old) < Epsilon:
            print(f"Zbieżność dla k={k} po {it} iteracjach.")
            break

        S_old = S_new

    # Zapis stanu końcowego potencjału dla danego k
    V_maps.append(V_current.copy())








# --- Wizualizacja ---
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes_flat = axes.flatten()


V_min_all = min(V.min() for V in V_maps)
V_max_all = max(V.max() for V in V_maps)
V_abs_max = max(abs(V_min_all), abs(V_max_all))

X_coords = np.linspace(0, X_max, N + 1)
Y_coords = np.linspace(0, Y_max, N + 1)



#Mapy Potencjału
for idx, k in enumerate(k_values):
    if idx < 5:
        V = V_maps[idx]
        ax = axes_flat[idx]

        im = ax.contourf(X_coords, Y_coords, V.T, levels=50,
                         norm=colors.Normalize(vmin=-V_abs_max, vmax=V_abs_max),
                         cmap='viridis')

        fig.colorbar(im, ax=ax, orientation='vertical', label='Potencjał V')

        ax.set_title(f'Mapa Potencjału dla k={k}', fontsize=12)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_aspect('equal', adjustable='box')


# Rysowanie Wykresu Zbieżności
ax_convergence = axes_flat[5]

for k in k_values:
    ax_convergence.plot(np.arange(len(S_history[k])), S_history[k], label=f'k={k} iter: {len(S_history[k])}')


ax_convergence.set_ylabel('$S^{(k)}$', fontsize=12)
ax_convergence.set_xlabel('it', fontsize=12)
ax_convergence.set_title('Zbieżność Całki Funkcjonalnej $S^{(k)}$ (Zadanie 1)', fontsize=14)
ax_convergence.grid(True, linestyle='--')
ax_convergence.legend(loc='upper right')


fig.suptitle('Wizualizacja Wyników Metody Relaksacji (Zadania 1 i 2)', fontsize=18, y=1.01)
plt.show()