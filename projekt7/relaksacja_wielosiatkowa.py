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
V_grid = np.zeros((N + 1, N + 1))

rho = np.zeros((N + 1, N + 1))

# Poprawna inicjalizacja rho używając meshgrid (jak w poprawnym rozwiązaniu)
# lub pozostawienie oryginalnej pętli (jak w Twoim rozwiązaniu) - pozostawiam pętlę dla minimalnej ingerencji
for i in range(N + 1):
    for j in range(N + 1):
        term1 = np.exp(
            -((x[i] - 0.35 * x_max) ** 2 + (y[j] - 0.5 * y_max) ** 2) / sigma ** 2
        )

        term2 = np.exp(
            -((x[i] - 0.65 * x_max) ** 2 + (y[j] - 0.5 * y_max) ** 2) / sigma ** 2
        )

        rho[i][j] = term1 - term2


# Funkcja calc_w jest niepotrzebna i błędna. Usunięta.


def avg_rho(k, rho, N):
    new_rho = np.zeros((N + 1, N + 1), dtype=float)
    r = k // 2
    norm = float(k * k)  # Prawidłowa normalizacja dla wag [0.5, 1, ..., 1, 0.5] to k^2

    # Prawidłowe wagi dla rzutowania (projekcji)
    w1 = np.ones(2 * r + 1, dtype=float)
    if k > 1:
        w1[0] = 0.5
        w1[-1] = 0.5
    W = np.outer(w1, w1)

    for i in range(k, N, k):
        for j in range(k, N, k):
            # Używamy uśredniania ważonego z poprawnego rozwiązania
            block = rho[i - r: i + r + 1, j - r: j + r + 1]
            if block.shape == W.shape:
                new_rho[i, j] = float((block * W).sum() / norm)

    # Przeniesienie oryginalnej inicjalizacji rho na zewnątrz pętli po k (jak w poprawnym rozwiązaniu)
    # Zwracamy rho_tilde, nie nadpisując pierwotnej rho, ale ponieważ oryginalna pętla nadpisuje rho, to ją zachowuję.
    return new_rho


@jit(nopython=True)
def relax(V, k, rho, dx):
    factor = (k * dx) ** 2
    for i in range(k, N, k):
        for j in range(k, N, k):
            V[i, j] = 0.25 * (
                    V[i + k, j]
                    + V[i - k, j]
                    + V[i, j + k]
                    + V[i, j - k]  # Poprawione: było V[i, j+k]
                    + factor * rho[i, j]
            )

    return V


@jit(nopython=True)
def single_integral(V: np.ndarray, N: int, k: int, dx: float, rho: np.ndarray):
    Sum = 0.0
    area_over_2 = ((k * dx) ** 2) / 2.0
    denom = 2 * k * dx

    for i in range(0, N, k):
        for j in range(0, N, k):
            ip = i + k
            jp = j + k

            # Prawidłowe numeryczne pochodne (jak w poprawnym rozwiązaniu)
            dVdx = ((V[ip, j] - V[i, j]) + (V[ip, jp] - V[i, jp])) / denom
            dVdy = ((V[i, jp] - V[i, j]) + (V[ip, jp] - V[ip, j])) / denom

            # Wzór na S: S += area_over_2 * ( (dVdx^2 + dVdy^2) - 2 * rho_t * V )
            Sum += area_over_2 * ((dVdx * dVdx + dVdy * dVdy) - 2.0 * rho[i, j] * V[i, j])

    return Sum


@jit(nopython=True)
def interpolate(V, k):
    half_k = k // 2
    for i in range(0, N, k):
        ip = i + k
        for j in range(0, N, k):
            jp = j + k

            # Interpolacja wewnątrz kwadratu (na środku)
            V[i + half_k, j + half_k] = 0.25 * (V[i, j] + V[ip, j] + V[i, jp] + V[ip, jp])

            # Interpolacja na krawędziach
            V[ip, j + half_k] = 0.5 * (V[ip, j] + V[ip, jp])  # Poprawione: było V[i+k, i+k]
            V[i + half_k, jp] = 0.5 * (V[i, jp] + V[ip, jp])
            V[i + half_k, j] = 0.5 * (V[i, j] + V[ip, j])
            V[i, j + half_k] = 0.5 * (V[i, j] + V[i, jp])

    # Warunki brzegowe są stałe i nie powinny być resetowane w pętli.
    # Wartości na granicy (i=0, i=N, j=0, j=N) pozostają zerowe po interpolacji (o ile były zerowe).
    # Usuwam zbędne resetowanie:
    # for i in range(0, N+1): V[i,0] = 0; V[i, N] = 0
    # for j in range(0, N+1): V[0,j] = 0; V[N,j] = 0

    return V


# --- Główna pętla i generowanie wykresów ---

K = [16, 8, 4, 2, 1]
current_sum = 0
it_total = 0
it = 0

# Dodane zmienne do przechowywania wyników dla wykresów
snapshots = {}
histories = {}
iters_total = []

# Nowa, jednolita siatka dla rho (zgodnie z poprawnym kodem)
original_rho = rho.copy()

for k in K:
    print(f'calculating k={k}')

    # Używamy uśrednionej gęstości ładunku
    rho_t = avg_rho(k, original_rho, N)

    List_of_sums = []
    sum_prev = 0.0  # Musi być float

    # Wprowadzenie relax do V_grid
    V_grid = relax(V_grid, k, rho_t, D_x)
    sum_prev = single_integral(V_grid, N, k, D_x, rho_t)
    List_of_sums.append(sum_prev)
    it_k = 1

    while it_k < 2000:
        it_k += 1
        V_grid = relax(V_grid, k, rho_t, D_x)
        Sum = single_integral(V_grid, N, k, D_x, rho_t)

        List_of_sums.append(Sum)
        if np.abs((Sum - sum_prev) / sum_prev) < 1e-8:
            break

        sum_prev = Sum


    it_total += it_k
    iters_total.append(it_total)
    histories[k] = np.asarray(List_of_sums, dtype=float)
    snapshots[k] = V_grid.copy()

    if k > 1:
        V_grid = interpolate(V_grid, k)

# --- Generowanie wykresów ---

fig, axes = plt.subplots(2, 3, figsize=(12, 7))
axes = axes.flatten()

heatmap_order = [(0, 16), (1, 8), (2, 4), (3, 2), (4, 1)]
tick_vals = [0, 5, 10, 15, 20, 25]

# Wykresy potencjału (5 podwykresów)
for ax_idx, (k) in enumerate(K):
    ax = axes[ax_idx]
    total_it = iters_total[ax_idx]

    Vsub = snapshots[k][::k, ::k]

    # Poprawne określenie vmax/vmin
    vmax = np.ceil(np.max(np.abs(Vsub)))
    if vmax == 0: vmax = 1.0  # Zabezpieczenie przed zerowym vmax
    vmin = -vmax

    im = ax.imshow(
        Vsub.T,
        origin="lower",
        extent=[0, x_max, 0, y_max],
        interpolation="nearest",
        vmin=vmin,
        vmax=vmax,
        cmap="viridis",
        aspect="equal",
    )

    ax.set_title(f"k={k}: {total_it} iteracji")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_xticks(tick_vals)
    ax.set_yticks(tick_vals)
    ax.tick_params(direction="in", top=True, right=True, length=8)

    # Lepsze zarządzanie paskiem kolorów
    # fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, ticks=list(range(int(vmin), int(vmax) + 1, 1)))

# Wykres funkcjonału S(it) (ostatni podwykres)
axS = axes[5]
colors = {16: "purple", 8: "green", 4: "deepskyblue", 2: "orange", 1: "gold"}

offset = 0
for k, hist in histories.items():
    it_k = len(hist)
    y = hist
    x = np.arange(offset, offset + it_k)
    axS.plot(x, y, label=f"k={k}", color=colors.get(k, 'black'), linewidth=2)
    offset += it_k

axS.set_title("S(it)")
axS.set_xlabel("it")
axS.set_ylabel("S")
axS.set_xlim(0, 1000)
# axS.set_ylim(-65, -35) # Użycie konkretnych limitów z poprawnych wyników
axS.grid(True, linestyle="--", alpha=0.6)
axS.legend(loc="upper right", frameon=False)
axS.tick_params(direction="in", top=True, right=True, length=8)

plt.tight_layout()
plt.show()