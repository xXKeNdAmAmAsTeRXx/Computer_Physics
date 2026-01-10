import numpy as np
import matplotlib.pyplot as plt
from numba import njit

# Constants provided in the project description
L = 10.0
T = 1.0
N = 100
dx = L / (N + 1)
eps = 1e-8


@njit
def get_u_N(omega, rho_type, alpha=0.0):
    """
    Calculates the displacement at the last node u_N using the shooting method.
    rho_type: 0 for constant rho=1, 1 for variable rho(x).
    """
    u_prev = 0.0  # u_0
    u_curr = 1.0  # u_1

    for i in range(1, N):
        x = i * dx
        if rho_type == 0:
            rho = 1.0
        else:
            rho = 1.0 + alpha * (x - L / 2) ** 2

        # Recurrence relation: u_{i+1}
        u_next = - (dx ** 2) * (rho * (omega ** 2) / T) * u_curr - u_prev + 2 * u_curr
        u_prev = u_curr
        u_curr = u_next

    return u_curr


@njit
def bisection(w1, w2, rho_type, alpha=0.0):
    """Finds the root (zero) of u_N(omega) using the bisection method [cite: 29-35]."""
    while abs(w1 - w2) > eps:
        m = (w1 + w2) / 2.0
        if get_u_N(w1, rho_type, alpha) * get_u_N(m, rho_type, alpha) < 0:
            w2 = m
        else:
            w1 = m
    return (w1 + w2) / 2.0


def get_full_u(omega, rho_type, alpha=0.0):
    """Generates the full array u(x) for plotting[cite: 27]."""
    u = np.zeros(N + 1)
    u[0] = 0.0
    u[1] = 1.0
    for i in range(1, N):
        x = i * dx
        rho = 1.0 if rho_type == 0 else 1.0 + alpha * (x - L / 2) ** 2
        u[i + 1] = - (dx ** 2) * (rho * (omega ** 2) / T) * u[i] - u[i - 1] + 2 * u[i]
    return u


# --- Task 1: Plot u_N(omega) ---
omegas = np.linspace(0, 1.5, 500)
u_N_vals = np.array([get_u_N(w, 0) for w in omegas])

plt.figure(figsize=(10, 4))
plt.plot(omegas, u_N_vals, label='$u_N(\omega)$')
plt.axhline(0, color='black', lw=1)
plt.title("Task 1: $u_N(\omega)$ for constant $\\rho$")
plt.xlabel("$\omega$")
plt.ylabel("$u_N$")
plt.grid()
plt.show()

# --- Task 3 & 4: Find 4 lowest eigenvalues ---
for label, rho_t, alpha_val in [("Constant $\\rho$", 0, 0.0), ("Variable $\\rho$", 1, 40.0)]:
    print(f"\n--- {label} ---")
    roots = []
    # Heuristic to find intervals for bisection by scanning
    search_range = np.linspace(0.01, 2.0 if rho_t == 0 else 0.5, 1000)
    for i in range(len(search_range) - 1):
        if get_u_N(search_range[i], rho_t, alpha_val) * get_u_N(search_range[i + 1], rho_t, alpha_val) < 0:
            root = bisection(search_range[i], search_range[i + 1], rho_t, alpha_val)
            roots.append(root)
            if len(roots) == 4: break

    plt.figure()
    for i, w in enumerate(roots):
        print(f"Mode {i + 1}: omega = {w:.8f}")
        u_plot = get_full_u(w, rho_t, alpha_val)
        plt.plot(np.linspace(0, L, N + 1), u_plot, label=f"$\omega_{i + 1}$={w:.4f}")

    plt.title(f"First 4 Eigenmodes ({label})")
    plt.legend()
    plt.show()