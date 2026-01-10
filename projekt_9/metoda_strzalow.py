import numpy as np
import matplotlib.pyplot as plt
from numba import njit

# --- Constants ---
L = 10.0
T = 1.0
N = 100
dx = L / (N + 1)
# Use N+2 to include both boundaries 0 and L
x_vals = np.linspace(0, L, N + 2)


@njit
def shoot_fast(omega, rho_type, alpha=40.0):
    """Numba-optimized shooting method returning u for all N+2 nodes."""
    u = np.zeros(N + 2)
    u[0] = 0.0
    u[1] = 1.0  # Initial 'shot' slope
    for i in range(1, N + 1):
        xi = i * dx
        rho_i = 1.0 if rho_type == 0 else 1.0 + alpha * (xi - L / 2) ** 2
        # Finite difference: u_{i+1} = 2u_i - u_{i-1} - dx^2 * (rho * w^2 / T) * u_i
        u[i + 1] = 2 * u[i] - u[i - 1] - (dx ** 2) * (rho_i * (omega ** 2) / T) * u[i]
    return u


@njit
def f_fast(omega, rho_type, alpha=40.0):
    """Boundary value u_{N+1} (at x=L) which must be 0 for a valid mode."""
    u_prev = 0.0
    u_curr = 1.0
    for i in range(1, N + 1):
        xi = i * dx
        rho_i = 1.0 if rho_type == 0 else 1.0 + alpha * (xi - L / 2) ** 2
        u_next = 2 * u_curr - u_prev - (dx ** 2) * (rho_i * (omega ** 2) / T) * u_curr
        u_prev = u_curr
        u_curr = u_next
    return u_curr


@njit
def bisection(w1, w2, rho_type, alpha=40.0, eps=1e-8):
    f1 = f_fast(w1, rho_type, alpha)
    while abs(w2 - w1) > eps:
        m = 0.5 * (w1 + w2)
        fm = f_fast(m, rho_type, alpha)
        if f1 * fm < 0:
            w2 = m
        else:
            w1 = m
            f1 = fm
    return 0.5 * (w1 + w2)


# --- Task 1: Plot u_N(omega) ---
w_scan = np.linspace(0, 1.5, 1000)
uN_vals = np.array([f_fast(w, 0) for w in w_scan])

plt.figure(figsize=(7, 4))
plt.plot(w_scan, uN_vals)
plt.axhline(0, color='black', lw=1)
plt.title("Zadanie 1: $u_N(\omega)$ dla $N=100$")
plt.xlabel("$\omega$")
plt.ylabel("$u_N$")
plt.ylim(-50, 50)  # Limit y to see crossings clearly like in lab.pdf
plt.grid(True)
plt.show()

# --- Task 2: Profile u(x) for rho=1 ---
w1 = bisection(0.2, 0.4, 0)
plt.figure(figsize=(7, 4))
plt.plot(x_vals, shoot_fast(w1, 0), label=fr'$\omega_1 \approx {w1:.5f}$')
plt.plot(x_vals, shoot_fast(w1 * 0.95, 0), label=fr'$0.95\omega_1$')
plt.plot(x_vals, shoot_fast(w1 * 1.05, 0), label=fr'$1.05\omega_1$')
plt.title("Zadanie 2: $u(x)$ dla $\\rho=1$")
plt.xlabel("x")
plt.ylabel("u(x)")
plt.legend()
plt.grid(True)
plt.show()

# --- Task 3 & 4: Eigenvalues and Modes ---


scenarios = [
    ("Zadanie 3: $u(x)$ dla $\\rho=1$", 0, 1.5),
    ("Zadanie 4: $u(x)$ dla $\\rho(x)=1+40(x-L/2)^2$", 1, 0.15)
]

for title, r_type, scan_max in scenarios:
    # Find intervals for bisection
    w_range = np.linspace(0.001, scan_max, 1000)
    roots = []
    uN_scan = [f_fast(w, r_type) for w in w_range]
    for i in range(len(w_range) - 1):
        if uN_scan[i] * uN_scan[i + 1] < 0:
            roots.append(bisection(w_range[i], w_range[i + 1], r_type))
            if len(roots) == 4: break

    plt.figure(figsize=(7, 4))
    print(f"\n{title}:")
    for i, w in enumerate(roots):
        print(f"  omega_{i + 1} = {w:.7f}")
        u_profile = shoot_fast(w, r_type)
        # To match the lab plots exactly, we plot the raw shooting profiles
        plt.plot(x_vals, u_profile, label=fr'$\omega_{{{i + 1}}} = {w:.4f}$')

    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("u(x)")
    plt.legend()
    plt.grid(True)
    plt.show()