import numpy as np
import matplotlib.pyplot as plt

L = 10.0
T = 1.0
N = 100
dx = L / (N+1)
x = np.linspace(0, L, N+1)

def shoot(omega, rho):
    u = np.zeros(N+1)
    u[0] = 0
    u[1] = 1

    for i in range(1, N):
        u[i+1] = 2*u[i] - u[i-1] - dx**2 * rho[i] * omega**2 / T * u[i]

    return u

def f(omega, rho):
    return shoot(omega, rho)[-1]

def bisection(w1, w2, rho, eps=1e-8):
    f1 = f(w1, rho)
    f2 = f(w2, rho)

    if f1 * f2 > 0:
        raise ValueError("Brak zera w podanym przedziale")

    while abs(w2 - w1) > eps:
        m = 0.5*(w1 + w2)
        fm = f(m, rho)

        if f1 * fm < 0:
            w2 = m
            f2 = fm
        else:
            w1 = m
            f1 = fm

    return 0.5*(w1 + w2)

rho_const = np.ones(N+1)

omega_scan = np.linspace(0.001, 1.5, 3000)
uN = np.array([f(w, rho_const) for w in omega_scan])

intervals = []
for i in range(len(omega_scan)-1):
    if uN[i]*uN[i+1] < 0:
        intervals.append((omega_scan[i], omega_scan[i+1]))

intervals = intervals[:4]

roots = [bisection(a,b,rho_const) for a,b in intervals]
modes = [shoot(w, rho_const) for w in roots]

# normalizacja
modes = [u/np.max(np.abs(u)) for u in modes]

w1 = roots[0]
u_w1   = shoot(w1, rho_const)
u_low  = shoot(0.95*w1, rho_const)
u_high = shoot(1.05*w1, rho_const)

alpha = 40
rho_var = 1 + alpha*(x - L/2)**2

omega_scan2 = np.linspace(0.001, 3.0, 5000)
uN2 = np.array([f(w, rho_var) for w in omega_scan2])

intervals2 = []
for i in range(len(omega_scan2)-1):
    if uN2[i]*uN2[i+1] < 0:
        intervals2.append((omega_scan2[i], omega_scan2[i+1]))

intervals2 = intervals2[:4]

roots2 = [bisection(a,b,rho_var) for a,b in intervals2]
modes2 = [shoot(w, rho_var) for w in roots2]
modes2 = [u/np.max(np.abs(u)) for u in modes2]

print("Zadanie 3: rho = 1")
for i,w in enumerate(roots):
    print(f"omega_{i+1} =", w)

print("\nZadanie 4: rho(x)")
for i,w in enumerate(roots2):
    print(f"omega_{i+1} =", w)