#%% md
# ## SRI Model
# 
# $$
# \begin{cases}
# \frac{ds}{dt} = f(s(t), i(t)) = -\beta is \\
# \frac{di}{dt} = g(s(t), i(t)) = \beta is - \gamma i
# \end{cases}
# $$
# 
#%% md
# ## Imports
#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#%% md
# ## Euler
#%%
def sir(s0,i0, parameters, time, dt, func, tolerance=1e-6, max_iterations=150, show_iter = False):
    n_steps = len(time)

    s = np.zeros_like(time)
    i = np.zeros_like(time)
    r = np.zeros_like(time)
    iter_count = np.zeros(n_steps, dtype=int)
    first_iteration = {"S_n1":[], "I_n1":[]}

    s[0] = s0
    i[0] = i0
    r[0] = 1.0 - s[0] - i[0]

    print(n_steps)
    for n in range(n_steps-1):
        current_s = s[n]
        current_i = i[n]

        i_mu = current_i
        s_mu = current_s

        converged = False
        for k in range(1, max_iterations):
            s_next, i_next = func(current_s, current_i, s_mu, i_mu, dt, parameters)

            if n==0:
                first_iteration['S_n1'].append(s_next)
                first_iteration['I_n1'].append(i_next)



            if (np.abs(s_next - s_mu) < tolerance) and (np.abs(i_next - i_mu) < tolerance):
                s[n+1] = s_next
                i[n+1] = i_next
                r[n+1] = 1.0 - s_next - i_next
                iter_count[n+1] = k
                converged = True
                break

            s_mu, i_mu = s_next, i_next


        if not converged:
            s[n+1] = s_next
            i[n+1] = i_next
            r[n+1] = 1.0 - s_next - i_next
            iter_count[n+1] = max_iterations

        if show_iter:
            print(f"Iters in {n}th step: {iter_count[n]}")

        if n==1:
            df_fr = pd.DataFrame(first_iteration)
            print(df_fr)

    return s, i, r, iter_count



def euler(s_n, i_n, s_mu, i_mu, dt, params):
    beta = params['beta']
    gamma = params['gamma']

    s_next = s_n - dt * beta * s_mu * i_mu
    i_next = i_n + dt * (beta * s_mu * i_mu - gamma * i_mu)

    return s_next, i_next

#%% md
# ### Calculation for $\Delta t = 10$
# 
#%%
parameters = {
    'beta': 0.34,
    'gamma': 0.07,
}

show_iter = True
i0, s0 = 0.1, 0.9

dt = 10
time10 = np.arange(start=0.0, stop=350.1, step=dt)

ddt = 0.1
time01 = np.arange(start=0.0, stop=350.01, step=ddt)

s10,i10,r10,iter_count10 = sir(s0, i0, parameters, time10, dt, euler, show_iter = True)
s01, i01, r01, iter_count01 = sir(s0,i0, parameters, time01, ddt, euler)

plt.figure(figsize=(10,4))
plt.scatter(time10, s10, label="s(t) dt=10", color="red")
plt.scatter(time10, i10, label="i(t) dt=10", color="green")
plt.scatter(time10, r10, label="r(t) dt=10", color="blue")

plt.plot(time01, s01, label="s(t) dt=0.1", color="red")
plt.plot(time01, i01, label="i(t) dt=0.1", color="green")
plt.plot(time01, r01, label="r(t) dt=0.1", color="blue")

plt.legend()
plt.title("z.1 - Metoda Eulera - Iteracja Funkcyjna")
plt.show()

plt.figure(figsize=(10,4))
plt.plot(time10, iter_count10)
plt.title("z.1 - Metoda Eulera - Iteracja Funkcyjna (dt=10)")
plt.ylabel("# iteracji")
plt.show()

plt.figure(figsize=(10,4))
plt.plot(time01, iter_count01)
plt.title("z.1 - Metoda Eulera - Iteracja Funkcyjna (dt=0.1)")
plt.ylabel("# iteracji")
plt.show()

#%% md
# ## Trapezoid Funcional
# 
#%%
def trapezoid_func(s_n, i_n, s_mu, i_mu, dt, params):
    beta = params['beta']
    gamma = params['gamma']

    s_next = s_n - (dt/2) * (beta*s_n*i_n + beta*s_mu*i_mu)
    i_next = i_n + (dt/2) * (beta*s_n*i_n - gamma*i_n + beta*s_mu*i_mu - gamma*i_mu)

    return s_next, i_next

#%%
s10,i10,r10,iter_count10 = sir(s0, i0, parameters, time10, dt, func = trapezoid_func,  show_iter = True)
s01, i01, r01, iter_count01 = sir(s0,i0, parameters, time01, ddt, func = trapezoid_func)

plt.figure(figsize=(10,4))
plt.scatter(time10, s10, label="s(t) dt=10", color="red")
plt.scatter(time10, i10, label="i(t) dt=10", color="green")
plt.scatter(time10, r10, label="r(t) dt=10", color="blue")

plt.plot(time01, s01, label="s(t) dt=0.1", color="red")
plt.plot(time01, i01, label="i(t) dt=0.1", color="green")
plt.plot(time01, r01, label="r(t) dt=0.1", color="blue")

plt.legend()
plt.title("z.2 - Metoda trapezów - iteracja funkcjonalna")
plt.show()

plt.figure(figsize=(10,4))
plt.plot(time10, iter_count10)
plt.title("z.2 - Metoda trapezów - iteracja funkcjonalna (dt=10)")
plt.ylabel("# iteracji")
plt.show()

plt.figure(figsize=(10,4))
plt.plot(time01, iter_count01)
plt.title("z.2 - Metoda trapezów - iteracja funkcjonalna (dt=0.1)")
plt.ylabel("# iteracji")
plt.show()
#%% md
# ## Trapezoid Newtwon
#%%
def sir_trpezoid_newton(s0,i0, parameters, time, dt, func=trapezoid_func, tolerance=1e-6, max_iterations=150, show_iter = False):
    beta = parameters['beta']
    gamma = parameters['gamma']

    n_steps = len(time)

    s = np.zeros_like(time)
    i = np.zeros_like(time)
    r = np.zeros_like(time)
    iter_count = np.zeros(n_steps, dtype=int)
    first_iteration = {"S_n1":[], "I_n1":[]}

    s[0] = s0
    i[0] = i0
    r[0] = 1.0 - s[0] - i[0]

    print(n_steps)
    for n in range(n_steps-1):
        current_s = s[n]
        current_i = i[n]

        i_mu = current_i
        s_mu = current_s

        converged = False
        for k in range(1, max_iterations+1):
            s_next, i_next = func(current_s, current_i, s_mu, i_mu, dt, parameters)


            #Iteracja Newtona
            F = s_next - current_s + (dt/2) * (beta*current_s*current_i + beta*s_next*i_next)
            G = i_next - current_i - (dt/2) * ((beta*current_s*current_i - gamma*current_i) + (beta*s_next*i_next - gamma*i_next))

            J11 = 1 + (dt/2)*beta*i_next
            J12 = (dt/2)*beta*s_next

            J21 = -(dt/2)*(beta*i_next)
            J22 = 1 - (dt/2)*(beta*s_next - gamma)


            w = J11 * J22 - J12 * J21
            w_s = F * J22 - J12 * G
            w_i = J11 * G - F * J21

            if abs(w) < 1e-12:
                break;

            delta_s = w_s/w
            delta_i = w_i/w

            s_next = s_next - delta_s
            i_next = i_next - delta_i



            if n==0:
                first_iteration['S_n1'].append(s_next)
                first_iteration['I_n1'].append(i_next)



            if (abs(delta_i) < tolerance) and (abs(delta_s) < tolerance):
                s[n+1] = s_next
                i[n+1] = i_next
                r[n+1] = 1.0 - s_next - i_next
                iter_count[n+1] = k
                converged = True
                break

            s_mu, i_mu = s_next, i_next


        if not converged:
            s[n+1] = s_next
            i[n+1] = i_next
            r[n+1] = 1.0 - s_next - i_next
            iter_count[n+1] = max_iterations

        if show_iter:
            print(f"Iters in {n}th step: {iter_count[n+1]}")

        if n==1:
            df_fr = pd.DataFrame(first_iteration)
            print(df_fr)

    return s, i, r, iter_count
#%%
s10,i10,r10,iter_count10 = sir_trpezoid_newton(s0, i0, parameters, time10, dt, show_iter=True)
s01, i01, r01, iter_count01 = sir_trpezoid_newton(s0,i0, parameters, time01, ddt)

plt.figure(figsize=(10,4))
plt.scatter(time10, s10, label="s(t) dt=10", color="red")
plt.scatter(time10, i10, label="i(t) dt=10", color="green")
plt.scatter(time10, r10, label="r(t) dt=10", color="blue")

plt.plot(time01, s01, label="(dt=0.1)", color="red")
plt.plot(time01, i01, label="(dt=0.1)", color="green")
plt.plot(time01, r01, label="r(dt=0.1)", color="blue")

plt.legend()
plt.title("z.3 Metoda trapezów - iteracja Newtona")
plt.show()

plt.figure(figsize=(10,4))
plt.plot(time10, iter_count10)
plt.title("z.3 Metoda trapezów - iteracja Newtona (dt=10)")
plt.ylabel("# iteracji")
plt.show()

plt.figure(figsize=(10,4))
plt.plot(time01, iter_count01)
plt.title("z.3 Metoda trapezów - iteracja Newtona (dt=0.1)")
plt.ylabel("# iteracji")
plt.show()
#%% md
# ## Trapezod functional for $\beta = 0.06$
#%%
params2 = parameters.copy()
params2['beta'] = 0.06
s01, i01, r01, iter_count01 = sir(s0,i0, params2, time01, ddt, func = trapezoid_func)


plt.figure(figsize=(10,4))
plt.plot(time01, s01, label="s(t) dt=0.1", color="red")
plt.plot(time01, i01, label="i(t) dt=0.1", color="green")
plt.plot(time01, r01, label="r(t) dt=0.1", color="blue")

plt.legend()
plt.title("z.4 Metoda Trapezów - iteracja funkcjonalna (beta=0.06)")
plt.show()

plt.figure(figsize=(10,4))
plt.plot(time01, iter_count01)
plt.title("z.4 Metoda Trapezów - iteracja funkcjonalna (beta=0.06)")
plt.ylabel("# iteracji")
plt.show()