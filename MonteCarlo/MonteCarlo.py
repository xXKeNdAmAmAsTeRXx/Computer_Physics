# %%
import numpy as np

CONSTS = {
    'R': 1.0,
    'CENTER': np.array([1.2, 1.2, 1.2]),
    'N': int(10e5),
    'A': 2.4
}
# %% [markdown]
# # Plot Funtion
# %%
import matplotlib.pyplot as plt

def plot(N:np.ndarray, expected:np.ndarray, std:np.ndarray, error, analitic, title) -> None:
    fig, ax = plt.subplots(3,1, figsize = (20,30))
    ax = ax.flatten()
    ax[0].plot(N[1:], expected[1:], 'b-', label='MC')
    ax[0].set_title(title)
    ax[0].axline((N[1], analitic), (N[-1], analitic), color='r', linestyle='-', label='Analitic')
    ax[0].set_xlabel('N')
    ax[0].set_xscale('log')
    ax[0].set_ylabel('expected')
    ax[0].set_xlim(N[1], N[-1])
    ax[0].legend()

    ax[1].plot(N[1:], error[1:], 'b-', label='MC')
    ax[1].axline((N[1], 0), (N[-1], 0), color='r', linestyle='-', label='Analitic')
    ax[1].set_xlabel('N')
    ax[1].set_xscale('log')
    ax[1].set_ylabel('error')
    ax[1].set_xlim(N[1], N[-1])
    ax[1].legend()


    ax[2].plot(N[1:], std[1:], 'b-')
    ax[2].set_xlabel('N')
    ax[2].set_ylabel('std')
    ax[2].set_yscale('log')
    ax[2].set_xscale('log')
    ax[2].set_xlim(N[1], N[-1])


    plt.show()

# %% [markdown]
# # Sphere Volume
# %%
from numba import jit

## SEED
np.random.seed = 42

def sphere_volume(R:float) -> float:
    return (4/3)*np.pi*R*R*R

@jit(nopython=True)
def sphere_volume_MC(R:float, A:float, center:np.ndarray[float] ,N:int) -> tuple[np.ndarray,np.ndarray,np.ndarray]:
    sqaure_vol:float = A**3
    sum:float = 0.0
    sum_of_squares:float = 0.0
    V = (4/3)*np.pi*R*R*R

    step = np.zeros(int(N / 100))
    expected = np.zeros(int(N / 100))
    variance = np.zeros(int(N / 100))
    error = np.zeros(int(N / 100))
    std = np.zeros(int(N / 100))

    for i in range(1, N+1):
        # Random Point
        new_point = np.random.uniform(0, A, size=3)

        # result
        dist_sq = np.sum((new_point - center)**2, axis=0)
        multiplyer = 1.0 if dist_sq <= R*R else 0.0

        # Sums
        sum += sqaure_vol * multiplyer
        sum_of_squares += (sqaure_vol * multiplyer)**2

        # Checkpoints
        if i%100==0:
            idx = int(i/100) - 1
            step[idx] = i
            expected[idx] = sum/i
            variance[idx] = (sum_of_squares - (sum**2)/i)/(i-1)
            std[idx] = np.sqrt(variance[idx]/i)
            error[idx] = np.abs(sum/i - V)



    return step, expected, std, error


# %% [markdown]
# ## Results
# %%
step, expected, std, error = sphere_volume_MC(CONSTS['R'], CONSTS['A'], CONSTS['CENTER'] ,CONSTS['N'])
v = sphere_volume(CONSTS['R'])

plot(step, expected, std, error, v, 'Sphere Volume')
print(expected[-1])
# %% [markdown]
# # Moment of Interia
# %%
import numpy as np
from numba import jit

def sphere_interia(R:float) -> float:
    return (2/5)*(4/3)*np.pi*R*R*R*R*R

@jit(nopython=True)
def sphere_interia_MC(R:float, A:float, center:np.ndarray, axis:np.ndarray, N:int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    sqaure_vol:float = A**3
    sum_val:float = 0.0
    sum_of_squares:float = 0.0
    V_exact = (2/5)*(4/3)*np.pi*R*R*R*R*R

    step = np.zeros(int(N / 100))
    expected = np.zeros(int(N / 100))
    variance = np.zeros(int(N / 100))
    error = np.zeros(int(N / 100))
    std = np.zeros(int(N / 100))

    for i in range(1, N + 1):
        new_point = np.random.uniform(0, A, size=3)

        dist_sq = np.sum((new_point - center)**2)
        multiplier = 1.0 if dist_sq <= R*R else 0.0

        dist_ax_sq = np.sum((new_point[0:2] - axis)**2)
        val = multiplier * sqaure_vol * dist_ax_sq

        sum_val += val
        sum_of_squares += val**2

        if i % 100 == 0:
            idx = int(i / 100) - 1
            step[idx] = i
            expected[idx] = sum_val / i
            if i > 1:
                variance[idx] = (sum_of_squares - (sum_val**2) / i) / (i - 1)
            else:
                variance[idx] = 0.0
            std[idx] = np.sqrt(variance[idx] / i)
            error[idx] = np.abs(expected[idx] - V_exact)

    return step, expected, std, error
# %% [markdown]
# ## Results
# %%
import matplotlib.pyplot as plt

step, expected, std, error = sphere_interia_MC(CONSTS['R'],CONSTS['A'], CONSTS['CENTER'], CONSTS['CENTER'][0:2], CONSTS['N'])
I = sphere_interia(CONSTS['R'])
plot(step, expected, std, error, I, 'Sphere Inertia')
# %% [markdown]
# # Steiner Theorem
# %%
def steiner(I_0:float, m:float, d:float):
    return I_0 + m*d**2

p_0 = np.array([1.3, 1.2])
I_0 = sphere_interia(CONSTS['R'])
d = 0.1
M = sphere_volume(CONSTS['R'])

I2 = steiner(I_0, M, d)

step, expected, std, error = sphere_interia_MC(CONSTS['R'],CONSTS['A'], CONSTS['CENTER'], p_0, CONSTS['N'])

plot(step, expected, std, error, I2, 'Steiner Theorem')
# %% [markdown]
# # New Parameters
# %%
CONSTS['A'] = 2
CONSTS['CENTER'] = np.array([1,1,1])

step, expected, std, error = sphere_volume_MC(CONSTS['R'], CONSTS['A'], CONSTS['CENTER'] ,CONSTS['N'])
plot(step, expected, std, error, v, 'Volume new params')


step, expected, std, error = sphere_interia_MC(CONSTS['R'],CONSTS['A'], CONSTS['CENTER'], CONSTS['CENTER'][0:2],CONSTS['N'])
plot(step, expected, std, error, I, 'Interia new params')