import numpy as np


def RK4(func, y0:np.array, dt:float, N:int, file_path:str = None) -> np.array :
    y = y0

    time = [0]
    output = [y0]

    t = 0;
    for i in range(N - 1):
        k1 = func(t, y)
        k2 = func(t + dt/2, y + dt/2 * k1)
        k3 = func(t + dt/2, y + dt/2 * k2)
        k4 = func(t + dt/2, y + dt * k3)

        y = y +dt/6 * (k1 + 2 * k2 + 2 * k3 + k4)


        output.append(y.copy())
        time.append(t.copy())

        if file_path is not None:
            with open(file_path, 'a') as f:
                f.write(f"t: {t} y={y}\n")
        t = t + dt;

    return time, np.array(output)