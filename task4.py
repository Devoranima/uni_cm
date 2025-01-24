import numpy as np
import matplotlib.pyplot as plt

# Уравнения для системы (2), (3)


def drop_system(s, y, alpha):
    r, Z, phi = y
    dr_ds = np.cos(phi)
    dZ_ds = -np.sin(phi)

    k = 0
    if r == 0:
        k = Z/(2*alpha**2)
    else:
        k = np.sin(phi) / r

    dphi_ds = -k - Z / alpha**2
    return np.array([dr_ds, dZ_ds, dphi_ds])

# Метод Рунге-Кутты второго порядка


def runge_kutta_2_drop(f, y0, s0, s_end, h, alpha):
    s = np.arange(s0, s_end + h, h)
    y = np.zeros((len(s), len(y0)))
    y[0] = y0

    for i in range(1, len(s)):
        k1 = f(s[i - 1], y[i - 1], alpha)
        k2 = f(s[i - 1] + h, y[i - 1] + h * k1, alpha)
        y[i] = y[i - 1] + h * (k1 + k2) / 2

        if np.pi - y[i][2] < 0:
            s = s[:i]
            y = y[:i]
            break
    return s, y


# Основная функция для решения задачи
if __name__ == "__main__":
    # Параметры задачи
    alpha = 0.39  # капиллярная постоянная для воды (в см)

    # Наборы начальных условий
    initial_conditions = [
        (1.0, np.pi / 12),
        (1.0, np.pi / 8),
        (1.0, np.pi / 6),
        (1.5, np.pi / 12),
        (1.5, np.pi / 8),
        (1.5, np.pi / 6)
    ]

    dep = []
    s_dep = []

    # Параметры интегрирования
    s0, s_end, h = 0, 10, 0.01

    plt.figure("Форма капли в зависимости от начальных условий",
               figsize=(15, 10))

    for i, (z0, phi0) in enumerate(initial_conditions):
        # Начальные условия
        y0 = [0, -z0, phi0]

        # Решение системы
        s, y = runge_kutta_2_drop(drop_system, y0, s0, s_end, h, alpha)
        dep.append(y)
        s_dep.append(s)

        r, Z, phi = y[:, 0], y[:, 1], y[:, 2]

        # Графики профиля капли в координатах r, Z
        plt.subplot(2, 3, i + 1)
        plt.plot(r, Z, label=f"r(s), z(s)")
        plt.xlabel("r")
        plt.ylabel("Z")
        plt.title(f"z0={z0}, phi0={phi0}")
        plt.grid()
        plt.legend()


    plt.figure("Зависимости r, Z, phi от s для всех начальных условий", figsize=(12, 8))
    
    dep_names = ["r", "Z", "phi"]
    
    for i, (z0, phi0) in enumerate(initial_conditions):
        y = dep[i]
        r, Z, phi = y[:, 0], y[:, 1], y[:, 2]
        for j in range(1, 4):
            y_dep = y[:, j-1]
            plt.subplot(1, 3, j)
            plt.plot(s_dep[i], y_dep, label=f"{dep_names[j-1]}(s), z0={z0}, phi0={phi0}")
            plt.xlabel("s")
            plt.ylabel(f"{dep_names[j-1]}(s)") 
            plt.grid()
            plt.legend()

        

    plt.show()
