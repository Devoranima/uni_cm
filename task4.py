import numpy as np
import matplotlib.pyplot as plt

# Уравнения для системы (2), (3)
def drop_system(s, y, alpha):
    r, Z, phi = y
    dr_ds = np.cos(phi)
    dZ_ds = -np.sin(phi)
    dphi_ds = -np.sin(phi) / r - Z / alpha**2
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

    return s, y

# Основная функция для решения задачи
if __name__ == "__main__":
    # Параметры задачи
    alpha = 0.39  # капиллярная постоянная для воды (в см)

    # Наборы начальных условий
    initial_conditions = [
        (1.0, np.pi / 6),
        (1.0, np.pi / 4),
        (1.0, np.pi / 3),
        (1.5, np.pi / 6),
        (1.5, np.pi / 4)
    ]

    # Параметры интегрирования
    s0, s_end, h = 0, 10, 0.01

    plt.figure(figsize=(15, 10))

    for i, (z0, phi0) in enumerate(initial_conditions):
        # Начальные условия
        y0 = [0, -z0, phi0]

        # Решение системы
        s, y = runge_kutta_2_drop(drop_system, y0, s0, s_end, h, alpha)
        r, Z, phi = y[:, 0], y[:, 1], y[:, 2]

        # Графики профиля капли в координатах r, Z
        plt.subplot(2, 3, i + 1)
        plt.plot(r, Z, label=f"z0={z0}, phi0={phi0}")
        plt.xlabel("r (см)")
        plt.ylabel("Z (см)")
        plt.title(f"Форма капли при z0={z0}, phi0={phi0}")
        plt.grid()
        plt.legend()

        # Графики r(s), Z(s), phi(s)
        plt.subplot(2, 3, 6)
        plt.plot(s, r, label=f"r(s), z0={z0}, phi0={phi0}")
        plt.plot(s, Z, label=f"Z(s), z0={z0}, phi0={phi0}")
        plt.plot(s, phi, label=f"phi(s), z0={z0}, phi0={phi0}")

    plt.subplot(2, 3, 6)
    plt.xlabel("s")
    plt.ylabel("Значения функций")
    plt.title("Зависимости r, Z, phi от s для всех начальных условий")
    plt.grid()
    plt.legend()

    plt.tight_layout()
    plt.show()
