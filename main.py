import numpy as np
import matplotlib.pyplot as plt

# Метод Рунге-Кутты второго порядка
def runge_kutta_2(f, y0, t0, t_end, h):
    t = np.arange(t0, t_end + h, h)
    y = np.zeros((len(t), len(y0)))
    y[0] = y0

    for i in range(1, len(t)):
        k1 = f(t[i - 1], y[i - 1])
        k2 = f(t[i - 1] + h, y[i - 1] + h * k1)
        y[i] = y[i - 1] + h * (k1 + k2) / 2

    return t, y

# Тестовая задача y' = f(t, y)
def test_system(t, y):
    dy1_dt = y[0] / (2 + 2 * t) - 2 * y[1]
    dy2_dt = y[1] / (2 + 2 * t) + 2 * y[0]
    return np.array([dy1_dt, dy2_dt])

# Точное решение тестовой задачи
def exact_solution(t):
    y1_exact = np.cos(t**2) * np.sqrt(1 + t)
    y2_exact = np.sin(t**2) * np.sqrt(1 + t)
    return y1_exact, y2_exact

# Решение тестовой задачи
if __name__ == "__main__":
    # Параметры интегрирования
    t0, t_end, h = 0, 2, 0.1
    y0 = [1, 0]

    # Численное решение
    t, y_numeric = runge_kutta_2(test_system, y0, t0, t_end, h)

    # Точное решение
    y1_exact, y2_exact = exact_solution(t)

    # Построение графиков
    plt.figure(figsize=(10, 6))

    plt.plot(t, y_numeric[:, 0], 'o-', label="Численное y1")
    plt.plot(t, y_numeric[:, 1], 'o-', label="Численное y2")
    plt.plot(t, y1_exact, '-', label="Точное y1")
    plt.plot(t, y2_exact, '-', label="Точное y2")

    plt.xlabel("t")
    plt.ylabel("y")
    plt.legend()
    plt.title("Сравнение численного и точного решения")
    plt.grid()
    plt.show()
