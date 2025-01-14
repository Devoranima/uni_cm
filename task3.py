import numpy as np
import matplotlib.pyplot as plt

# Тестовая система для проверки ошибки
def test_system(t, y):
    y1, y2 = y
    dy1_dt = y1 / (2 + 2 * t) - 2 * y2
    dy2_dt = y2 / (2 + 2 * t) + 2 * y1
    return np.array([dy1_dt, dy2_dt])

# Точное решение для тестовой задачи
def exact_solution(t):
    y1_exact = np.cos(t**2) * np.sqrt(1 + t)
    y2_exact = np.sin(t**2) * np.sqrt(1 + t)
    return y1_exact, y2_exact

# Метод Рунге-Кутты второго порядка
def runge_kutta_2_test(f, y0, t0, t_end, h):
    t = np.arange(t0, t_end + h, h)
    y = np.zeros((len(t), len(y0)))
    y[0] = y0

    for i in range(1, len(t)):
        k1 = f(t[i - 1], y[i - 1])
        k2 = f(t[i - 1] + h, y[i - 1] + h * k1)
        y[i] = y[i - 1] + h * (k1 + k2) / 2

    return t, y

# Основная функция для анализа ошибок
if __name__ == "__main__":
    # Начальные условия
    t0, t_end = 0, 2
    y0 = [1, 0]

    # Различные значения шага h
    h_values = [0.2, 0.1, 0.05, 0.01]

    # Для хранения ошибок
    max_errors = []
    error_ratios = []

    for h in h_values:
        # Численное решение
        t, y_numeric = runge_kutta_2_test(test_system, y0, t0, t_end, h)
        y1_numeric, y2_numeric = y_numeric[:, 0], y_numeric[:, 1]

        # Точное решение
        y1_exact, y2_exact = exact_solution(t)

        # Максимальная ошибка
        error = np.max(np.abs(y1_numeric - y1_exact))
        max_errors.append(error)
        error_ratios.append(error / h**2)

    # Построение графиков
    plt.figure(figsize=(12, 6))

    # Зависимость максимальной погрешности от h
    plt.subplot(1, 2, 1)
    plt.plot(h_values, max_errors, 'o-', label="Max Error")
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel("h (шаг)")
    plt.ylabel("Max Error")
    plt.title("Зависимость максимальной ошибки от шага h")
    plt.grid()
    plt.legend()

    # Зависимость ошибки/h^2 от h
    plt.subplot(1, 2, 2)
    plt.plot(h_values, error_ratios, 'o-', label="Error / h^2")
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel("h (шаг)")
    plt.ylabel("Error / h^2")
    plt.title("Зависимость ошибки/h^2 от шага h")
    plt.grid()
    plt.legend()

    plt.tight_layout()
    plt.show()
