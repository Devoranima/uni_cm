import numpy as np
import matplotlib.pyplot as plt

# Тестовая задача y' = f(t, y)


def test_system(t, y):
    """
    Система уравнений для тестирования метода.

    :param t: Время
    :param y: Вектор функции [y1, y2]
    :return: Производные [dy1/dt, dy2/dt]
    """
    dy1_dt = y[0] / (2 + 2 * t) - 2*t * y[1]
    dy2_dt = y[1] / (2 + 2 * t) + 2*t * y[0]
    return np.array([dy1_dt, dy2_dt])

# Точное решение тестовой задачи


def exact_solution(t):
    """
    Точное решение системы уравнений.

    :param t: Время (массив)
    :return: Векторы точных решений [y1, y2]
    """
    y1_exact = np.cos(t**2) * np.sqrt(1 + t)
    y2_exact = np.sin(t**2) * np.sqrt(1 + t)
    return y1_exact, y2_exact

# Метод Рунге-Кутты второго порядка


def runge_kutta_2_test(f, y0, t0, t_end, h):
    t = np.arange(t0, t_end, h)
    y = np.zeros((len(t), len(y0)))
    y[0] = y0

    for i in range(1, len(t)):
        k1 = f(t[i - 1], y[i - 1])
        k2 = f(t[i - 1] + h, y[i - 1] + h * k1)
        y[i] = y[i - 1] + h * (k1 + k2) / 2

    return t, y

# Метод Рунге-Кутты второго порядка для задачи Коши


def runge_kutta_2(f, y0, t0, t_end, h):
    """
    Решение задачи Коши методом Рунге-Кутты второго порядка.

    :param f: Правая часть системы уравнений y' = f(t, y)
    :param y0: Начальное условие (вектор)
    :param t0: Начальное значение времени
    :param t_end: Конечное значение времени
    :param h: Шаг интегрирования
    :return: Массив времени t и массив решений y
    """
    n = int((t_end - t0)/h)+1
    t = np.linspace(t0, t_end, n)
    y = np.zeros((n, len(y0)))
    y[0] = y0

    for i in range(n-1):
        k1 = f(t[i], y[i])
        k2 = f(t[i] + h, y[i] + h * k1)
        y[i+1] = y[i] + h * (k1 + k2) / 2

    return t, y


def generateErrorGrid(base=10, num=21):
    x_start = 0
    x_end = np.log10(10)/np.log10(base)

    h_def = np.linspace(x_start, x_end, num)
    return base**h_def/1000


# Основная функция для анализа ошибок
if __name__ == "__main__":
    # Начальные условия
    t0, t_end = 0, 2
    y0 = [1, 0]

    h_values = generateErrorGrid()

    # Для хранения ошибок
    max_errors = []
    error_ratios = []

    for h in h_values:
        # Численное решение
        t, y_numeric = runge_kutta_2_test(test_system, y0, t0, t_end, h)

        y1_numeric, y2_numeric = y_numeric[:, 0], y_numeric[:, 1]

        # Точное решение
        y1_exact, y2_exact = exact_solution(t)

        err_max_y1 = np.abs(y1_numeric - y1_exact)
        err_max_y2 = np.abs(y2_numeric - y2_exact)
        # Максимальная ошибка
        error = max(max(err_max_y1), max(err_max_y2))

        max_errors.append(error)
        error_ratios.append(error / h**2)

    # Построение графиков
    plt.figure(figsize=(12, 6))

    # Зависимость максимальной погрешности от h
    plt.subplot(1, 2, 1)
    plt.plot(h_values, max_errors, 'x-', label="Max Error")
    plt.xlabel("h (шаг)")
    plt.ylabel("Max Error")
    # plt.xscale("log")
    # plt.yscale("log")
    plt.title("Зависимость максимальной ошибки от шага h")
    plt.grid()
    plt.legend()

    # Зависимость ошибки/h^2 от h
    plt.subplot(1, 2, 2)
    plt.plot(h_values, error_ratios, 'o-', label="Error / h^2")
    # plt.xscale('log')
    # plt.yscale('log')
    plt.xlabel("h (шаг)")
    plt.ylabel("Error / h^2")
    plt.title("Зависимость ошибки/h^2 от шага h")
    plt.grid()
    plt.legend()

    plt.tight_layout()
    plt.show()
