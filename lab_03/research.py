import math
import matplotlib.pyplot as plt

TABLE_T = [2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
TABLE_K = [8.200e-3, 2.768e-2, 6.560e-2, 1.281e-1, 2.214e-1, 3.516e-1, 5.248e-1, 7.472e-1, 1.025e+0]

TABLE_XI = [math.log(t) for t in TABLE_T]
TABLE_ETA = [math.log(k) for k in TABLE_K]

DEFAULT_PARAMS = {
    'Tw': 2000.0,
    'T0': 10000.0,
    'p': 4.0,
    'R': 0.35
}

def temp(r, params):
    """Температурное поле с учетом текущих параметров."""
    z = r / params['R']
    return (params['Tw'] - params['T0']) * (z**params['p']) + params['T0']

def plank(T):
    """Функция Планка."""
    return (3.084e-4) / (math.exp(4.799e4 / T) - 1.0)

def get_k(T):
    """Интерполяция коэффициента поглощения."""
    xi_val = math.log(T)
    if xi_val <= TABLE_XI[0]:
        return math.exp(TABLE_ETA[0])
    if xi_val >= TABLE_XI[-1]:
        return math.exp(TABLE_ETA[-1])

    for i in range(len(TABLE_XI) - 1):
        if TABLE_XI[i] <= xi_val <= TABLE_XI[i + 1]:
            x1, x2 = TABLE_XI[i], TABLE_XI[i + 1]
            y1, y2 = TABLE_ETA[i], TABLE_ETA[i + 1]
            eta = y1 + (y2 - y1) * (xi_val - x1) / (x2 - x1)
            return math.exp(eta)

def shoot_implicit_euler(xi, params, N_steps=1000):
    """Один выстрел неявным методом Эйлера."""
    R = params['R']
    h = R / N_steps

    r, F = 0.0, 0.0
    u = xi * plank(temp(0, params))

    history = {'r': [r], 'u': [u], 'F': [F]}

    for _ in range(N_steps):
        r_next = r + h
        T_next = temp(r_next, params)
        k_next = get_k(T_next)
        up_next = plank(T_next)

        # Неявный шаг
        denom = 1.0 - 3.0 * (h**2) * (k_next**2) + (h / r_next)
        num = F - h * k_next * (u - up_next)

        F_next = num / denom
        u_next = u - 3.0 * h * k_next * F_next

        r, u, F = r_next, u_next, F_next

        history['r'].append(r)
        history['u'].append(u)
        history['F'].append(F)

    # Невязка на правой границе: F(R) - 0.39 * u(R) = 0
    residual = history['F'][-1] - 0.39 * history['u'][-1]
    return residual, history

def solve_bvp(params, max_iter=30):
    """Находит решение краевой задачи, подбирая xi методом секущих."""
    xi0, xi1 = 0.5, 1.0
    res0, _ = shoot_implicit_euler(xi0, params)
    res1, history = shoot_implicit_euler(xi1, params)

    for _ in range(max_iter):
        if abs(res1 - res0) < 1e-15:
            break

        xi_new = xi1 - res1 * (xi1 - xi0) / (res1 - res0)
        xi_new = max(0.0, min(1.0, xi_new)) # Защита от вылета за границы физики

        xi0, res0 = xi1, res1
        xi1 = xi_new
        res1, history = shoot_implicit_euler(xi1, params)

        if abs(res1) < 1e-8:
            break

    return history

def run_parametric_study(param_name, values, title):
    """Запускает серию расчетов для разных значений одного параметра и строит графики."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(title, fontsize=14, fontweight='bold')

    colors = ['blue', 'green', 'red', 'purple']

    for val, color in zip(values, colors):
        current_params = DEFAULT_PARAMS.copy()
        current_params[param_name] = val

        print(f"Считаю {param_name} = {val}...")
        hist = solve_bvp(current_params)

        ax1.plot(hist['r'], hist['u'], color=color, linewidth=2, label=f'{param_name} = {val}')
        ax2.plot(hist['r'], hist['F'], color=color, linewidth=2, label=f'{param_name} = {val}')

    ax1.set_xlabel('Радиус r, см')
    ax1.set_ylabel('Плотность энергии u(r), Дж/см³')
    ax1.set_title('Плотность энергии излучения')
    ax1.grid(True, linestyle='--')
    ax1.legend()

    ax2.set_xlabel('Радиус r, см')
    ax2.set_ylabel('Поток излучения F(r)')
    ax2.set_title('Поток излучения')
    ax2.grid(True, linestyle='--')
    ax2.legend()

    plt.tight_layout()
    plt.show()

def main():
    run_parametric_study('Tw', [2000.0, 4000.0, 6000.0],
                         '1. Влияние температуры стенки (Tw)')
    run_parametric_study('T0', [6000.0, 8000.0, 10000.0],
                         '2. Влияние температуры в центре (T0)')
    run_parametric_study('p', [1.0, 2.0, 4.0, 8.0],
                         '3. Влияние профиля температуры (параметр p)')
    run_parametric_study('R', [0.15, 0.35, 0.55],
                         '4. Влияние радиуса цилиндра (R)')

if __name__ == '__main__':
    main()
