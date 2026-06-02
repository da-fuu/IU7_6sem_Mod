import sys
import math
import os
import matplotlib.pyplot as plt

# =====================================================================
# ТАБЛИЧНЫЕ ДАННЫЕ
# =====================================================================
TABLE_T = [2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 11000, 12000]
TABLE_SIGMA = [0.309e-3, 0.309e-2, 0.309e-1, 0.270, 2.05, 6.06, 12.0, 19.9, 29.6, 41.1, 54.1]
TABLE_LAMBDA = [0.381e-3, 0.381e-3, 0.381e-3, 0.448e-3, 0.577e-3, 0.733e-3, 0.131e-2, 0.218e-2, 0.358e-2, 0.562e-2, 0.832e-2]
TABLE_C = [1.90e-3, 1.90e-3, 0.95e-3, 0.75e-3, 0.64e-3, 0.61e-3, 0.66e-3, 0.66e-3, 1.15e-3, 1.79e-3, 2.02e-3]

TABLE_T_k = [2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
TABLE_k = [8.2e-3, 2.768e-2, 6.56e-2, 1.281e-1, 2.214e-1, 3.516e-1, 5.248e-1, 7.472e-1, 1.025]

C_LIGHT = 3e10

# =====================================================================
# ИНТЕРПОЛЯЦИЯ
# =====================================================================
def get_index(val, x_array):
    if val <= x_array[0]: return 0
    if val >= x_array[-1]: return len(x_array) - 2
    for i in range(len(x_array) - 1):
        if x_array[i] <= val <= x_array[i+1]: return i
    return 0

def interpolate_log(val, x_array, y_array):
    i = get_index(val, x_array)
    x0, x1 = x_array[i], x_array[i+1]
    y0, y1 = math.log(y_array[i]), math.log(y_array[i+1])
    return math.exp(y0 + (y1 - y0) * (val - x0) / (x1 - x0))

def get_sigma(T): return interpolate_log(T, TABLE_T, TABLE_SIGMA)
def get_lambda(T): return interpolate_log(T, TABLE_T, TABLE_LAMBDA)
def get_c(T): return interpolate_log(T, TABLE_T, TABLE_C)
def get_k(T): return interpolate_log(T, TABLE_T_k, TABLE_k)

def u_p(T):
    T_safe = max(T, 1.0)
    power = 4.799e4 / T_safe
    if power > 700: return 0.0
    return (3.084e-4) / math.expm1(power)

# =====================================================================
# МАТЕМАТИКА (ПРОГОНКА)
# =====================================================================
def progonka(A, B, C, F):
    n = len(B)
    alpha, beta, x = [0.0] * n, [0.0] * n, [0.0] * n
    alpha[1], beta[1] = C[0] / B[0], F[0] / B[0]
    for i in range(1, n - 1):
        denom = B[i] - A[i] * alpha[i]
        alpha[i+1] = C[i] / denom
        beta[i+1] = (A[i] * beta[i] + F[i]) / denom
    denom = B[n-1] - A[n-1] * alpha[n-1]
    x[n-1] = (A[n-1] * beta[n-1] + F[n-1]) / denom
    for i in range(n - 2, -1, -1):
        x[i] = alpha[i+1] * x[i+1] + beta[i+1]
    return x

# =====================================================================
# ФУНКЦИЯ-ОБЕРТКА СИМУЛЯЦИИ ДЛЯ ИССЛЕДОВАНИЯ
# =====================================================================
def run_simulation(R=0.35, I_max=1000.0, t_max=80e-6, T_w=1800.0, T0=8000.0, p_param=2.0,
                   m_sigma=1.0, m_lam=1.0, m_c=1.0, m_k=1.0):

    N = 50
    tau = 1e-6
    h = R / N
    r = [i * h for i in range(N + 1)]
    T = [T0 + (T_w - T0) * (ri / R)**p_param for ri in r]

    t = 0.0
    t_target = 80e-6 # Сравниваем профили на пике тока (80 мкс)

    def I(curr_t):
        return I_max * (curr_t / t_max) * math.exp(-(curr_t / t_max - 1))

    while t <= t_target + 1e-9:
        I_t = I(t)

        sigma = [get_sigma(T[i]) * m_sigma for i in range(N + 1)]
        lam = [get_lambda(T[i]) * m_lam for i in range(N + 1)]
        c_T = [get_c(T[i]) * m_c for i in range(N + 1)]
        k_coef = [get_k(T[i]) * m_k for i in range(N + 1)]
        up = [u_p(T[i]) for i in range(N + 1)]

        integral_sigma = sum(0.5 * (sigma[i]*r[i] + sigma[i+1]*r[i+1]) * h for i in range(N))
        E = I_t / (2.0 * math.pi * integral_sigma) if integral_sigma > 0 else 0.0

        Au, Bu, Cu, Fu = [0.0]*(N+1), [0.0]*(N+1), [0.0]*(N+1), [0.0]*(N+1)
        for i in range(1, N):
            D_minus = 1.0 / (3.0 * 0.5 * (k_coef[i-1] + k_coef[i]))
            D_plus  = 1.0 / (3.0 * 0.5 * (k_coef[i] + k_coef[i+1]))
            Au[i] = (r[i] - 0.5 * h) * D_minus / (h**2)
            Cu[i] = (r[i] + 0.5 * h) * D_plus / (h**2)
            Bu[i] = Au[i] + Cu[i] + k_coef[i] * r[i]
            Fu[i] = k_coef[i] * up[i] * r[i]
        D_half = 1.0 / (3.0 * 0.5 * (k_coef[0] + k_coef[1]))
        Cu[0] = 4.0 * D_half / (h**2)
        Bu[0] = Cu[0] + k_coef[0]
        Au[0] = 0.0
        Fu[0] = k_coef[0] * up[0]

        D_N_half = 1.0 / (3.0 * 0.5 * (k_coef[N-1] + k_coef[N]))
        Au[N] = (R - 0.5 * h) * D_N_half / (h**2)
        Bu[N] = Au[N] + 0.39 * R / h + k_coef[N] * R / 2.0
        Cu[N] = 0.0
        Fu[N] = k_coef[N] * up[N] * R / 2.0
        u_rad = progonka(Au, Bu, Cu, Fu)
        q = [C_LIGHT * k_coef[i] * (up[i] - u_rad[i]) for i in range(N + 1)]

        At, Bt, Ct, Ft = [0.0]*(N+1), [0.0]*(N+1), [0.0]*(N+1), [0.0]*(N+1)
        for i in range(1, N):
            lam_m = 0.5 * (lam[i-1] + lam[i])
            lam_p  = 0.5 * (lam[i] + lam[i+1])
            At[i] = (r[i] - 0.5 * h) * lam_m / (h**2)
            Ct[i] = (r[i] + 0.5 * h) * lam_p / (h**2)
            Bt[i] = At[i] + Ct[i] + (c_T[i] * r[i]) / tau
            Ft[i] = (c_T[i] * r[i] / tau) * T[i] + (sigma[i] * E**2 - q[i]) * r[i]
        lam_half = 0.5 * (lam[0] + lam[1])
        Ct[0] = 4.0 * lam_half / (h**2)
        Bt[0] = Ct[0] + c_T[0] / tau
        At[0] = 0.0
        Ft[0] = (c_T[0] / tau) * T[0] + sigma[0] * E**2 - q[0]
        At[N], Bt[N], Ct[N], Ft[N] = 0.0, 1.0, 0.0, T_w

        T = progonka(At, Bt, Ct, Ft)
        t += tau

    return r, T


# =====================================================================
# ЗАПУСК ИССЛЕДОВАНИЙ С СОХРАНЕНИЕМ В ФАЙЛЫ
# =====================================================================

# Словарь сценариев: 10 параметров, по 5 значений на каждый
studies = {
    "Влияние радиуса R": {"param": "R", "values": [0.15, 0.25, 0.35, 0.45, 0.55], "unit": "см"},
    "Влияние амплитуды тока I_max": {"param": "I_max", "values": [600, 800, 1000, 1200, 1400], "unit": "А"},
    "Влияние времени пика тока t_max": {"param": "t_max", "values": [40e-6, 60e-6, 80e-6, 100e-6, 120e-6], "unit": "с"},
    "Влияние температуры стенки T_w": {"param": "T_w", "values": [1000, 1400, 1800, 2200, 2600], "unit": "К"},
    "Влияние начальной температуры T_0": {"param": "T0", "values": [6000, 7000, 8000, 9000, 10000], "unit": "К"},
    "Влияние параметра формы p": {"param": "p_param", "values": [1.0, 1.5, 2.0, 3.0, 4.0], "unit": ""},
    "Множитель теплоемкости c_T": {"param": "m_c", "values": [0.5, 0.75, 1.0, 1.5, 2.0], "unit": ""},
    "Множитель оптич. поглощения k": {"param": "m_k", "values": [0.5, 0.75, 1.0, 1.5, 2.0], "unit": ""},
    "Множитель электропроводности \u03C3": {"param": "m_sigma", "values": [0.5, 0.75, 1.0, 1.5, 2.0], "unit": ""},
    "Множитель теплопроводности \u03BB": {"param": "m_lam", "values": [0.005, 0.1, 1.0, 15.0, 100.0], "unit": ""} # <-- Добавлена лямбда
}

# Создаем папку для картинок
output_dir = "research_plots"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print(f"Начат расчет. Всего 50 симуляций. Графики будут сохранены в папку '{output_dir}'")
print("-" * 50)

# Цветовая палитра для 5 линий
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

for title, setup in studies.items():
    param_name = setup["param"]
    print(f"Исследуется параметр: {param_name: <10} ...", end="", flush=True)

    plt.figure(figsize=(8, 6))

    for idx, val in enumerate(setup["values"]):
        kwargs = {param_name: val}
        r, T_res = run_simulation(**kwargs)

        # Форматирование подписи в легенде
        label_val = f"{val:g}" if val > 1e-3 else f"{val:.1e}"
        line_label = f"{param_name} = {label_val} {setup['unit']}".strip()

        plt.plot(r, T_res, label=line_label, color=colors[idx], linewidth=2)

    plt.title(f"{title} (при t = 80 мкс)", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Радиус r, см", fontsize=12)
    plt.ylabel("Температура T, К", fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend(fontsize=11)

    # Сохранение файла
    filename = os.path.join(output_dir, f"study_{param_name}.png")
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close() # Закрываем полотно

    print(" Готово!")

print("-" * 50)
print(f"Все 10 исследований завершены! Ищите картинки в папке '{output_dir}'")
