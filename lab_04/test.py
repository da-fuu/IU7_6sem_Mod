import sys
import math
import matplotlib.pyplot as plt

# =====================================================================
# ТАБЛИЧНЫЕ ДАННЫЕ И ИНТЕРПОЛЯЦИЯ
# =====================================================================
TABLE_T = [2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 11000, 12000]
TABLE_SIGMA = [0.309e-3, 0.309e-2, 0.309e-1, 0.270, 2.05, 6.06, 12.0, 19.9, 29.6, 41.1, 54.1]
TABLE_LAMBDA = [0.381e-3, 0.381e-3, 0.381e-3, 0.448e-3, 0.577e-3, 0.733e-3, 0.131e-2, 0.218e-2, 0.358e-2, 0.562e-2, 0.832e-2]
TABLE_C = [1.90e-3, 1.90e-3, 0.95e-3, 0.75e-3, 0.64e-3, 0.61e-3, 0.66e-3, 0.66e-3, 1.15e-3, 1.79e-3, 2.02e-3]
TABLE_T_k = [2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
TABLE_k = [8.2e-3, 2.768e-2, 6.56e-2, 1.281e-1, 2.214e-1, 3.516e-1, 5.248e-1, 7.472e-1, 1.025]
C_LIGHT = 3e10

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
# НАСТРОЙКИ СИМУЛЯЦИИ
# =====================================================================
N = 50
R = 0.35
h = R / N
tau = 1e-6
T0 = 8000.0
p_param = 10.0
T_w = 1800.0
r = [i * h for i in range(N + 1)]
T = [T0 + (T_w - T0) * (ri / R)**p_param for ri in r]

# Функция постоянного тока
def I_const(t):
    I_target = 1000.0
    # t_ramp = 1e-6  # Разгоняем плавно за 100 мкс
    # if t < t_ramp:
    #     return I_target * (t / t_ramp)
    return I_target

# =====================================================================
# ГЛАВНЫЙ ЦИКЛ
# =====================================================================
print("Запуск симуляции с током I = const = 150 А...")
t = 0.0
MAX_TIME = 20000e-6      # УВЕЛИЧИЛИ ВРЕМЯ до 20 мс
TOLERANCE = 1e-4

history_t = []
history_T_center = []
is_steady = False

while t <= MAX_TIME:
    I_t = I_const(t)

    sigma = [get_sigma(T[i]) for i in range(N + 1)]
    lam = [get_lambda(T[i]) for i in range(N + 1)]
    c_T = [get_c(T[i]) for i in range(N + 1)]
    k_coef = [get_k(T[i]) for i in range(N + 1)]
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

    T_new = progonka(At, Bt, Ct, Ft)

    max_diff = max(abs(T_new[i] - T[i]) for i in range(N+1))

    # Сохраняем каждую 10-ю точку для ускорения работы
    if int(t / tau) % 10 == 0:
        history_t.append(t * 1e6)
        history_T_center.append(T_new[0])

    T = T_new
    t += tau

    if max_diff < TOLERANCE and t > 200e-6:
        is_steady = True
        break

# =====================================================================
# ВЫЧИСЛЕНИЕ ЭНЕРГЕТИЧЕСКОГО БАЛАНСА
# =====================================================================
P_in = (E * I_t) / (2.0 * math.pi)
integral_q = sum(0.5 * (q[i]*r[i] + q[i+1]*r[i+1]) * h for i in range(N))
dT_dr_R = (3.0*T[N] - 4.0*T[N-1] + T[N-2]) / (2.0 * h)
F_N = -lam[N] * dT_dr_R
P_heat = R * F_N
P_out = integral_q + P_heat
epsilon = abs(P_in - P_out) / P_in

# =====================================================================
# ВЫВОД В КОНСОЛЬ
# =====================================================================
print("-" * 60)
if is_steady:
    print(f"[УСПЕХ] Стационарный режим достигнут на {t*1e6:.1f} мкс!")
else:
    print(f"[ПРЕДУПРЕЖДЕНИЕ] Лимит времени исчерпан. Система еще в динамике.")
print("-" * 60)
print(f"Ток:                        {I_t:.2f} А")
print(f"Температура в центре (T_0): {T[0]:.1f} К")
print("-" * 60)
print(f"ЛЕВАЯ ЧАСТЬ (Подвод):       {P_in:.4f} Вт/см")
print(f"ПРАВАЯ ЧАСТЬ (Отвод):       {P_out:.4f} Вт/см")
print("-" * 60)
print(f"ТОЧНОСТЬ ВЫПОЛНЕНИЯ (ε):    {epsilon:.2e} (Требовалось ~10^-2)")

# =====================================================================
# ГРАФИК
# =====================================================================
plt.figure(figsize=(10, 5))
plt.plot(history_t, history_T_center, color='firebrick', linewidth=2)
plt.title("Установление стационарного режима температуры", fontsize=12)
plt.xlabel("Время t, мкс")
plt.ylabel("Температура в центре T(0), К")
plt.grid(True, linestyle='--')

plt.scatter([history_t[-1]], [history_T_center[-1]], color='black', zorder=5)
plt.annotate(f'Стационар\n{T[0]:.0f} К', xy=(history_t[-1], history_T_center[-1]),
             xytext=(-60, -30), textcoords='offset points', arrowprops=dict(arrowstyle="->"))

plt.tight_layout()
plt.show()
