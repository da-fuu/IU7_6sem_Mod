import math
import matplotlib.pyplot as plt

VARIANT = 1

VARIANTS_DATA = {
    1: {
        "N": 50, "tau": 1e-6, "EPSILON": 1e-4, "MAX_ITER": 50,
        "TABLE_k": [8.2e-3, 2.768e-2, 6.56e-2, 1.281e-1, 2.214e-1, 3.516e-1, 5.248e-1, 7.472e-1, 1.025]
    },
    2: {
        "N": 150, "tau": 1e-7, "EPSILON": 1e-4, "MAX_ITER": 150,
        "TABLE_k": [1.600, 5.400, 1.280e+1, 2.500e+1, 4.320e+1, 6.860e+1, 1.024e+2, 1.458e+2, 2.000e+2]
    }
}

cfg = VARIANTS_DATA[VARIANT]

T0 = 8000.0
T_w = 1800.0
p_param = 2.0
R = 0.35
I_max = 1000.0
t_max = 80e-6
t_end = 200e-6
C_LIGHT = 3e10

N = cfg["N"]
tau = cfg["tau"]
h = R / N

PRINT_TIMES = [10e-6, 30e-6, 50e-6, 80e-6, 100e-6, 120e-6, 150e-6, 200e-6]

EPSILON = cfg["EPSILON"]
MAX_ITER = cfg["MAX_ITER"]

TABLE_T = [2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 11000, 12000]
TABLE_SIGMA = [0.309e-3, 0.309e-2, 0.309e-1, 0.270, 2.05, 6.06, 12.0, 19.9, 29.6, 41.1, 54.1]
TABLE_LAMBDA = [0.381e-3, 0.381e-3, 0.381e-3, 0.448e-3, 0.577e-3, 0.733e-3, 0.131e-2, 0.218e-2, 0.358e-2, 0.562e-2, 0.832e-2]
TABLE_C = [1.90e-3, 1.90e-3, 0.95e-3, 0.75e-3, 0.64e-3, 0.61e-3, 0.66e-3, 0.66e-3, 1.15e-3, 1.79e-3, 2.02e-3]

TABLE_T_k = [2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
TABLE_k = cfg["TABLE_k"]


def get_index(val, x_array):
    if val <= x_array[0]:
        return 0
    if val >= x_array[-1]:
        return len(x_array) - 2
    for i in range(len(x_array) - 1):
        if x_array[i] <= val <= x_array[i+1]:
            return i
    return 0

def interpolate(val, x_array, y_array):
    i = get_index(val, x_array)
    x0, x1 = x_array[i], x_array[i+1]
    y0, y1 = y_array[i], y_array[i+1]
    return y0 + (y1 - y0) * (val - x0) / (x1 - x0)

def interpolate_log(val, x_array, y_array):
    i = get_index(val, x_array)
    x0, x1 = x_array[i], x_array[i+1]
    y0, y1 = math.log(y_array[i]), math.log(y_array[i+1])
    y_ln = y0 + (y1 - y0) * (val - x0) / (x1 - x0)
    return math.exp(y_ln)

def get_sigma(T): return interpolate_log(T, TABLE_T, TABLE_SIGMA)
def get_lambda(T): return interpolate_log(T, TABLE_T, TABLE_LAMBDA)
def get_c(T): return interpolate(T, TABLE_T, TABLE_C)
def get_k(T): return interpolate_log(T, TABLE_T_k, TABLE_k)

def I(t):
    return I_max * (t / t_max) * math.exp(-(t / t_max - 1))

def u_p(T):
    T_safe = max(T, 1.0)
    power = 4.799e4 / T_safe
    if power > 700:
        return 0.0
    return (3.084e-4) / math.expm1(power)

def progonka(A, B, C, F):
    n = len(B)
    alpha, beta, x = [0.0] * n, [0.0] * n, [0.0] * n

    alpha[1] = C[0] / B[0]
    beta[1] = F[0] / B[0]

    for i in range(1, n - 1):
        denom = B[i] - A[i] * alpha[i]
        alpha[i+1] = C[i] / denom
        beta[i+1] = (A[i] * beta[i] + F[i]) / denom

    denom = B[n-1] - A[n-1] * alpha[n-1]
    x[n-1] = (A[n-1] * beta[n-1] + F[n-1]) / denom
    for i in range(n - 2, -1, -1):
        x[i] = alpha[i+1] * x[i+1] + beta[i+1]
    return x

def compute_coeffs(T_dist):
    sigma = [get_sigma(T_dist[i]) for i in range(N + 1)]
    lam = [get_lambda(T_dist[i]) for i in range(N + 1)]
    c_T = [get_c(T_dist[i]) for i in range(N + 1)]
    k_coef = [get_k(T_dist[i]) for i in range(N + 1)]
    up = [u_p(T_dist[i]) for i in range(N + 1)]
    return sigma, lam, c_T, k_coef, up

def solve_radiation(r, h, k_coef, up):
    Au, Bu, Cu, Fu = [0.0] * (N + 1), [0.0] * (N + 1), [0.0] * (N + 1), [0.0] * (N + 1)
    for i in range(1, N):
        D_minus = 1.0 / (3.0 * 0.5 * (k_coef[i-1] + k_coef[i]))
        D_plus  = 1.0 / (3.0 * 0.5 * (k_coef[i] + k_coef[i+1]))
        Au[i] = (r[i] - 0.5 * h) * D_minus / (h**2)
        Cu[i] = (r[i] + 0.5 * h) * D_plus / (h**2)
        Bu[i] = Au[i] + Cu[i] + k_coef[i] * r[i]
        Fu[i] = k_coef[i] * up[i] * r[i]

    Bu[0], Cu[0], Au[0], Fu[0] = 1.0, 1.0, 0.0, 0.0
    D_N = 1.0 / (3.0 * k_coef[N])
    Au[N], Bu[N], Cu[N], Fu[N] = D_N, D_N + 0.39 * h, 0.0, 0.0

    u_rad = progonka(Au, Bu, Cu, Fu)
    q = [C_LIGHT * k_coef[i] * (up[i] - u_rad[i]) for i in range(N + 1)]
    return q

def solve_temperature_step(T_prev, T_guess, r, h, tau, sigma, lam, c_T, E, q):
    At, Bt, Ct, Ft = [0.0] * (N + 1), [0.0] * (N + 1), [0.0] * (N + 1), [0.0] * (N + 1)
    for i in range(1, N):
        lam_minus = 0.5 * (lam[i-1] + lam[i])
        lam_plus  = 0.5 * (lam[i] + lam[i+1])
        At[i] = (r[i] - 0.5 * h) * lam_minus / (h**2)
        Ct[i] = (r[i] + 0.5 * h) * lam_plus / (h**2)
        Bt[i] = At[i] + Ct[i] + (c_T[i] * r[i]) / tau
        Ft[i] = (c_T[i] * r[i] / tau) * T_prev[i] + (sigma[i] * E**2 - q[i]) * r[i]

    Bt[0], Ct[0], At[0], Ft[0] = 1.0, 1.0, 0.0, 0.0
    At[N], Bt[N], Ct[N], Ft[N] = 0.0, 1.0, 0.0, T_w

    return progonka(At, Bt, Ct, Ft)


r = [i * h for i in range(N + 1)]
T = [T0 + (T_w - T0) * (ri / R)**p_param for ri in r]

t = 0.0
results = {0.0: list(T)}
history_t, history_I = [], []

global_max_T = max(T)
while t <= t_end + 1e-9:
    I_t = I(t)
    history_t.append(t * 1e6)
    history_I.append(I_t)

    T_guess = list(T)

    for iteration in range(MAX_ITER):
        sigma, lam, c_T, k_coef, up = compute_coeffs(T_guess)

        integral_sigma = sum(0.5 * (sigma[i] * r[i] + sigma[i + 1] * r[i + 1]) * h for i in range(N))
        E = I_t / (2.0 * math.pi * integral_sigma) if integral_sigma > 0 else 0.0

        q = solve_radiation(r, h, k_coef, up)
        T_new_calc = solve_temperature_step(T, T_guess, r, h, tau, sigma, lam, c_T, E, q)

        max_diff = max(abs(T_new_calc[i] - T_guess[i]) for i in range(N + 1))
        T_guess = list(T_new_calc)

        if max_diff < EPSILON:
            break

    T = list(T_guess)
    t += tau

    global_max_T = max(global_max_T, max(T))

    for pt in PRINT_TIMES:
        if abs(t - pt) < (tau * 0.5):
            results[pt] = list(T)

print(f"Максимальная достигнутая температура за весь расчет: {global_max_T:.2f} К")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

for pt in [0.0] + PRINT_TIMES:
    if pt in results:
        t_mks = int(pt * 1e6)
        linestyle = '-' if t_mks <= 80 else '--'
        linewidth = 2 if t_mks == 80 else 1.5

        ax1.plot(r, results[pt], label=f't = {t_mks} мкс', linestyle=linestyle, linewidth=linewidth)

ax1.set_title("T(r)", fontsize=12)
ax1.grid(True, linestyle=':', alpha=0.7)
ax1.legend(loc='upper right', fontsize=9)

ax2.plot(history_t, history_I, color='blue', linewidth=2)
for pt in PRINT_TIMES:
    t_mks = pt * 1e6
    i_val = I(pt)
    ax2.plot(t_mks, i_val, marker='o', color='red')

ax2.set_title("I(t)", fontsize=12)
ax2.grid(True, linestyle=':', alpha=0.7)

plt.tight_layout()
plt.show()
