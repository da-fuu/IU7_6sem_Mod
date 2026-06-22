from copy import deepcopy
import math

VARIANT = 1

# Геометрические размеры пластины
a = 10.0            # Длина пластины по X, см
b = 10.0            # Ширина пластины по Z, см
delta = 0.5         # Толщина пластины, см

# Параметры теплоотдачи и окружения
u0 = 300.0          # Температура окружающей среды (воздуха), К
alpha0 = 0.05       # Коэф. теплоотдачи плоских граней пластины, Вт/(см² К)

# Коэффициенты теплоотдачи на торцах
alpha1 = 0.08       # Для границы 4 (левая, x = 0) в Варианте 2
alpha2 = 0.08       # Для границы 2 (правая, x = a)
alpha3 = 0.08       # Для границы 3 (нижняя, z = 0)
alpha4 = 0.08       # Для границы 1 (верхняя, z = b)

# Тепловой поток на левой границе (только для Варианта 1)
F0 = 30.0           # Вт/см²

LASER_SOURCES = [
    {"f0": 150.0, "beta": 0.15, "x0": 5.0, "z0": 5.0},
    {"f0": 150.0, "beta": 0.5, "x0": 3.0, "z0": 9.0},
    # {"f0": 150.0, "beta": 0.15, "x0": 9.0, "z0": 1.0},

]

Nx = 50             # Шагов по оси X
Nz = 50             # Шагов по оси Z
tau = 0.05          # Шаг виртуального времени, сек
EPSILON = 1e-4      # Критерий выхода на стационарный режим

hx = a / Nx         # Шаг по пространству X, см
hz = b / Nz         # Шаг по пространству Z, см

def get_lambda(u):
    """Вычисляет коэффициент теплопроводности lambda(u) по формуле (1)"""
    a1 = 0.0134
    b1 = 1.0
    c1 = 4.35e-4
    m1 = 1
    return a1 * (b1 + c1 * (u ** m1))

def get_f_source(x, z):
    """Вычисляет плотность нагрева от лазерных лучей в точке (x, z)"""
    val = 0.0
    for src in LASER_SOURCES:
        power = -src["beta"] * ((x - src["x0"])**2 + (z - src["z0"])**2)
        val += src["f0"] * math.exp(power)
    return val

def progonka(A, B, C, F):
    """Решает трехдиагональную систему уравнений: A_i*X_i-1 - B_i*X_i + C_i*X_i+1 = -F_i"""
    n = len(B)
    alpha = [0.0] * n
    beta = [0.0] * n
    x = [0.0] * n

    alpha[1] = C[0] / B[0]
    beta[1] = F[0] / B[0]

    for i in range(1, n - 1):
        denom = B[i] - A[i] * alpha[i]
        if denom == 0: denom = 1e-15
        alpha[i+1] = C[i] / denom
        beta[i+1] = (A[i] * beta[i] + F[i]) / denom

    denom = B[n-1] - A[n-1] * alpha[n-1]
    if denom == 0: denom = 1e-15
    x[n-1] = (A[n-1] * beta[n-1] + F[n-1]) / denom

    for i in range(n - 2, -1, -1):
        x[i] = alpha[i+1] * x[i+1] + beta[i+1]
    return x

# 4. ЛОКАЛЬНО-ОДНОМЕРНЫЙ МЕТОД

def x_sweep(u, u_star):
    """Прогонка по строкам"""
    for k in range(Nz + 1):
        A = [0.0] * (Nx + 1)
        B = [0.0] * (Nx + 1)
        C = [0.0] * (Nx + 1)
        F = [0.0] * (Nx + 1)

        # Внутренние узлы строки
        for i in range(1, Nx):
            lam_minus = 0.5 * (get_lambda(u[i-1][k]) + get_lambda(u[i][k]))
            lam_plus  = 0.5 * (get_lambda(u[i][k]) + get_lambda(u[i+1][k]))

            A[i] = (tau / hx**2) * lam_minus
            C[i] = (tau / hx**2) * lam_plus
            B[i] = A[i] + C[i] + 1.0 + (tau * alpha0) / delta
            F[i] = u[i][k] + (tau * alpha0 / delta) * u0 + (tau / 2.0) * get_f_source(i * hx, k * hz)

        # Левая граница
        if VARIANT == 1:
            # II род:
            lam_0 = get_lambda(u[0][k])
            A[0] = 0.0
            B[0] = lam_0
            C[0] = lam_0
            F[0] = F0 * hx
        else:
            # III род:
            lam_0 = get_lambda(u[0][k])
            A[0] = 0.0
            B[0] = lam_0 + alpha1 * hx
            C[0] = lam_0
            F[0] = alpha1 * hx * u0

        # Правая граница
        lam_Nx = get_lambda(u[Nx][k])
        A[Nx] = lam_Nx
        B[Nx] = lam_Nx + alpha2 * hx
        C[Nx] = 0.0
        F[Nx] = alpha2 * hx * u0

        if VARIANT == 3:
            A[0] = A[Nx] = 0.0
            B[0] = B[Nx] = 1.0
            C[0] = C[Nx] = 0.0
            F[0] = F[Nx] = u0

        sol = progonka(A, B, C, F)
        for i in range(Nx + 1):
            u_star[i][k] = sol[i]

def z_sweep(u_star, u_new):
    """Прогонка по столбцам"""
    for i in range(Nx + 1):
        A = [0.0] * (Nz + 1)
        B = [0.0] * (Nz + 1)
        C = [0.0] * (Nz + 1)
        F = [0.0] * (Nz + 1)

        # Внутренние узлы столбца
        for k in range(1, Nz):
            lam_minus = 0.5 * (get_lambda(u_star[i][k-1]) + get_lambda(u_star[i][k]))
            lam_plus  = 0.5 * (get_lambda(u_star[i][k]) + get_lambda(u_star[i][k+1]))

            A[k] = (tau / hz**2) * lam_minus
            C[k] = (tau / hz**2) * lam_plus
            B[k] = A[k] + C[k] + 1.0 + (tau * alpha0) / delta
            F[k] = u_star[i][k] + (tau * alpha0 / delta) * u0 + (tau / 2.0) * get_f_source(i * hx, k * hz)

        # Нижняя граница
        lam_0 = get_lambda(u_star[i][0])
        A[0] = 0.0
        B[0] = lam_0 + alpha3 * hz
        C[0] = lam_0
        F[0] = alpha3 * hz * u0

        # Верхняя граница
        lam_Nz = get_lambda(u_star[i][Nz])
        A[Nz] = lam_Nz
        B[Nz] = lam_Nz + alpha4 * hz
        C[Nz] = 0.0
        F[Nz] = alpha4 * hz * u0

        if VARIANT == 3:
            A[0] = A[Nz] = 0.0
            B[0] = B[Nz] = 1.0
            C[0] = C[Nz] = 0.0
            F[0] = F[Nz] = u0

        sol = progonka(A, B, C, F)
        for k in range(Nz + 1):
            u_new[i][k] = sol[k]

u = [[u0] * (Nz + 1) for _ in range(Nx + 1)]
u_star = [[0.0] * (Nz + 1) for _ in range(Nx + 1)]
u_new = [[0.0] * (Nz + 1) for _ in range(Nx + 1)]

t = 0.0
step = 0
max_steps = 10000

print(f"Вариант: {VARIANT}")
print(f"Сетка: {Nx}x{Nz}, Шаг времени tau: {tau} сек")
print("-" * 50)

while step < max_steps:
    x_sweep(u, u_star)
    z_sweep(u_star, u_new)

    max_diff = 0.0
    for i in range(Nx + 1):
        if max_diff > EPSILON:
            break
        for k in range(Nz + 1):
            diff = (abs(u_new[i][k] - u[i][k])) / u_new[i][k]
            if diff > max_diff:
                max_diff = diff
                if max_diff > EPSILON:
                    break

    u = deepcopy(u_new)
    t += tau
    step += 1

    if step % 50 == 0 or step == 1:
        print(f"Шаг: {step:<4} | Время: {t:<6.2f} сек | Изменение T_max: {max_diff:.3e} К")

    if max_diff < EPSILON:
        print("-" * 50)
        print(f"СТАЦИОНАРНЫЙ РЕЖИМ ДОСТИГНУТ!")
        print(f"Всего шагов: {step} | Время симуляции: {t:.2f} сек")
        print(f"Температура в центре пластины: {u[Nx//2][Nz//2]:.2f} К")
        break
else:
    print("Превышен лимит шагов по времени!")


print("\n--- ПРОВЕРКА БАЛАНСА МОЩНОСТИ ---")

P_in = 0.0   # Подводимая мощность (Вт)
P_out = 0.0  # Отводимая мощность (Вт)

# Вспомогательная функция для численного интегрирования (метод трапеций)
# Краевые узлы "отвечают" за половину объема ячейки
def get_weight(idx, max_idx):
    return 0.5 if (idx == 0 or idx == max_idx) else 1.0

# 1. Считаем внутренний объемный нагрев от лазера и охлаждение с плоскостей
for i in range(Nx + 1):
    for k in range(Nz + 1):
        w_x = get_weight(i, Nx)
        w_z = get_weight(k, Nz)

        # Элементарный объем и площадь для данного узла
        dV = (hx * w_x) * (hz * w_z) * delta
        dS_flat = (hx * w_x) * (hz * w_z)

        # Подвод тепла: Лазер (интеграл по объему)
        f_val = get_f_source(i * hx, k * hz)
        P_in += f_val * dV

        # Отвод тепла: Конвекция с верхней и нижней плоскостей пластины
        # Умножаем на 2, так как поверхностей две (верх и низ)
        P_out += 2.0 * alpha0 * (u[i][k] - u0) * dS_flat

# 2. Считаем потоки на торцах пластины (границах 1-4)
for k in range(Nz + 1):
    w_z = get_weight(k, Nz)
    dS_x = (hz * w_z) * delta  # Элементарная площадь на торцах x=0 и x=a

    # Левая граница (x = 0)
    if VARIANT == 1:
        P_in += F0 * dS_x  # II род: постоянный тепловой поток внутрь
    elif VARIANT == 2:
        P_out += alpha1 * (u[0][k] - u0) * dS_x  # III род: конвекция
    elif VARIANT == 3:
        # I род: тепло уходит за счет теплопроводности (Закон Фурье)
        lam = get_lambda(u[0][k])
        grad = (u[1][k] - u[0][k]) / hx
        P_out += lam * grad * dS_x

    # Правая граница (x = a)
    if VARIANT in [1, 2]:
        P_out += alpha2 * (u[Nx][k] - u0) * dS_x
    elif VARIANT == 3:
        lam = get_lambda(u[Nx][k])
        grad = (u[Nx-1][k] - u[Nx][k]) / hx
        P_out += lam * grad * dS_x

for i in range(Nx + 1):
    w_x = get_weight(i, Nx)
    dS_z = (hx * w_x) * delta  # Элементарная площадь на торцах z=0 и z=b

    # Нижняя граница (z = 0)
    if VARIANT in [1, 2]:
        P_out += alpha3 * (u[i][0] - u0) * dS_z
    elif VARIANT == 3:
        lam = get_lambda(u[i][0])
        grad = (u[i][1] - u[i][0]) / hz
        P_out += lam * grad * dS_z

    # Верхняя граница (z = b)
    if VARIANT in [1, 2]:
        P_out += alpha4 * (u[i][Nz] - u0) * dS_z
    elif VARIANT == 3:
        lam = get_lambda(u[i][Nz])
        grad = (u[i][Nz-1] - u[i][Nz]) / hz
        P_out += lam * grad * dS_z

# 3. Вывод результатов
print(f"Подводимая мощность (P_in):  {P_in:.2f} Вт")
print(f"Отводимая мощность (P_out): {P_out:.2f} Вт")

balance_error = abs(P_in - P_out) / P_in * 100 if P_in != 0 else 0
print(f"Относительная невязка баланса: {balance_error:.3f}%")
print("-" * 50)




import numpy as np
import matplotlib.pyplot as plt

r_x = [i * hx for i in range(Nx + 1)]
r_z = [k * hz for k in range(Nz + 1)]

X, Z = np.meshgrid(r_x, r_z)
U = np.array(u).T

fig = plt.figure(figsize=(15, 6))

ax3d = fig.add_subplot(121, projection='3d')
surf = ax3d.plot_surface(X, Z, U, cmap='hot', edgecolor='none')
ax3d.set_title(f"3D тепловая карта пластины (Вариант {VARIANT})")
ax3d.set_xlabel("Ось X, см")
ax3d.set_ylabel("Ось Z, см")
ax3d.set_zlabel("Температура u, К")
fig.colorbar(surf, ax=ax3d, shrink=0.5, aspect=5)

ax2d = fig.add_subplot(122)
for p in range(1, 4):
    ax2d.plot(r_x, [u[i][Nz * p // 4] for i in range(Nx + 1)], linewidth=2, label=f'Срез вдоль X при z={r_z[Nz * p // 4]:.1f} см')

for p in range(1, 4):
    ax2d.plot(r_z, [u[Nx * p // 4][k] for k in range(Nz + 1)], '--', linewidth=2, label=f'Срез вдоль Z при x={r_x[Nx * p // 4]:.1f} см')

ax2d.set_title("Одномерные температурные срезы пластины")
ax2d.set_xlabel("Координата, см")
ax2d.set_ylabel("Температура u, К")
ax2d.grid(True, linestyle=':', alpha=0.8)
ax2d.legend()

plt.tight_layout()
plt.show()
