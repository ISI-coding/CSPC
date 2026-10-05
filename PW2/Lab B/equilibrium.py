import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import newton, minimize

K = 15.6
a = b = 1.0

def k_imbalance(x):
    return (2*x)**2 / ((a-x)*(b-x)) - K

x_newton = newton(k_imbalance, x0 = 0.5)
print("Newton x =", x_newton)

result = minimize(lambda x: k_imbalance(x[0])**2, x0=[0.5], method="SLSQP", bounds=[(0, 0.999)])
x_slsqp = result.x[0]
print("SLSQP x =", x_slsqp)

x_eq = x_newton
H2_eq = a - x_eq
I2_eq = b - x_eq
HI_eq = 2*x_eq

print("Equilibrium H2 =", H2_eq)
print("Equilibrium I2 =", I2_eq)
print("Equilibrium HI =", HI_eq)

x_values = np.linspace(0, 0.999, 200)
H2 = a - x_values
I2 = b - x_values
HI = 2*x_values

plt.plot(x_values, H2, label="H2")
plt.plot(x_values, I2, label="I2")
plt.plot(x_values, HI, label="HI")
plt.axvline(x_eq, linestyle="--", label="Equilibrium")
plt.xlabel("Extent x")
plt.ylabel("Amount (mol)")
plt.legend()
plt.savefig("equilibrium.png")