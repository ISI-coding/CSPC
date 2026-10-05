import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

data = np.loadtxt("kinetics.csv", delimiter=",", skiprows=1)
t = data[:, 0]
C = data[:, 1]
C0 = C[0]

def total_error(k):
    predicted = C0 * np.exp(-k*t)
    error = np.sum((C - predicted)**2)
    return error

result = minimize(total_error, x0=[0.5], method="SLSQP", bounds=[(0, 5)])
k_fit = result.x[0]
print("Fitted k =", k_fit)

C_fit = C0 * np.exp(-k_fit * t)
plt.scatter(t, C, label="Measured data")
plt.plot(t, C_fit, label="Fitted curve")
plt.xlabel("Time")
plt.ylabel("Concentration")
plt.legend()
plt.savefig("kinetics.png")
