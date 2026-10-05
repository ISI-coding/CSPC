import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt("titration.csv", delimiter=",", skiprows=1)
V = data[:, 0]
pH = data[:, 1]

slope = np.gradient(pH, V)
index = np.argmax(slope)
V_eq = V[index]
print("Equivalence point =", V_eq, "ml")

fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].plot(V, pH, label="pH")
ax[0].axvline(V_eq, linestyle="--", label="Equivalence point")
ax[0].set_xlabel("Volume of base (ml)")
ax[0].set_ylabel("pH")
ax[0].legend()

ax[1].plot(V, slope, label="slope")
ax[1].axvline(V_eq, linestyle="--", label="Equivalence point")
ax[1].set_xlabel("Volume of base (ml)")
ax[1].set_ylabel("dpH/dV")
ax[1].legend()

plt.tight_layout()
plt.savefig("titration.png")