import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid

data = np.loadtxt("freefall.csv", delimiter=",", skiprows=1)
t = data[:, 0]
y = data[:, 1]

v = np.gradient(y, t)
a = np.gradient(v, t)

print("Mean acceleration:", a.mean())
print("Acceleration std:", a.std())

v_recovered = cumulative_trapezoid(a, t, initial=0) + v[0]
y_recovered = cumulative_trapezoid(v_recovered, t, initial=0) + y[0]
max_difference = np.max(np.abs(y_recovered - y))

print(f"Largest position difference: {max_difference:.3f}m")

fig, axes = plt.subplots(3, 1, sharex=True)

axes[0].plot(t, y)
axes[0].set_ylabel("Position (m)")
axes[0].set_title("Motion from free fall data")

axes[1].plot(t, v)
axes[1].set_ylabel("Velocity (m/s)")

axes[2].plot(t, a)
axes[2].axhline(-9.81, linestyle="--", label="g = -9.81")
axes[2].set_ylabel("Acceleration (m/s^2)")
axes[2].set_xlabel("Time (s)")
axes[2].legend()

plt.tight_layout()
plt.savefig("motion.png")

#Bonus Part
trajectory = np.loadtxt("trajectory.csv", delimiter=",", skiprows=1)
t2 = trajectory[:, 0]
x = trajectory[:, 1]
y2 = trajectory[:, 2]
vx = np.gradient(x, t2)
vy = np.gradient(y2, t2)

speed = np.sqrt(vx**2 + vy**2)
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(x, y2)
plt.xlabel("x")
plt.ylabel("y")
plt.title("Trajectory")
plt.axis("equal")

plt.subplot(1,2,2)
plt.plot(t2, speed)
plt.xlabel("Time(s)")
plt.ylabel("Speed(m/s)")
plt.title("Speed vs Time")
plt.tight_layout()
plt.savefig("trajectory_bonus.png")