import numpy as np
from scipy.optimize import newton, minimize

def f(x):   return (x-3)**2 + 1
def df(x):  return 2*(x-3)
def d2f(x): return 2.0

print("***Part 2A***")
x0 = 0
for lr in [0.1, 0.01, 0.001]:
    x = x0
    for i in range(10000):
        x = x - lr * df(x)
    print("")
    print("Gradient descent")
    print("learning rate =", lr)
    print("x =", x)
    print("f(x) =", f(x))
    print("iterations =", i + 1)

x_newton = newton(df, x0, fprime=d2f)
print("")
print("Newton")
print("x =", x_newton)
print("f(x) =", f(x_newton))

result = minimize(f, x0, method="SLSQP")
print("")
print("SLSQP")
print("x =", result.x[0])
print("f(x) =", result.fun)

print("")
def g(x):   return x**4 - 3*x**2 + x + 5
def dg(x):  return 4*x**3 - 6*x + 1
def d2g(x): return 12*x**2 - 6


print("***Part 2B (x0 = 0)***")
x0 = 0
for lr in [0.1, 0.01, 0.001]:
    x = x0
    for i in range(10000):
        x = x - lr * dg(x)
    print("")
    print("Gradient descent")
    print("learning rate =", lr)
    print("x =", x)
    print("g(x) =", g(x))
    print("iterations =", i + 1)

x_newton = newton(dg, x0, fprime=d2g)
print("")
print("Newton")
print("x =", x_newton)
print("g(x) =", g(x_newton))
print("g''(x) =", d2g(x_newton))
if d2g(x_newton) > 0:
    print("This is minimum")
elif d2g(x_newton) < 0:
    print("This is maximum")
else:
    print("Second derivative is zero")

result = minimize(g, x0, method="SLSQP")
print("")
print("SLSQP")
print("x =", result.x[0])
print("g(x) =", result.fun)
print("")


print("***Part 2B (x0 = 2)***")
x0 = 2
for lr in [0.1, 0.01, 0.001]:
    x = x0
    for i in range(10000):
        x = x - lr * dg(x)
    print("")
    print("Gradient descent")
    print("learning rate =", lr)
    print("x =", x)
    print("g(x) =", g(x))
    print("iterations =", i + 1)

x_newton = newton(dg, x0, fprime=d2g)
print("")
print("Newton")
print("x =", x_newton)
print("g(x) =", g(x_newton))
print("g''(x) =", d2g(x_newton))
if d2g(x_newton) > 0:
    print("This is minimum")
elif d2g(x_newton) < 0:
    print("This is maximum")
else:
    print("Second derivative is zero")

result = minimize(g, x0, method="SLSQP")
print("")
print("SLSQP")
print("x =", result.x[0])
print("g(x) =", result.fun)