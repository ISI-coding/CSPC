from time import perf_counter
from decay import simulate, simulate_loop
N0 = 200000
lam = 0.4
start = perf_counter()
simulate_loop(N0, lam)
loop_time = perf_counter() - start
start = perf_counter()
simulate(N0, lam)
numpy_time = perf_counter() - start 
speedup = loop_time / numpy_time
print(f"Loop time : {loop_time:.6f} s")
print(f"NumPy time : {numpy_time:.6f} s")
print(f"NumPy is {speedup:.2f} times faster")
