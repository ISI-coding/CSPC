
# CSPC - Computer Science for Physics and Chemistry
My coursework repository. Each practical is under PW<n>/Lab <X>/.
## Setup
Create the environment for a given lab:
conda env create -f PW<n>/Lab\ <X>/environment.yml
conda activate cspc

## PW1 - Lab A: Reproducible Foundations 

**What I built:**

I built radioactive decay simulation
Added tests and compared Python loop with NumPy implementation

**Speed comparison (loop vs NumPy):**

loop:3.445878s
NumPy:0.000309s
speed-up:11135.39x faster 

**Tests:** Yes, all 3 tests passed 

**Conclusion:**
Using Git and Github made the project easy to share and reproducible. NumPy implementation was much faster than Python loop, tests verified that the simulation behaved correctly.


## PW1 - Lab B: Radioactive decay
Observed radioactive decay data was compared with the analytical exponential decay law
The observed data followed the same overall exponential decay trend as the analytical curve, 
although the measured points have small imbalance

The snakemake pipeline automatically rebuilds figure.png from the csv data whenever the input changes 
and skips rebuilding when the files are already up to date 


## PW2 - Lab A: 

**Mean acceleration:** -8.58 m/s^2
**Acceleration standard deviation:** 28.72 m/s^2

**Noise observation:**
The acceleration is much noisier than the position because differentiating noisy measurements
increases the noise, and taking two derivatives makes the effect even stronger

**Integration result:**
Integrating the noisy acceleration recovered the original position within 0.785 m
showing that integration suppresses measurement noise 

**Bonus:**
In bonus part we read data from trajectory.csv with np.loadtxt as in main part of lab
Computed the velocity components vx and vy using np.gradient
Calculated the speed using given formula
Created a figure with two subplots:
    -The trajectory(x and y)
    -The speed as a function of time 
By using plt.axis("equal") the trajectory keeps its correct shape


## PW2 - Lab B:

**Part 2 - Three routes to a minimum**

For (f(x) = (x-3)^2 + 1), all three methods converged to x = 3(approximately)
For(g(x) = x^4-3x^2 + x + 5), the result depended on the starting point. From(x0 = 0), gradient descent and SLSQP reached a local minimum, while Newton's method reached another stationary point. From(x0 = 2), the methods reached the other local minimum. A smaller learning rate made gradient descent slower.

**Part 3 - Reaction rate**

The measured concentration data were fitted with the first-order model

C(t) = C0e^{-kt}

SLSQP gave a fitted rate constant of approximately (k = 0.25). The fitted curve follows the measured data reasonably well.

**Part 4 - Chemical equilibrium**
For
H2 + I2 -> 2HI
with (K = 15.6), Newton's method and SLSQP both gave (x = 0.66(approximately))
The equilibrium amount are approximately:
    H2 = 0.34 mol
    I2 = 0.34 mol
    HI = 1.33 mol

**Part 5 - Titration(bonus)**
The maximum pH slope occured at approximately 50 ml, giving the equivalence point.

**Conclusion:**
This lab showed how different optimisation methods can be used to solve mathematical and chemistry problems. We also observed that the starting point and learning rate can affect the result