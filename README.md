
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

 ## PW3 - Statistical analysis

**1.Heart disease dataset**
The dataset contains 303 patients and 14 columns. I used pandas to examine the data and summarize age,
cholesterol, resting blood pressure and maximum heart rate.

Normality
I used histograms and the Shapiro Wilk test to investigate normality. A p-value below 0.05 indicates evidence against
normal distribution. The histograms were used alongside the test results to describe each variable's distribution.

Heart rate comparison
I compared maximum heart rate (thalach) between disease and healthy groups using the Mann Whitney U test because the healthy group's
normality test was significant.

Disease: mean = 139.26, 95% CI[135.47, 143.05].
Healthy: mean = 158.38, 95% CI[155.42, 161.34].
U = 5806.5, p = 1.86 * 10^(-13)
The difference is statistically significant; patients with heart disease had a lower mean maximum heart rate.

Age and Heart rate
Pearson correlation was r = -0.3938, p = 1.11 * 10^(-12). This indicates a moderate negative relationship: maximum heart rate tends to decrease with age.

Normality Results
I used Shapiro Wilk test with a significance level 0f 0.05, supported by histograms.
Age: W = 0.9865, p = 0.00607 - evidence against normality.
Cholesterol: W = 0.9473, p = 5.91 * 10^(-9) - evidence against normality.
Resting blood pressure: W = 0.9665, p = 1.80 * 10^(-6) - evidence against normality.
Maximum heart rate: W = 0.9765, p = 7.00 * 10^(-5) - evidence against normality.

Since all p-values are below 0.05, i rejected the null hypothesis of normality for all four variables. These results helped guide my choice of statistical methods.

**2. Chemical Exposure Dataset**
The dataset contains 1000 patients.
Initial Associations

Benzene: r = 0.3847, p = 1.28 * 10^(-36)
Cadmium: r = 0.88, p < 0.001
Initially cadmium appeared much more strongly associated with malignancy.

Controlling for Pollution
I restricted the analysis to 201 patients with pollution indices between 40 and 60.

Benzene: r = 0.6979, p = 1.15 * 10^(-30)
Cadmium: r = 0.3132, p = 5.99 * 10^(-6)
Cadmium's association became much weaker, while benzene's became stronger. This suggests that pollution may explain part of the original cadmium association. Benzene appears more likely to be the casual chemical, although further analysis controlling for age and pollution is needed to establish causation.

**3. Bonus: Entropy**
The target proportions were 54.13% for category 0 and 45.87% for category 1. Entropy was 0.9951 bits, close to the binary maximum of 1 bit. Therefore, the categories are balanced and reatively unpredictable.

**Conclusion**
The analysis used distribution checks, hypothesis testing, confidence intervals, correlation, and entropy. The chemical example demonstrates that correlation alone does not establish causation.

