
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
