import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
from scipy.stats import shapiro 

heart_data = pd.read_csv("heart.csv")
"""
print("Dataset Shape:", heart_data.shape)
print("")
print("First five rows:")
print(heart_data.head())
continuous = heart_data[["age", "chol", "trestbps", "thalach"]]
print("Summary statistics:")
print(continuous.describe())

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
columns = ["age", "chol", "trestbps", "thalach"]

for ax, column in zip(axes.flat, columns):
    ax.hist(heart_data[column], bins = 15, edgecolor="black")
    ax.set_title(f"Distribution of {column}")
    ax.set_xlabel(column)
    ax.set_ylabel("Frequency")
plt.tight_layout()
plt.savefig("plots/heart_distributions.png", dpi=150)

fig2, axes2 = plt.subplots(2, 2, figsize=(10, 8))
for ax, column in zip(axes2.flat, columns):
    stats.probplot(heart_data[column], dist="norm", plot=ax)
    ax.set_title(f"Q-Q plot of {column}")

plt.tight_layout()
plt.savefig("plots/heart_Q-Q_plots.png", dpi=150)


disease = heart_data[heart_data["target"] == 1]["thalach"]
healthy = heart_data[heart_data["target"] == 0]["thalach"]
W_disease, p_disease = stats.shapiro(disease)
W_healthy, p_healthy = stats.shapiro(healthy)
print("")
print("Shapiro-Wilk normality test:")
print(f"Disease group: W = {W_disease:.4f}, p = {p_disease:.6g}")
print(f"Healthy group: W = {W_healthy:.4f}, p = {p_healthy:.6g}")

#Mann-Whitney U test
u_stat, p_value = stats.mannwhitneyu(disease, healthy, alternative="two-sided")
print("")
print("Mann-Whitney U test:")
print(f"U statistic = {u_stat:.3f}")
print(f"p value = {p_value:.6g}")
if p_value < 0.05:
    print("Reject H0: there is a statistically significant difference.")
else:
    print("Fail to reject H0: insufficient evidence of difference.")

#Mean and 95% confidence interval 
def mean_ci(data):
    mean = data.mean()
    se = stats.sem(data)
    margin = stats.t.ppf(0.975, len(data) - 1) * se
    return mean, mean - margin, mean + margin
for name, group in[("Disease", disease), ("Healthy", healthy)]:
    mean, low, high = mean_ci(group)
    print(f"{name}: mean = {mean:.2f}, " f"95% CI = [{low:.2f}, {high:.2f}]")

#Plotting results
groups = [disease, healthy]
labels = ["Disease", "Healthy"]
means = [g.mean() for g in groups]
errors = [stats.t.ppf(0.975, len(g) - 1) * stats.sem(g) for g in groups]
plt.figure(figsize=(7, 5))
plt.errorbar(labels, means, yerr=errors, fmt="o", capsize=6)
plt.ylabel("Maximum heart rate (thalach)")
plt.title(f"Mean maximum heart rate with 95% confidence intervals")
plt.tight_layout()
plt.savefig("plots/thalach_comparison.png", dpi=150)

#Correlation between age and maximum heart rate
age = heart_data["age"]
thalach = heart_data["thalach"]
r, p_value = stats.pearsonr(age, thalach)
print("")
print("Correlation between age and maximum heart rate")
print(f"Pearson correlation coefficient = {r:.4f}")
print(f"p_value = {p_value:.6g}")

plt.figure(figsize=(7, 5))
plt.scatter(age, thalach, alpha=0.7)
plt.xlabel("Age (years)")
plt.ylabel("Maximum heart rate (thalach)")
plt.title("Age vs maximum heart rate")
plt.tight_layout()
plt.savefig("plots/age_thalach_correlation.png", dpi=150)

#Task 6: Naive chemical analysis
chemical_data = pd.read_csv("chemicals_cancer.csv")
print("")
print("Chemical dataset shape:")
print(chemical_data.shape)
print("")
print("First five rows")
print(chemical_data.head())
print("")
print("Column names:")
print(chemical_data.columns.tolist())

benzene_r, benzene_p = stats.pearsonr(chemical_data["benzene"], chemical_data["malignancy"])
cadmium_r, cadmium_p = stats.pearsonr(chemical_data["cadmium"], chemical_data["malignancy"])
print("")
print("Naive chemical associations:")
print(f"Benzene: r = {benzene_r:.4f}, p = {benzene_p:.6g}")
print(f"Cadmium: r = {cadmium_r:.4f}, p = {cadmium_p:.6g}")

#Restricting patients to a similar polution groups
moderate_pollution = chemical_data[(chemical_data["pollution_index"] >= 40) & (chemical_data["pollution_index"] <=60)]
print("")
print("Patients with pollution index between 40 and 60:")
print(len(moderate_pollution))
benzene_r_restricted, benzene_p_restricted = stats.pearsonr(moderate_pollution["benzene"], moderate_pollution["malignancy"])
cadmium_r_restricted, cadmium_p_restricted = stats.pearsonr(moderate_pollution["cadmium"], moderate_pollution["malignancy"])
print("")
print("Associations within the restricted groups:")
print(f"Benzene: r = {benzene_r_restricted:.4f}, p = {benzene_p_restricted:.6g}")
print(f"Cadmium: r = {cadmium_r_restricted:.4f}, p = {cadmium_p_restricted:.6g}")

print(chemical_data["pollution_index"].describe())

#Bonus
proportions = heart_data["target"].value_counts(normalize=True)
entropy = -(proportions * np.log2(proportions)).sum()
print("Proportions:")
print(proportions)
print(f"Entropy: {entropy:.4f} bits")
"""
columns = ["age", "chol", "trestbps", "thalach"]

for col in columns:
    W, p = shapiro(heart_data[col])
    print(f"{col}: W = {W:.4f}, p = {p:.6g}")