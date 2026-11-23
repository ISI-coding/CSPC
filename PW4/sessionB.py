
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

os.makedirs("results", exist_ok=True)


# B1
wine = load_wine(as_frame=True)

X = wine.data
cultivar = wine.target  # Do not use until B6

print("\nB1:\n")
print(wine.DESCR)

print("\nDataset shape:", X.shape)

print("\nDescriptive statistics:")
print(X.describe())

statistics = pd.DataFrame({
    "Range": X.max() - X.min(),
    "Standard deviation": X.std()
})

print("\nRanges and standard deviations:")
print(statistics.sort_values("Range", ascending=False))


# B2
pca_raw = PCA()
pca_raw.fit(X)

raw_variance = pca_raw.explained_variance_ratio_

print("\nB2:\n")
print("PC1 explained variance ratio:", raw_variance[0])
print("PC1 explained variance percentage:", raw_variance[0] * 100)

raw_pc1_loadings = pd.Series(
    pca_raw.components_[0],
    index=X.columns
).sort_values(key=abs, ascending=False)

print("\nPC1 loadings, sorted by absolute value:")
print(raw_pc1_loadings)

# B3
scaler = StandardScaler()
Z = scaler.fit_transform(X)
print("\nB3:\n")
print("Means after scaling:")
print(pd.DataFrame(Z, columns=X.columns).mean())

print("\nStandard deviations after scaling:")
print(pd.DataFrame(Z, columns=X.columns).std())

pca = PCA()
pca.fit(Z)

explained = pca.explained_variance_ratio_
cumulative = np.cumsum(explained)

variance_table = pd.DataFrame({
    "Component": [f"PC{i}" for i in range(1, len(explained) + 1)],
    "Explained variance ratio": explained,
    "Cumulative variance": cumulative
})

print("\nExplained variance and cumulative variance:")
print(variance_table.to_string(index=False))

components_90 = np.argmax(cumulative >= 0.90) + 1

print("\nNumber of components needed for 90% variance:",
      components_90)

print("Variance retained by PC1 and PC2:",
      explained[0] + explained[1])

fig, ax1 = plt.subplots(figsize=(9, 5))

component_numbers = np.arange(1, len(explained) + 1)

ax1.bar(component_numbers, explained, label="Individual variance")
ax1.set_xlabel("Principal component")
ax1.set_ylabel("Explained variance ratio")
ax1.set_xticks(component_numbers)

ax2 = ax1.twinx()
ax2.plot(
    component_numbers,
    cumulative,
    marker="o",
    label="Cumulative variance"
)
ax2.axhline(y=0.90, linestyle="--", label="90% variance")
ax2.set_ylabel("Cumulative explained variance")
ax2.set_ylim(0, 1.05)

plt.title("Scree Plot - Scaled Wine Data")
fig.tight_layout()
plt.savefig("results/B_scree_plot.png")
plt.close()

loadings = pd.DataFrame(
    pca.components_[:2].T,
    index=X.columns,
    columns=["PC1", "PC2"]
)

print("\nPC1 and PC2 loadings, sorted by absolute value:")
print(loadings.reindex(
    loadings.abs().max(axis=1).sort_values(ascending=False).index
))

loadings.to_csv("results/B_PCA_loadings.csv")
Z_pca = pca.transform(Z)

plt.figure(figsize=(8, 6))
plt.scatter(Z_pca[:, 0], Z_pca[:, 1])
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("Wine Data on the First Two Principal Components")
plt.savefig("results/B_PCA_scatter.png")
plt.close()

# B4
print("\nB4:\n")

print("\nInertia for different random states (n_init=1):")
for random_state in range(8):
    km_test = KMeans(
        n_clusters=3,
        init="k-means++",
        n_init=1,
        max_iter=300,
        random_state=random_state
    )

    km_test.fit(Z)

    print(
        f"Random state {random_state}: "
        f"inertia = {km_test.inertia_:.4f}"
    )

inertias = []
for k in range(1, 11):
    km_test = KMeans(
        n_clusters=k,
        init="k-means++",
        n_init=10,
        max_iter=300,
        random_state=0
    )

    km_test.fit(Z)
    inertias.append(km_test.inertia_)

inertia_table = pd.DataFrame({
    "k": range(1, 11),
    "Inertia": inertias
})

print("\nInertia for k = 1 to 10:")
print(inertia_table.to_string(index=False))

plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), inertias, marker="o")
plt.xlabel("Number of clusters (k)")
plt.ylabel("Inertia")
plt.title("Elbow Method - Wine Data")
plt.xticks(range(1, 11))
plt.savefig("results/B_elbow_plot.png")
plt.close()

chosen_k = 3

print("\nChosen number of clusters:", chosen_k)

km = KMeans(
    n_clusters=chosen_k,
    init="k-means++",
    n_init=10,
    max_iter=300,
    random_state=0
)

km.fit(Z)

# B5
print("\nB5:\n")

cluster_sizes = np.bincount(km.labels_)

print("\nCluster sizes:")
for cluster, size in enumerate(cluster_sizes):
    print(f"Cluster {cluster}: {size} wines")

centres_original = scaler.inverse_transform(km.cluster_centers_)

centres_df = pd.DataFrame(
    centres_original,
    columns=X.columns
)

print("\nCluster centres in original measurement units:")
print(centres_df.round(3))

centres_df.to_csv("results/B_cluster_centres.csv", index=False)
plt.figure(figsize=(8, 6))
plt.scatter(
    Z_pca[:, 0],
    Z_pca[:, 1],
    c=km.labels_,
    cmap="viridis"
)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("Wine Clusters on the PCA Plane")
plt.savefig("results/B_clusters_PCA.png")
plt.close()


# B6
print("\nB6:\n")

cultivar_names = pd.Series(
    wine.target_names[cultivar],
    index=X.index,
    name="Cultivar"
)

def evaluate_clustering(labels, description):
    table = pd.crosstab(cultivar_names, labels)
    agreement = table.max(axis=0).sum() / len(cultivar)

    print(f"\n{description} - Crosstab:")
    print(table)

    print(f"Agreement: {agreement:.4f} ({agreement * 100:.2f}%)")

    return agreement

km_scaled = KMeans(
    n_clusters=3,
    n_init=10,
    random_state=0
)
km_scaled.fit(Z)

agreement_scaled = evaluate_clustering(
    km_scaled.labels_,
    "Scaled data"
)

km_raw = KMeans(
    n_clusters=3,
    n_init=10,
    random_state=0
)
km_raw.fit(X)

agreement_raw = evaluate_clustering(
    km_raw.labels_,
    "Raw data"
)

Z_pc2 = Z_pca[:, :2]

km_pc2 = KMeans(
    n_clusters=3,
    n_init=10,
    random_state=0
)
km_pc2.fit(Z_pc2)

agreement_pc2 = evaluate_clustering(
    km_pc2.labels_,
    "First two principal components"
)

agreement_table = pd.DataFrame({
    "Method": [
        "Scaled data",
        "Raw data",
        "First two principal components"
    ],
    "Agreement": [
        agreement_scaled,
        agreement_raw,
        agreement_pc2
    ]
})

print("\nAgreement comparison:")
print(agreement_table.to_string(index=False))

agreement_table.to_csv(
    "results/B agreement.csv",
    index=False
)

# Bonus
print("\nBonus:\n")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    cultivar,
    test_size=0.25,
    random_state=42,
    stratify=cultivar
)

knn_model = make_pipeline(
    StandardScaler(),
    KNeighborsClassifier(n_neighbors=5)
)

knn_model.fit(X_train, y_train)
knn_predictions = knn_model.predict(X_test)

accuracy_knn = accuracy_score(y_test, knn_predictions)
pca_knn_model = make_pipeline(
    StandardScaler(),
    PCA(n_components=2),
    KNeighborsClassifier(n_neighbors=5)
)

pca_knn_model.fit(X_train, y_train)
pca_knn_predictions = pca_knn_model.predict(X_test)

accuracy_pca_knn = accuracy_score(y_test, pca_knn_predictions)

print("KNN accuracy with all scaled features:", accuracy_knn)
print("KNN accuracy with 2 PCA components:", accuracy_pca_knn)

