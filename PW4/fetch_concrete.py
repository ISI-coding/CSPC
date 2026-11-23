import os
import pandas as pd
from ucimlrepo import fetch_ucirepo

concrete = fetch_ucirepo(id=165)

X = concrete.data.features
Y = concrete.data.targets
print("Dataset Metadata")
print(concrete.metadata)
print("")
print("Dataset Variables")
print(concrete.variables)
X = X.copy()
X.columns = [
    "cement",
    "slag",
    "ash",
    "water",
    "superplastic",
    "coarse_agg",
    "fine_agg",
    "age"
]
Y = Y.iloc[:, 0].rename("strength")
print("")
print("Input Features")
print(X.head())
print("")
print("Target")
print(Y.head())
print("")
print("Feature Shape:", X.shape)
print("Target Shape:", Y.shape)
os.makedirs("data", exist_ok=True)

df = X.assign(strength=Y)
df.to_csv("data/concrete.csv", index = False)
print("")
print("Dataset saved to data/concrete.csv")
print("FINAL DATASET SHAPE:", df.shape)