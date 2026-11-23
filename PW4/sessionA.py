import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.inspection import permutation_importance
os.makedirs("results", exist_ok=True)
df = pd.read_csv("data/concrete.csv")
print("Original shape:", df.shape)
print("Duplicate rows:", df.duplicated().sum())
print("")
print("Missing values:")
print(df.isna().sum())
df = df.drop_duplicates().reset_index(drop=True)
print("")
print("Shape after removing duplicates:", df.shape)

X = df.drop(columns="strength")
Y = df["strength"]
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)
print("")
print("Training and testing sets")
print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
print("Y_train:", Y_train.shape)
print("Y_test:", Y_test.shape)
model = LinearRegression()
model.fit(X_train, Y_train)
Y_train_pred = model.predict(X_train)
Y_test_pred = model.predict(X_test)
print("")
print("Linear regression equation")
print("Intercept:", model.intercept_)

for feature, coefficient in zip(X.columns, model.coef_):
    print(f"{feature}: {coefficient:.4f}")

train_rmse = np.sqrt(mean_squared_error(Y_train, Y_train_pred))
test_rmse = np.sqrt(mean_squared_error(Y_test, Y_test_pred))
test_mae = mean_absolute_error(Y_test, Y_test_pred)
train_r2 = r2_score(Y_train, Y_train_pred)
test_r2 = r2_score(Y_test, Y_test_pred)

n = len(Y_train)
p = X_train.shape[1]
adjusted_r2 = 1 - (1 - train_r2) * (n - 1) / (n - p - 1)
print("")
print("Linear regression metrics")
print("Train rmse:", train_rmse)
print("Test rmse:", test_rmse)
print("Test mae:", test_mae)
print("Train R2:", train_r2)
print("Train Adjusted R2:", adjusted_r2)
print("Test R2:", test_r2)
print("")
print("Descriptive statistics:")
print(df.describe())

plt.figure()
plt.scatter(df["cement"], df["strength"], alpha=0.6)
plt.xlabel("Cement (kg/m^3)")
plt.ylabel("Compressive strength (MPa)")
plt.title("Concrete Strength vs Cement")
plt.tight_layout()
plt.savefig("results/strength_vs_cement.png", dpi=150)
plt.show()

plt.figure()
plt.scatter(df["age"], df["strength"], alpha=0.6)
plt.xlabel("Curing age (days)")
plt.ylabel("Compressive strength (MPa)")
plt.title("Concrete Strength vs Age")
plt.tight_layout()
plt.savefig("results/strength_vs_age.png", dpi=150)
plt.show()

residuals = Y_test - Y_test_pred
plt.figure()
plt.scatter(X_test["age"], residuals, alpha=0.6)
plt.axhline(y=0, color="red", linestyle="--")
plt.xlabel("Age (days)")
plt.ylabel("Residuals (MPa)")
plt.title("Linear Regression Residuals vs Age")
plt.xscale("log")
plt.tight_layout()
plt.savefig("results/residuals_vs_age.png", dpi=150)
plt.show()

degrees = [1, 2, 3, 4]
results = []
for d in degrees:
    polynom = make_pipeline(
        PolynomialFeatures(degree=d, include_bias=False),
        StandardScaler(),
        LinearRegression()
    )
    polynom.fit(X_train, Y_train)

    Y_train_polynom_pred = polynom.predict(X_train)
    Y_test_polynom_pred = polynom.predict(X_test)

    p = polynom[0].n_output_features_
    train_rmse = np.sqrt(mean_squared_error(Y_train, Y_train_polynom_pred))
    test_rmse = np.sqrt(mean_squared_error(Y_test, Y_test_polynom_pred))

    train_r2 = r2_score(Y_train, Y_train_polynom_pred)
    test_r2 = r2_score(Y_test, Y_test_polynom_pred)

    n = len(Y_train)
    adjusted_r2 = 1 - (1 - train_r2) * (n - 1) / (n - p - 1)

    results.append({
        "Degree": d,
        "Features": p,
        "Train RMSE": train_rmse,
        "Test RMSE": test_rmse,
        "Train R2": train_r2,
        "Adjusted R2": adjusted_r2,
        "Test R2": test_r2
    })

results_df = pd.DataFrame(results)
print("")
print("Polynomial Regression Results")
print(results_df.to_string(index=False))

#Part A6
y_class = (df["strength"] >= 40).astype(int)

print("\nClass proportions:")
print(y_class.value_counts(normalize=True))

Xc_train, Xc_test, c_train, c_test = train_test_split(
    X, y_class, test_size=0.2,
    random_state=42, stratify=y_class
)

K_values = [1, 3, 5, 7, 15, 25, 51, 101]
knn_results = []

for K in K_values:
    knn = make_pipeline(
        StandardScaler(),
        KNeighborsClassifier(n_neighbors=K)
    )

    knn.fit(Xc_train, c_train)

    knn_results.append({
        "K": K,
        "Train Accuracy": knn.score(Xc_train, c_train),
        "Test Accuracy": knn.score(Xc_test, c_test)
    })

knn_df = pd.DataFrame(knn_results)

print("\nKNN Results:")
print(knn_df.to_string(index=False))

plt.figure()
plt.plot(K_values, knn_df["Train Accuracy"], marker="o", label="Train")
plt.plot(K_values, knn_df["Test Accuracy"], marker="o", label="Test")
plt.xscale("log")
plt.xlabel("K")
plt.ylabel("Accuracy")
plt.title("KNN Accuracy vs K")
plt.legend()
plt.tight_layout()
plt.savefig("results/knn_accuracy.png")
plt.show()

CHOSEN_K = 7

knn = make_pipeline(
    StandardScaler(),
    KNeighborsClassifier(n_neighbors=CHOSEN_K)
)
knn.fit(Xc_train, c_train)

knn_no_scaler = KNeighborsClassifier(n_neighbors=CHOSEN_K)
knn_no_scaler.fit(Xc_train, c_train)

print("\nKNN with scaler:", knn.score(Xc_test, c_test))
print("KNN without scaler:", knn_no_scaler.score(Xc_test, c_test))

split_accuracies = []

for seed in range(5):
    Xa_train, Xa_test, ya_train, ya_test = train_test_split(
        X, y_class, test_size=0.2,
        random_state=seed, stratify=y_class
    )

    temp_knn = make_pipeline(
        StandardScaler(),
        KNeighborsClassifier(n_neighbors=CHOSEN_K)
    )

    temp_knn.fit(Xa_train, ya_train)
    split_accuracies.append(temp_knn.score(Xa_test, ya_test))

print("\nFive split accuracies:", split_accuracies)
print("Spread:", np.ptp(split_accuracies))


#Decision Tree

depths = [1, 2, 3, 4, 5, 6, 8, None]
tree_results = []

for depth in depths:
    tree = DecisionTreeClassifier(
        criterion="entropy",
        max_depth=depth,
        min_samples_leaf=1,
        random_state=0
    )

    tree.fit(Xc_train, c_train)

    tree_results.append({
        "Depth": str(depth),
        "Leaves": tree.get_n_leaves(),
        "Train Accuracy": tree.score(Xc_train, c_train),
        "Test Accuracy": tree.score(Xc_test, c_test)
    })

tree_df = pd.DataFrame(tree_results)

print("\nDecision Tree Results:")
print(tree_df.to_string(index=False))

plt.figure()
plt.plot(tree_df["Depth"], tree_df["Train Accuracy"], marker="o", label="Train")
plt.plot(tree_df["Depth"], tree_df["Test Accuracy"], marker="o", label="Test")
plt.xlabel("Maximum Depth")
plt.ylabel("Accuracy")
plt.title("Decision Tree Accuracy vs Depth")
plt.legend()
plt.tight_layout()
plt.savefig("results/tree_accuracy.png")
plt.show()

CHOSEN_DEPTH = 5

tree = DecisionTreeClassifier(
    criterion="entropy",
    max_depth=CHOSEN_DEPTH,
    min_samples_leaf=1,
    random_state=0
)

tree.fit(Xc_train, c_train)

print("\nChosen tree depth:", tree.get_depth())
print("Number of leaves:", tree.get_n_leaves())

print("\nTop two levels:")
print(export_text(
    tree,
    feature_names=list(X.columns),
    max_depth=2
))

#Part A7
def calculate_metrics(model, X_test, y_test):
    predictions = model.predict(X_test)

    tn, fp, fn, tp = confusion_matrix(
        y_test, predictions, labels=[0, 1]
    ).ravel()

    return {
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(y_test, predictions, zero_division=0),
        "Recall": recall_score(y_test, predictions, zero_division=0),
        "F1": f1_score(y_test, predictions, zero_division=0)
    }


metrics_results = []

for name, model in [("KNN", knn), ("Decision Tree", tree)]:
    result = calculate_metrics(model, Xc_test, c_test)
    result["Model"] = name
    metrics_results.append(result)

metrics_df = pd.DataFrame(metrics_results)

print("\nClassification Metrics:")
print(metrics_df.to_string(index=False))

probabilities = knn.predict_proba(Xc_test)[:, 1]

threshold_results = []

for threshold in [0.3, 0.5, 0.7, 0.9]:
    predictions = (probabilities >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        c_test, predictions, labels=[0, 1]
    ).ravel()

    threshold_results.append({
        "Threshold": threshold,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
        "Precision": precision_score(c_test, predictions, zero_division=0),
        "Recall": recall_score(c_test, predictions, zero_division=0),
        "F1": f1_score(c_test, predictions, zero_division=0)
    })

threshold_df = pd.DataFrame(threshold_results)

print("\nThreshold Results:")
print(threshold_df.to_string(index=False))

metrics_df.to_csv("results/A metrics.csv", index=False)
print("\nMetrics saved.")

#Part A8
linear_model = LinearRegression()
linear_model.fit(X_train, Y_train)

raw_coef = linear_model.coef_

standard_model = make_pipeline(
    StandardScaler(),
    LinearRegression()
)

standard_model.fit(X_train, Y_train)
standard_coef = standard_model[-1].coef_

tree_importance = tree.feature_importances_

knn_perm = permutation_importance(
    knn, Xc_test, c_test,
    scoring="f1", n_repeats=20, random_state=0
)

tree_perm = permutation_importance(
    tree, Xc_test, c_test,
    scoring="f1", n_repeats=20, random_state=0
)

importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Raw Coefficients": raw_coef,
    "Standardized Coefficients": standard_coef,
    "Tree Importance": tree_importance,
    "KNN Permutation": knn_perm.importances_mean,
    "Tree Permutation": tree_perm.importances_mean
})

print("\nFeature Importance:")
print(importance_df.to_string(index=False))

for column, filename in [
    ("Standardized Coefficients", "standardized_coefficients.png"),
    ("Tree Importance", "tree_importance.png"),
    ("KNN Permutation", "knn_permutation.png")
]:
    plt.figure()
    plt.bar(importance_df["Feature"], importance_df[column])
    plt.xticks(rotation=45)
    plt.ylabel(column)
    plt.title(column)
    plt.tight_layout()
    plt.savefig("results/" + filename)
    plt.show()


#Bonus

X_physics = X.copy()

X_physics["age"] = np.log(X_physics["age"])

X_physics["w_b"] = (
    df["water"] /
    (df["cement"] + df["slag"] + df["ash"])
)

X_physics = X_physics.drop(columns="water")

Xp_train, Xp_test, yp_train, yp_test = train_test_split(
    X_physics, Y,
    test_size=0.2, random_state=42
)

physics_model = LinearRegression()
physics_model.fit(Xp_train, yp_train)

physics_pred = physics_model.predict(Xp_test)

print("\nPhysics-based model:")
print("Number of features:", Xp_train.shape[1])
print("Test R2:", r2_score(yp_test, physics_pred))