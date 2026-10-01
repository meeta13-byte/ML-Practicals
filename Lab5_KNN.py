# Lab 5: K-Nearest Neighbor (KNN) for Classification and Regression
# Name: Meet Agrawal | Section: A | Roll No: 41 | Batch: A3
# Dataset: Pima Indians Diabetes (Kaggle: saurabh00007/diabetescsv)

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import kagglehub

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report,
                             mean_absolute_error, mean_squared_error, r2_score)

plt.style.use("default")   # white background for screenshots

# ---------------------------------------------------------------
# 1. Load dataset
# ---------------------------------------------------------------
path = kagglehub.dataset_download("saurabh00007/diabetescsv")
print("Path to dataset files:", path)
csv_file = [f for f in os.listdir(path) if f.endswith(".csv")][0]
df = pd.read_csv(os.path.join(path, csv_file))

print("\nShape:", df.shape)
print("\nFirst 5 rows:\n", df.head())
print("\nInfo:")
df.info()
print("\nStatistical summary:\n", df.describe())
print("\nClass distribution:\n", df["Outcome"].value_counts())

# ---------------------------------------------------------------
# 2. Pre-processing: zeros in these columns are invalid -> NaN -> median
# ---------------------------------------------------------------
cols_with_zero = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
df[cols_with_zero] = df[cols_with_zero].replace(0, np.nan)
print("\nMissing values after replacing 0 with NaN:\n", df.isnull().sum())
df[cols_with_zero] = df[cols_with_zero].fillna(df[cols_with_zero].median())

plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Heatmap of Diabetes Dataset")
plt.tight_layout()
plt.show()

# ===============================================================
# PART A : KNN CLASSIFICATION  (target = Outcome)
# ===============================================================
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# Find best K
k_values = range(1, 26)
acc = []
for k in k_values:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train_s, y_train)
    acc.append(accuracy_score(y_test, model.predict(X_test_s)))

best_k = k_values[int(np.argmax(acc))]
print(f"\nBest K for classification = {best_k}, accuracy = {max(acc):.4f}")

plt.figure(figsize=(8, 5))
plt.plot(k_values, acc, marker="o")
plt.axvline(best_k, color="red", linestyle="--", label=f"Best K = {best_k}")
plt.xlabel("Value of K"); plt.ylabel("Test Accuracy")
plt.title("KNN Classification: K vs Accuracy")
plt.legend(); plt.grid(True); plt.tight_layout(); plt.show()

clf = KNeighborsClassifier(n_neighbors=best_k)
clf.fit(X_train_s, y_train)
y_pred = clf.predict(X_test_s)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["No Diabetes", "Diabetes"],
            yticklabels=["No Diabetes", "Diabetes"])
plt.xlabel("Predicted"); plt.ylabel("Actual")
plt.title(f"Confusion Matrix (K = {best_k})")
plt.tight_layout(); plt.show()

# ===============================================================
# PART B : KNN REGRESSION  (target = BMI)
# ===============================================================
Xr = df.drop("BMI", axis=1)
yr = df["BMI"]

Xr_train, Xr_test, yr_train, yr_test = train_test_split(
    Xr, yr, test_size=0.2, random_state=42)

scaler_r = StandardScaler()
Xr_train_s = scaler_r.fit_transform(Xr_train)
Xr_test_s = scaler_r.transform(Xr_test)

rmse_list = []
for k in k_values:
    reg = KNeighborsRegressor(n_neighbors=k)
    reg.fit(Xr_train_s, yr_train)
    rmse_list.append(np.sqrt(mean_squared_error(yr_test, reg.predict(Xr_test_s))))

best_kr = k_values[int(np.argmin(rmse_list))]
print(f"\nBest K for regression = {best_kr}, RMSE = {min(rmse_list):.4f}")

plt.figure(figsize=(8, 5))
plt.plot(k_values, rmse_list, marker="o", color="green")
plt.axvline(best_kr, color="red", linestyle="--", label=f"Best K = {best_kr}")
plt.xlabel("Value of K"); plt.ylabel("RMSE")
plt.title("KNN Regression: K vs RMSE")
plt.legend(); plt.grid(True); plt.tight_layout(); plt.show()

reg = KNeighborsRegressor(n_neighbors=best_kr)
reg.fit(Xr_train_s, yr_train)
yr_pred = reg.predict(Xr_test_s)

print("\nRegression metrics (target = BMI)")
print("MAE  :", mean_absolute_error(yr_test, yr_pred))
print("MSE  :", mean_squared_error(yr_test, yr_pred))
print("RMSE :", np.sqrt(mean_squared_error(yr_test, yr_pred)))
print("R2   :", r2_score(yr_test, yr_pred))

plt.figure(figsize=(6, 6))
plt.scatter(yr_test, yr_pred, alpha=0.6)
lims = [yr_test.min(), yr_test.max()]
plt.plot(lims, lims, "r--", label="Perfect prediction")
plt.xlabel("Actual BMI"); plt.ylabel("Predicted BMI")
plt.title(f"KNN Regression: Actual vs Predicted BMI (K = {best_kr})")
plt.legend(); plt.grid(True); plt.tight_layout(); plt.show()
