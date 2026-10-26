# ML Lab-8: Support Vector Machine (SVM) for Classification
# Name: Meet Agrawal | Section: A | Roll No: 41 | Batch: A3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

pd.set_option("display.width", 120)
pd.set_option("display.max_columns", 10)
plt.style.use("default")
sns.set_theme(style="whitegrid")

# 1. Load dataset
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df["species"] = pd.Categorical.from_codes(iris.target, iris.target_names)
print("Dataset shape:", df.shape)
print(df.head())
print("\nClass distribution:\n", df["species"].value_counts())

# 2. Exploratory pair plot
sns.pairplot(df, hue="species", palette="Set2")
plt.suptitle("Pair Plot of Iris Features", y=1.02)
plt.show()

# 3. Train-test split and feature scaling
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# 4. Train SVM with different kernels
kernels = ["linear", "poly", "rbf", "sigmoid"]
results = {}
for k in kernels:
    model = SVC(kernel=k, C=1.0, gamma="scale", random_state=42)
    model.fit(X_train_s, y_train)
    results[k] = accuracy_score(y_test, model.predict(X_test_s))
    print(f"Kernel = {k:8s} Accuracy = {results[k]:.4f}")

plt.figure(figsize=(8, 5))
bars = plt.bar(results.keys(), results.values(), color=sns.color_palette("Set2"))
for b, v in zip(bars, results.values()):
    plt.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}", ha="center")
plt.ylim(0, 1.1)
plt.title("SVM Accuracy by Kernel")
plt.xlabel("Kernel")
plt.ylabel("Test Accuracy")
plt.show()

# 5. Hyperparameter tuning with GridSearchCV
param_grid = {"C": [0.1, 1, 10, 100], "gamma": [0.001, 0.01, 0.1, 1], "kernel": ["rbf"]}
grid = GridSearchCV(SVC(), param_grid, cv=5, scoring="accuracy")
grid.fit(X_train_s, y_train)
print("\nBest parameters:", grid.best_params_)
print("Best CV accuracy: %.4f" % grid.best_score_)

best = grid.best_estimator_
y_pred = best.predict(X_test_s)
print("Test accuracy: %.4f" % accuracy_score(y_test, y_pred))
print("\nClassification Report:\n",
      classification_report(y_test, y_pred, target_names=iris.target_names))

# 6. Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.title("Confusion Matrix - Tuned RBF SVM")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# 7. Decision boundaries on two features (petal length & width)
X2 = iris.data[:, 2:4]
X2_s = StandardScaler().fit_transform(X2)
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
xx, yy = np.meshgrid(np.linspace(X2_s[:, 0].min() - 1, X2_s[:, 0].max() + 1, 400),
                     np.linspace(X2_s[:, 1].min() - 1, X2_s[:, 1].max() + 1, 400))
for ax, k in zip(axes, ["linear", "poly", "rbf"]):
    clf = SVC(kernel=k, C=1.0, gamma="scale").fit(X2_s, y)
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.3, cmap="Set2")
    ax.scatter(X2_s[:, 0], X2_s[:, 1], c=y, cmap="Set2", edgecolors="k", s=35)
    ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1],
               s=110, facecolors="none", edgecolors="red", linewidths=1.2,
               label="Support vectors")
    ax.set_title(f"{k.upper()} kernel ({len(clf.support_vectors_)} SVs)")
    ax.set_xlabel("Petal length (scaled)")
    ax.set_ylabel("Petal width (scaled)")
    ax.legend(loc="upper left")
plt.suptitle("SVM Decision Boundaries")
plt.tight_layout()
plt.show()

# 8. Predict a new sample
sample = np.array([[5.1, 3.5, 1.4, 0.2]])
pred = best.predict(scaler.transform(sample))
print("Prediction for", sample[0], "->", iris.target_names[pred[0]])
