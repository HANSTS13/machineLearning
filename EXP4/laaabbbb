# ==========================================
# TEAM 02: CLASS IMBALANCE INVESTIGATION
# Research Question: Can a classifier with 
# high accuracy still be practically poor?
# ==========================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    confusion_matrix, 
    classification_report
)

# 1. Generate an Imbalanced Dataset (95% Majority Class 0, 5% Minority Class 1)
X, y = make_classification(
    n_samples=10000, 
    n_features=10, 
    n_informative=5, 
    n_redundant=2, 
    weights=[0.95, 0.05], # Severe class imbalance ratio
    random_state=42
)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print("--- Dataset Information ---")
print(f"Total training samples: {X_train.shape[0]}")
print(f"Total testing samples: {X_test.shape[0]}")
print(f"Class distribution in test set:\n{pd.Series(y_test).value_counts(normalize=True)}")

# 2. Train a Standard Classifier (Logistic Regression)
model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)

# 3. Predict Test Data
y_pred = model.predict(X_test)

# 4. Evaluate Performance Metrics
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, zero_division=0)
rec = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

print("\n--- Performance Metrics ---")
print(f"Accuracy  : {acc:.4f}")
print(f"Precision : {prec:.4f}")
print(f"Recall    : {rec:.4f}")
print(f"F1-Score  : {f1:.4f}")

# Detailed Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 5. Create Results DataFrame for Comparison Table
results_df = pd.DataFrame({
    "Metric": ["Accuracy", "Precision", "Recall", "F1-Score"],
    "Score": [acc, prec, rec, f1]
})
print("\n", results_df)

# 6. Visualizations (Confusion Matrix & Metric Bar Chart)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot A: Confusion Matrix Heatmap
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0], cbar=False)
axes[0].set_title("Confusion Matrix (Exposing the Flaw)")
axes[0].set_xlabel("Predicted Label")
axes[0].set_ylabel("True Label")

# Plot B: Metric Comparison Bar Chart
axes[1].bar(results_df["Metric"], results_df["Score"], color=['skyblue', 'salmon', 'lightgreen', 'orange'])
axes[1].set_title("Metrics Comparison: High Accuracy vs. Poor Recall")
axes[1].set_ylabel("Score (0 to 1)")
axes[1].set_ylim(0, 1.05)
axes[1].grid(axis='y', alpha=0.3)

# Add value labels on top of bars
for bar in axes[1].patches:
    height = bar.get_height()
    axes[1].annotate(f'{height:.3f}',
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3),  # 3 points vertical offset
                     textcoords="offset points",
                     ha='center', va='bottom')

plt.tight_layout()
plt.show()
