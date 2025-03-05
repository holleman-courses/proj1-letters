import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# Load the data
df = pd.read_csv('hw4_data.csv')

# Extract columns into NumPy arrays
model_output = df['model_output'].to_numpy()
true_class = df['true_class'].to_numpy()
y_pred = df['prediction'].to_numpy()

#1 For the data shown in the table above, find the number of 
#True positives 
#False positives
#True negatives
#False negatives

TP = np.sum((true_class == 1) & (y_pred == 1))
FP = np.sum((true_class == 0) & (y_pred == 1))
TN = np.sum((true_class == 0) & (y_pred == 0))
FN = np.sum((true_class == 1) & (y_pred == 0))
print(f"True Positives (TP): {TP}")
print(f"False Positives (FP): {FP}")
print(f"True Negatives (TN): {TN}")
print(f"False Negatives (FN): {FN}")

#2 Find the precision and recall.
precision = TP / (TP + FP) if (TP + FP) > 0 else 0
recall = TP / (TP + FN) if (TP + FN) > 0 else 0
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")

#3 Plot a ROC curve
fpr, tpr, thresholds = roc_curve(true_class, model_output)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Diagonal line
plt.xlabel("False Positive Rate (FPR)")
plt.ylabel("True Positive Rate (TPR)")
plt.title("Receiver Operating Characteristic (ROC) Curve")
plt.legend()
plt.grid()
plt.show()

#4 What is the minimum False Positive Rate you can achieve (by varying the threshold), 
# if you must correctly detect at least 90% of all actual positives?  
required_recall = 0.90
valid_fpr = fpr[tpr >= required_recall]
min_fpr = np.min(valid_fpr) if len(valid_fpr) > 0 else None

print(f"Minimum False Positive Rate at ≥90% Recall: {min_fpr:.4f}" if min_fpr is not None else "No threshold meets the recall requirement.")
