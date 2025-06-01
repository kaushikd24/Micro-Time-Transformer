import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from src.eval import pred_actions, true_actions

# Compute confusion matrix
cm = confusion_matrix(true_actions, pred_actions)
labels = ["SELL", "HOLD", "BUY"]

# Plot
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix — Decision Transformer")
plt.show()
