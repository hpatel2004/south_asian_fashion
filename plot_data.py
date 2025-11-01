import json
import matplotlib.pyplot as plt
from pathlib import Path
import itertools

# === Load logs from folders ===
def load_jsons_from(folder):
    logs = []
    for file in Path(folder).glob("*.json"):
        with open(file, "r") as f:
            logs.extend(json.load(f))
    return logs

baseline_logs = load_jsons_from("model_results/baseline")
final_logs = load_jsons_from("model_results/final")
all_logs = baseline_logs + final_logs

# === Distinct color cycle ===
color_cycle = itertools.cycle(plt.cm.tab10.colors)  # 10 high-contrast colors

plt.figure(figsize=(10, 6))

for log in all_logs:
    color = next(color_cycle)
    epochs = range(1, len(log["train_acc"]) + 1)

    # Plot train (dashed) and val (solid) with same color
    plt.plot(epochs, log["train_acc"], linestyle='--', color=color, alpha=0.7)
    plt.plot(epochs, log["val_acc"], linestyle='-', color=color, label=f"{log['model']} (Val)")
    plt.scatter(len(log["val_acc"]), log["best_acc"], color=color, edgecolor='black', s=80, marker='*')

plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Training vs Validation Accuracy — ResNet18 Models")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("accuracy_curves.png", dpi=300, bbox_inches='tight')

