# utils.py
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle

def plot_ga_progress(log, save_path="models/ga_progress.png"):
    """
    Plot GA fitness progress over generations.
    """
    gen    = log.select("gen")
    avgfit = log.select("avg")
    maxfit = log.select("max")

    plt.figure(figsize=(8, 4))
    plt.plot(gen, avgfit, label="Avg Fitness", color="steelblue", linewidth=2)
    plt.plot(gen, maxfit, label="Max Fitness", color="green",
             linewidth=2, linestyle="--")
    plt.xlabel("Generation")
    plt.ylabel("Accuracy")
    plt.title("🧬 Genetic Algorithm Progress")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"📈 GA progress chart saved to {save_path}")
    plt.close()


def print_summary(selected_features, class_names, accuracy):
    """
    Print a nice summary after training.
    """
    print("\n" + "="*50)
    print("       🌿 PLANT DISEASE EA — SUMMARY")
    print("="*50)
    print(f"  Classes detected : {len(class_names)}")
    for i, cls in enumerate(class_names):
        print(f"    [{i}] {cls}")
    print(f"\n  Features selected by GA : {len(selected_features)} / 112")
    print(f"  Final Test Accuracy      : {accuracy * 100:.2f}%")
    print("="*50 + "\n")


def check_dataset(dataset_path="dataset"):
    """
    Check dataset folder and print stats.
    """
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset folder '{dataset_path}' not found!")
        return False

    classes = [c for c in os.listdir(dataset_path)
               if os.path.isdir(os.path.join(dataset_path, c))]

    if len(classes) == 0:
        print("❌ No class folders found inside dataset/")
        return False

    print(f"\n📂 Dataset Summary:")
    print(f"   Path    : {dataset_path}")
    print(f"   Classes : {len(classes)}")
    total = 0
    for cls in sorted(classes):
        count = len(os.listdir(os.path.join(dataset_path, cls)))
        total += count
        print(f"   {cls:<40} {count} images")
    print(f"   {'TOTAL':<40} {total} images")
    return True