# train.py
import os
import numpy as np
import pickle
from feature_extractor import extract_features
from genetic_algorithm import run_genetic_algorithm
from classifier import train_classifier

# ── SETTINGS ────────────────────────────────────────────────
DATASET_PATH = "dataset/PlantVillage/PlantVillage"     # folder with subfolders per class
MODEL_DIR    = "models"

def load_dataset(dataset_path):
    """
    Loads images from folder structure:
    dataset/
      ├── Tomato_Healthy/
      ├── Tomato_EarlyBlight/
      └── ...
    Returns: X (features), y (labels), class_names
    """
    X, y = [], []
    class_names = sorted(os.listdir(dataset_path))
    class_names = [c for c in class_names if os.path.isdir(
                   os.path.join(dataset_path, c))]

    print(f"\n📂 Found {len(class_names)} classes:")
    for i, cls in enumerate(class_names):
        print(f"   [{i}] {cls}")

    print("\n⚙️  Extracting features from images...")

    for label, cls in enumerate(class_names):
        cls_path = os.path.join(dataset_path, cls)
        images = os.listdir(cls_path)

        print(f"   Processing {cls} ({len(images)} images)...")

        for img_file in images:
            img_path = os.path.join(cls_path, img_file)
            try:
                features = extract_features(img_path)
                X.append(features)
                y.append(label)
            except Exception as e:
                print(f"   ⚠️  Skipped {img_file}: {e}")

    X = np.array(X)
    y = np.array(y)

    print(f"\n✅ Dataset loaded!")
    print(f"   Total samples : {len(X)}")
    print(f"   Feature vector: {X.shape[1]}")

    return X, y, class_names


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Step 1: Load dataset & extract features
    X, y, class_names = load_dataset(DATASET_PATH)

    # Save class names
    with open(f"{MODEL_DIR}/class_names.pkl", "wb") as f:
        pickle.dump(class_names, f)
    print(f"\n💾 Class names saved.")

    # Step 2: Run Genetic Algorithm
    selected_features, ga_log = run_genetic_algorithm(X, y)

    # Save selected features
    with open(f"{MODEL_DIR}/selected_features.pkl", "wb") as f:
        pickle.dump(selected_features, f)
    print(f"💾 Selected features saved.")

    # Step 3: Train classifier
    clf, accuracy = train_classifier(X, y, selected_features)

    print(f"\n🎉 Training Complete!")
    print(f"   Final Accuracy : {accuracy * 100:.2f}%")
    print(f"   Models saved in: {MODEL_DIR}/")


if __name__ == "__main__":
    main()