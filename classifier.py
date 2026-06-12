# classifier.py
import numpy as np
import pickle
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

def train_classifier(X, y, selected_features, model_save_path="models/classifier.pkl"):
    """
    Train final classifier using GA-selected features.
    Saves model to disk.
    """
    # Use only selected features
    X_selected = X[:, selected_features]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y, test_size=0.2, random_state=42, stratify=y
    )

    print("\n🌿 Training Final Classifier...")
    print(f"   Training samples: {len(X_train)}")
    print(f"   Testing samples:  {len(X_test)}")
    print(f"   Features used:    {len(selected_features)}")

    # Train Random Forest
    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        random_state=42,
        n_jobs=-1
    )
    clf.fit(X_train, y_train)

    # Evaluate
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"\n✅ Classifier Trained!")
    print(f"   Test Accuracy: {acc * 100:.2f}%")
    print("\n📊 Classification Report:")
    print(classification_report(y_test, y_pred))

    # Save model
    os.makedirs("models", exist_ok=True)
    with open(model_save_path, "wb") as f:
        pickle.dump(clf, f)
    print(f"\n💾 Model saved to {model_save_path}")

    return clf, acc


def load_classifier(model_path="models/classifier.pkl"):
    """Load saved classifier from disk."""
    with open(model_path, "rb") as f:
        clf = pickle.load(f)
    return clf


def predict(clf, feature_vector, selected_features, class_names):
    """
    Predict disease for a single image's feature vector.
    Returns predicted class name and confidence.
    """
    # Select only GA-chosen features
    feat = feature_vector[selected_features].reshape(1, -1)

    pred_idx = clf.predict(feat)[0]
    proba = clf.predict_proba(feat)[0]

    confidence = proba[pred_idx] * 100
    predicted_class = class_names[pred_idx]

    return predicted_class, confidence, proba