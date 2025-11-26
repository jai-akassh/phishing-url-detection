import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import joblib

from features import extract_features

DATA_PATH = os.path.join("data", "phishing_dataset.csv")
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

def build_features(df):
    all_features = []
    for url in df["url"]:
        all_features.append(extract_features(url))
    feat_df = pd.DataFrame(all_features)
    feat_df["label"] = df["label"]
    return feat_df

def main():
    print("Loading dataset...")
    df = pd.read_csv(DATA_PATH)

    print("Extracting features...")
    feat_df = build_features(df)

    X = feat_df.drop("label", axis=1)
    y = feat_df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Gradient Boosting
    print("\nTraining Gradient Boosting Model...")
    gbc = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1)
    gbc.fit(X_train, y_train)
    gbc_pred = gbc.predict(X_test)
    print("GBC Accuracy:", accuracy_score(y_test, gbc_pred))
    joblib.dump(gbc, os.path.join(MODELS_DIR, "gbc_model.pkl"))

    # SVM
    print("\nTraining SVM Model...")
    svm_clf = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(kernel='rbf', C=2, gamma='scale', probability=True))
    ])
    svm_clf.fit(X_train, y_train)
    svm_pred = svm_clf.predict(X_test)
    print("SVM Accuracy:", accuracy_score(y_test, svm_pred))
    joblib.dump(svm_clf, os.path.join(MODELS_DIR, "svm_model.pkl"))

    print("\nTraining complete! Models saved in /models folder.")

if __name__ == "__main__":
    main()
