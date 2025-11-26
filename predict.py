import os
import joblib
import pandas as pd
from features import extract_features

MODELS_DIR = "models"
MODEL_NAME = "svm_model.pkl"   # You can change to gbc_model.pkl if you want

def load_model():
    model_path = os.path.join(MODELS_DIR, MODEL_NAME)
    return joblib.load(model_path)

def predict_url(url: str):
    model = load_model()
    feats = extract_features(url)
    X = pd.DataFrame([feats])
    
    pred = model.predict(X)[0]
    
    prob = None
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(X)[0][int(pred)]
    
    return pred, prob

if __name__ == "__main__":
    url = input("Enter a URL to check: ").strip()
    label, confidence = predict_url(url)

    if label == 1:
        print("\n🚨 RESULT: This URL is likely **PHISHING**")
    else:
        print("\n✅ RESULT: This URL is likely **LEGITIMATE**")

    if confidence is not None:
        print(f"Model confidence: {confidence:.4f}")
