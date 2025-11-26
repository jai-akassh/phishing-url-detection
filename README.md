# phishing-url-detection
Machine Learning based phishing URL detection system using SVM, Gradient Boosting, feature engineering and Flask UI.
# 🛡️ Phishing URL Detection using Machine Learning  
A Machine Learning powered phishing URL detector that analyzes URL patterns, domain structure, lexical features, and suspicious keywords to classify websites as **Legitimate** or **Phishing**.  
Built using **Python, Scikit-Learn, Flask, Bootstrap**, and a complete **feature engineering pipeline**.

---

## 🚀 Features

### ✔️ Machine Learning Models
- **Support Vector Machine (SVM)**  
- **Gradient Boosting Classifier (GBC)**  
- Both models trained on URL-based lexical & domain features  
- Models saved using `joblib`

### ✔️ Feature Engineering (features.py)
Automatically extracts:
- URL length  
- Number of dots, slashes, hyphens  
- HTTPS or HTTP  
- Presence of IP address  
- Subdomain depth  
- Suspicious keywords (`login`, `verify`, `secure`, etc.)  

### ✔️ Web Interface (UI)
- Flask backend  
- Bootstrap 5 responsive frontend  
- Paste a URL and instantly receive prediction  
- Shows model confidence score  
- Clean alert-style result display

### ✔️ CLI Support
Run URL predictions directly from the terminal using:
```bash
python predict.py
