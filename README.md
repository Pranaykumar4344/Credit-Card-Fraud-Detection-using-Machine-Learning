
# 💳 Credit Card Fraud Detection API

A machine learning-powered REST API to detect fraudulent credit card transactions using Python, Flask, and XGBoost.

---

## 🚀 Overview

This project demonstrates how to:

- Build a classification model to detect fraudulent transactions
- Handle imbalanced datasets using SMOTE
- Standardize features using scikit-learn's `StandardScaler`
- Train and evaluate an XGBoost model
- Deploy the model via a Flask REST API
- Test real-time predictions using Postman

---

## 📊 Dataset

- **Source:** [Kaggle Credit Card Fraud Detection Dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- **Features:** 30 anonymized variables including `Amount`
- **Target:** `Class` (0 = Legitimate, 1 = Fraudulent)

---

## 🧠 Model Details

- **Algorithm:** XGBoost Classifier
- **Preprocessing:** StandardScaler
- **Imbalanced Data Handling:** SMOTE (Synthetic Minority Over-sampling Technique)
- **Evaluation Metrics:**
  - Accuracy: `~99.98%`
  - ROC-AUC: `0.9999`
  - Precision, Recall, F1-Score: `High`

---

## 🔧 Installation

Clone the repository and install the dependencies:

```bash
git clone https://github.com/Pranaykumar4344/Credit-Card-Fraud-Detection-using-Machine-Learning
cd Credit-Card-Fraud-Detection-using-Machine-Learning
pip install -r requirements.txt
```

---

## 🛠️ Project Structure

```
fraud-detection-api/
├── app.py                  # Flask API
├── train_model.py          # Model training script
├── fraud_detection_model.pkl  # Trained model
├── scaler.pkl              # Trained StandardScaler
└── requirements.txt
```

---

## 🚀 How to Run

### Step 1: Train the Model
```bash
python train_model.py
```

### Step 2: Start the Flask App
```bash
python app.py
```

Flask will be available at: `http://127.0.0.1:5000`

---

## 📬 API Endpoints

### `/`
- **Method:** GET
- **Description:** Health check

### `/predict`
- **Method:** POST
- **Payload Format:**
```json
{
  "features": [value1, value2, ..., value29]
}
```
- **Response:**
```json
{
  "fraud": true | false
}
```

---

## 🧪 Testing with Postman

1. Run `app.py`
2. Open Postman
3. Send a POST request to `http://127.0.0.1:5000/predict`
4. Add JSON body like:
```json
{
  "features": [0.1, -1.2, 2.3, ..., 100.25]
}
```

---

## 📈 Future Improvements

- Add user authentication to the API
- Visual dashboard for real-time monitoring
- Frontend UI for non-technical users
- Model explainability using SHAP or LIME

---

## 🙌 Credits

- [Kaggle ULB Fraud Dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- scikit-learn, XGBoost, Flask, imbalanced-learn

---

## 🪪 License

MIT License

---

## 🔗 Connect With Me

📧 Email: jpranaykumar1205@gmail.com 
🌐 LinkedIn: [My LinkedIn](https://www.linkedin.com/in/janapareddi-pranay-kumar-5897a828a/)  
🐙 GitHub: [My GitHub](https://github.com/Pranaykumar4344)
