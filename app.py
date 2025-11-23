from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np

# Initialize Flask
app = Flask(__name__)
CORS(app)   # Allow requests from frontend (very important)

# Load model, scaler, and feature order
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
feature_cols = joblib.load("features.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json  # Input from frontend (JSON)

        # Extract features in correct order
        x = [data[col] for col in feature_cols]
        x = np.array(x).reshape(1, -1)

        # Scale input
        x_scaled = scaler.transform(x)

        # Predict probability
        prob = model.predict_proba(x_scaled)[0][1]

        # Define risk category
        if prob < 0.3:
            risk = "Low"
        elif prob < 0.7:
            risk = "Medium"
        else:
            risk = "High"

        return jsonify({
            "prepayment_probability": float(prob),
            "risk_category": risk
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/", methods=["GET"])
def home():
    return "Loan Prepayment Prediction API is running successfully!"


if __name__ == "__main__":
    app.run(debug=True)
