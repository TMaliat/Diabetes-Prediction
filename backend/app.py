from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
import joblib  # To save and load the model

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Load dataset
diabetes_dataset = pd.read_csv("diabetes.csv")

# Data preparation
X = diabetes_dataset.drop(columns="Outcome", axis=1)
Y = diabetes_dataset["Outcome"]

# Standardize data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

# Train model
classifier = svm.SVC(kernel="linear")
classifier.fit(X_train, Y_train)

# Save the model and scaler to disk
joblib.dump(classifier, 'diabetes_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

# Default route
@app.route("/")
def home():
    return "Welcome to the Diabetes Prediction API!"

# Handle favicon requests
@app.route("/favicon.ico")
def favicon():
    return "", 204  # No content

# Prediction endpoint
@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    try:
        input_data = [
            data["pregnancies"],
            data["glucose"],
            data["bloodPressure"],
            data["skinThickness"],
            data["insulin"],
            data["bmi"],
            data["diabetesPedigreeFunction"],
            data["age"],
        ]

        input_array = np.asarray(input_data).reshape(1, -1)

        # Load the saved model and scaler
        classifier = joblib.load('diabetes_model.pkl')
        scaler = joblib.load('scaler.pkl')

        input_standardized = scaler.transform(input_array)
        prediction = classifier.predict(input_standardized)
        result = "Diabetic" if prediction[0] == 1 else "Non-Diabetic"
        return jsonify({"prediction": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
