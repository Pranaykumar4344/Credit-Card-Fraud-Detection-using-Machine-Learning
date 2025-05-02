from flask import Flask, request, jsonify
import joblib
import numpy as np

# Load the saved model and scaler
model = joblib.load('fraud_detection_model.pkl')  
scaler = joblib.load('scaler.pkl') 

# Initialize Flask app
app = Flask(__name__)

# Define a route for prediction
@app.route('/')
def home():
    return "Welcome to the Fraud Detection API!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from request
        data = request.get_json()

        # Check if 'features' key is in the request body
        if 'features' not in data:
            return jsonify({'error': "'features' key is missing in the request"}), 400

        # Convert JSON data to a numpy array
        features = np.array(data['features']).reshape(1, -1)

        # Standardize the features using the scaler
        features_scaled = scaler.transform(features)

        # Make prediction using the model
        prediction = model.predict(features_scaled)

        # Return the result as JSON
        result = {'fraud': bool(prediction[0])}
        return jsonify(result), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 400

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
