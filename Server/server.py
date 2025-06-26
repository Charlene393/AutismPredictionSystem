from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend (e.g., React at http://localhost:3000)

# Load the trained model pipeline
model_pipeline = joblib.load("best_model.pkl")

# Define expected feature columns (for validation)
expected_cols = [
    'A1_Score', 'A2_Score', 'A3_Score', 'A4_Score', 'A5_Score',
    'A6_Score', 'A7_Score', 'A8_Score', 'A9_Score', 'A10_Score',
    'age', 'gender', 'ethnicity', 'jaundice', 'autism',
    'contry_of_res', 'used_app_before', 'result', 'relation'
]

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from frontend
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No input data provided'}), 400

        # Convert to DataFrame
        input_data = pd.DataFrame([data])

        # Validate input columns
        missing_cols = [col for col in expected_cols if col not in input_data.columns]
        if missing_cols:
            return jsonify({'error': f'Missing columns: {missing_cols}'}), 400

        # Ensure correct column order
        input_data = input_data[expected_cols]

        # Make prediction using the pipeline
        prediction = model_pipeline.predict(input_data)[0]

        # Map prediction to human-readable result
        result = 'Yes' if prediction == 1 else 'No'

        return jsonify({
            'prediction': int(prediction),  # Return 0 or 1
            'result': result
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'Server is running'}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5007, debug=True)