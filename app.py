from flask import Flask, request, jsonify
from joblib import load
import pickle
import pandas as pd

# Initialize Flask app
app = Flask(__name__)

# Load models
models = {
    'logistic_regression': load('logistic_regression_model.joblib'),
    # 'svm': load('svm_model.joblib'),
    # 'random_forest': load('random_forest_modelx.pkl'),  # Example of using a pickle file
    # 'bagging': load('bagging_modelx.pkl'),
    # 'random_forest': load('random_forest_model.joblib'),
    # 'bagging': load('bagging_model.joblib'),
    # 'extra_trees': load('extra_trees_model.joblib'),
    # 'xgboost': load('xgboost_model.joblib'),
    # 'adaboost': load('adaboost_model.joblib'),
    # 'catboost': load('catboost_model.joblib'),
    'lgbm': load('lgbm_model.joblib')
}
@app.route('/')
def index():
    return "Welcome to the heart disease prediction API!"
# Define route for predictions
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json

    # Ensure required fields are provided
    required_fields = ['Age', 'Sex', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'ExerciseAngina']
    # print(data)
    if not all(field in data for field in required_fields):
        return jsonify({"error": "Missing required fields"}), 400

    # Convert data to DataFrame for prediction
    filtered_data = {key: value for key, value in data.items() if key in required_fields}

    # Convert filtered data to DataFrame
    df = pd.DataFrame([filtered_data])

    # Select model
    # model_name = data.get('model')  # Default to random forest if not provided
    # if model_name not in models:
    #     return jsonify({"error": f"Model '{model_name}' not found"}), 404
    model_name='lgbm'

    # Get model
    model = models[model_name]
    pred=model.predict(df)
    print(pred)
    # Make prediction
    prediction = model.predict(df)[0]

    return jsonify({
        "model": model_name,
        "prediction": int(prediction)
    })

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
