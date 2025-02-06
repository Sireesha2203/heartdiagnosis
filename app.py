from flask import Flask, request, jsonify
from joblib import load
import pickle
import pandas as pd
import google.generativeai as genai
import os
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file


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

def RegularizeText(text):
    text = text.replace('\n', ' ')
    text = text.replace('\r', ' ')
    text = text.replace('\t', ' ')
    text = text.replace('  ', ' ')
    text = text.replace('**', '')
    text = text.replace('*', '')
    return text

def getAdvice(health_data):
    # OpenAI API key (replace with your actual API key)
    API_KEY = os.getenv("GENAI_API_KEY")
    
    if not API_KEY:
        print("Error: API key not found. Make sure it's set in the .env file.")
        return None
    
    genai.configure(api_key=API_KEY)
    model = genai.GenerativeModel('gemini-pro')
    chat = model.start_chat(history=[])
    health_text = (
        f"Age: {health_data.get('Age')}, Sex: {health_data.get('Sex')}, "
        f"RestingBP: {health_data.get('RestingBP')}, Cholesterol: {health_data.get('Cholesterol')}, "
        f"FastingBS: {health_data.get('FastingBS')}, MaxHR: {health_data.get('MaxHR')}, "
        f"ExerciseAngina: {health_data.get('ExerciseAngina')}, Height: {health_data.get('Height')}, "
        f"Weight: {health_data.get('Weight')}, Steps: {health_data.get('Steps')}, "
        f"Calories Burned: {health_data.get('Calories Burned')}"
    )
    prompt = (
        "Act as a Doctor and Give me a short health advice based on my health data. "
        f"My Health Data: {health_text}"
        "If the condition is critical, recommend me to visit the nearest hospital immediately. Do not use bold, newline, or any other special characters in your response."
    )
    response = chat.send_message(prompt)
    reply= response.text
    return (reply)

@app.route('/')
def index():
    return "Welcome to the heart disease prediction API!"
# Define route for predictions
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    print(data)
    # Ensure required fields are provided
    required_fields = ['Age', 'Sex', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'ExerciseAngina']
    # print(data)
    if not all(field in data for field in required_fields):
        return jsonify({"error": "Missing required fields"}), 400
    print("1")
    # Convert data to DataFrame for prediction
    filtered_data = {key: value for key, value in data.items() if key in required_fields}
    if filtered_data['Sex']=='Male':
        filtered_data['Sex']=1
    else:
        filtered_data['Sex']=0
    if filtered_data['ExerciseAngina']=='Yes':
        filtered_data['ExerciseAngina']=1
    else:
        filtered_data['ExerciseAngina']=0
    print("2")
    # Convert filtered data to DataFrame
    df = pd.DataFrame([filtered_data])

    # Select model
    # model_name = data.get('model')  # Default to random forest if not provided
    # if model_name not in models:
    #     return jsonify({"error": f"Model '{model_name}' not found"}), 404
    # model_name='lgbm'
    # Select model (default to logistic regression if not provided)
    model_name = data.get('model','logistic_regression')  # Default to 'logistic_regression'
    if model_name=='':
        model_name='logistic_regression'
    if model_name not in models:
        return jsonify({"error": f"Model '{model_name}' not found"}), 404 
    print("3")
    # Get model
    model = models[model_name]
    try:
        pred = model.predict(df)[0]
        print(pred)
    except Exception as e:
        return jsonify({"error": f"Model prediction failed: {str(e)}"}), 500
    print("4")
    # Make prediction
    advice = None
    try:
        advice = getAdvice(data)  # Passing full data object
    except Exception as e:
        print(f"GenAI error: {str(e)}")

    response = {
        "model": model_name,
        "prediction": int(pred),
    }

    # Include AI advice only if it was successfully generated
    if advice:
        response["advice"] = advice

    return jsonify(response)

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
