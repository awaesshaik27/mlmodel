from flask import Flask,request,jsonify

app = Flask(__name__)


# Load your trained model
from joblib import load
import pandas as pd

model = load('linear_regression_model.pkl')
scaler_reg = load('standard_scaler.pkl')  # Assuming you saved StandardScaler object
@app.route('/')
def index():
    return 'Welcome to My Housing Price Prediction API'

@app.route('/hello/<name>')
def hello_name(name):
    return f'Hello, {name}!'

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        data = request.get_json(force=True)
        # Assuming you will send data similar to the features used in training
        # Ensure the order of features matches the model input
        features = pd.DataFrame(data, index=[0])
        
        # Perform any necessary preprocessing (e.g., scaling) before prediction
        scaled_features = scaler_reg.transform(features)
        
        # Make prediction
        prediction = model.predict(scaled_features)
        
        # Format the prediction as needed
        return jsonify({'prediction': prediction.tolist()})
    else:
        # Handle GET request
        return 'This endpoint is for making predictions. Send a POST request with JSON data.'


if __name__ == '__main__':
    app.run(debug=True)
