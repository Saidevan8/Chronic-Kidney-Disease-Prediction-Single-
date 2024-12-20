from flask import Flask, render_template, request, redirect, url_for
import json
import joblib
import numpy as np

app = Flask(__name__)

def predict(features):
    model = joblib.load('final_model.pkl')
    prediction = model.predict(features)
    predicted_class = prediction[0]
    result = 'Positive for CKD' if predicted_class == 0 else 'Negative for CKD'
    return result

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/submit', methods=['POST'])
def submit():
    # Get features from the form
    features = [
        float(request.form['feature1']),
            float(request.form['feature2']),
            float(request.form['feature3']),
            float(request.form['feature4']),
            float(request.form['feature5']),
            int(request.form['feature6']),
            int(request.form['feature7']),
            int(request.form['feature8']),
            int(request.form['feature9']),
            float(request.form['feature10']),
            float(request.form['feature11']),
            float(request.form['feature12']),
            float(request.form['feature13']),
            float(request.form['feature14']),
            float(request.form['feature15']),
            float(request.form['feature16']),
            float(request.form['feature17']),
            float(request.form['feature18']),
            int(request.form['feature19']),
            int(request.form['feature20']),
            int(request.form['feature21']),
            int(request.form['feature22']),
            int(request.form['feature23']),
            int(request.form['feature24'])
    ]
    # Prepare input for prediction model
    input_features = np.array(features).reshape(1, -1)
    
    # Get prediction
    predicted_value = predict(input_features)
    return render_template('result.html', predicted_value=predicted_value)

if __name__ == '__main__':
    app.run(debug=True)
