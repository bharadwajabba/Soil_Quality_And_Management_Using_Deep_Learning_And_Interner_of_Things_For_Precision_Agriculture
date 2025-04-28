from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib

app = Flask(__name__)


try:
    model = joblib.load('model/model.pkl')
except:
    model = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return render_template('index.html', 
                            prediction_text='Model not loaded!',
                            show_result=True)

    try:
        
        fields = ['N', 'P', 'K', 'pH', 'EC', 'OC', 'S', 'Zn', 'Fe', 'Cu', 'Mn', 'B']
        
        
        for field in fields:
            if field not in request.form:
                return f"Missing field: {field}", 400
        
        
        data = {field: float(request.form[field]) for field in fields}
        
        
        features = pd.DataFrame([data])
        
        
        features = features.apply(lambda x: np.log1p(x) if np.issubdtype(x.dtype, np.number) else x)
        
        
        prediction = model.predict(features)[0]
        
        
        results = {
            0: "Poor soil quality - Needs significant improvement",
            1: "Moderate soil quality - Some improvement needed",
            2: "Good soil quality - Minimal improvement needed"
        }
        
        return render_template('index.html', 
                            prediction_text=f'Prediction: {prediction} - {results[prediction]}',
                            show_result=True)
    
    except ValueError as e:
        return render_template('index.html', 
                            prediction_text=f'Error: Please enter valid numbers ({str(e)})',
                            show_result=True)
    except Exception as e:
        return render_template('index.html', 
                            prediction_text=f'Server error: {str(e)}',
                            show_result=True)

if __name__ == '__main__':
    app.run(debug=True)