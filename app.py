from flask import Flask, render_template, request
import numpy as np
import pickle
import os

app = Flask(__name__)

# Load model and scaler
model = pickle.load(open('abalone_model.pkl','rb'))
scaler = pickle.load(open('scaler.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():

    Sex = float(request.form['Sex'])
    Length = float(request.form['Length'])
    Diameter = float(request.form['Diameter'])
    Height = float(request.form['Height'])
    Whole_weight = float(request.form['Whole_weight'])
    Shucked_weight = float(request.form['Shucked_weight'])
    Viscera_weight = float(request.form['Viscera_weight'])
    Shell_weight = float(request.form['Shell_weight'])

    features = np.array([[Sex,Length,Diameter,Height,
                          Whole_weight,Shucked_weight,
                          Viscera_weight,Shell_weight]])

    features = scaler.transform(features)

    # prediction = model.predict(features)
    # Predict rings using Ridge model
    rings = model.predict(features)[0]

    # Convert rings to age
    age = rings + 1.5

    return render_template(
        'index.html',
        prediction_text=f"Predicted Rings: {rings:.2f} | Estimated Age of Abalone: {age:.2f} years"
    )

try:
    port = os.environ['PORT'];
except KeyError:
    port = 3000;

if __name__ == "__main__":
    app.run(debug=True, port=port)