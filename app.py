import numpy as np
from flask import Flask, request, render_template
import pickle

# Create flask app
flask_app = Flask(__name__)

# Load model
model = pickle.load(open("model.pkl", "rb"))

# Map numeric labels to species names
species_mapping = {0: "Setosa", 1: "Versicolor", 2: "Virginica"}

@flask_app.route("/")
def Home():
    return render_template("index.html")

@flask_app.route("/predict", methods=["POST"])
def predict():
    # Get input values and convert to float
    float_features = [float(x) for x in request.form.values()]
    features = [np.array(float_features)]

    # Make prediction
    prediction = model.predict(features)[0]  # get scalar, not array

    # Map numeric prediction to species name
    species_name = species_mapping[prediction]

    return render_template("index.html", prediction_text=f"The flower species is {species_name}")

if __name__ == "__main__":
    flask_app.run(debug=True)
