import flask
import pickle
import pandas as pd
import numpy as np

app = flask.Flask(__name__)

with open('model.pkl', 'rb') as f:
    MODEL = pickle.load(f)

test_data = pd.read_csv('data/test_data.csv')

@app.route("/")
def home():
    return """
    Hello! 
    
    Make a prediction at /predict and a sample from the test set will be predicted on.
    
    Thanks for visiting the Machine Failure Prediction API!
    """

@app.route("/predict")
def predict():
    sample = test_data.sample(n=1)

    features = sample.drop(columns=['fail']).values[0]
    target = sample['fail'].values[0]

    prediction = MODEL.predict([features])[0]
    probability = MODEL.predict_proba([features])[0]

    return flask.jsonify({
        'features': features.tolist(),
        'prediction': int(prediction),
        'actual': int(target),
        'probability': probability,
        'correct': prediction == target,
    })


if __name__ == "__main__":
    app.run(host='127.0.0.1', port=5000, debug=True)