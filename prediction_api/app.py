import pickle
import flask
import mlflow.sklearn
import pandas as pd
app = flask.Flask(__name__)

with open('model.pkl', 'rb') as f:
    MODEL = pickle.load(f) #mlflow.sklearn.load_model(f'runs:/251a717917964b1ebe9f74b499f47b58/logistic_model')

DATA = pd.read_csv('data/test_data.csv', header=None)

@app.route("/")
def home():
    return """
    Hello! 
    
    Make a prediction at /predict and a sample from the test set will be predicted on.
    
    Thanks for visiting the Machine Failure Prediction API!
    """

@app.route("/predict")
def predict():
    sample = DATA.sample(n=1)

    features = sample.values[:,:-1]
    target = sample.values[0,-1]

    prediction = MODEL.predict(features)[0]
    probability = MODEL.predict_proba(features)[0]

    return flask.jsonify({
        'prediction': int(prediction),
        'actual': int(target),
        'probability': probability.astype(float).tolist(),
    })

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)