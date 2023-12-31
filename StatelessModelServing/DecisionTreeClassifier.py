import json
import pickle
from sklearn import tree
import numpy as np
from flask import Flask, jsonify, request


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


app = Flask(__name__)


@app.route("/predict-with-params", methods=['POST'])
def predict_loading_params():
    with open("dt.pkl", "rb") as file:
        model: tree.DecisionTreeClassifier = pickle.load(file)
        # Loading params from param store
        param_file = open("dt_params.pkl", "rb")
        params = pickle.load(param_file)
        print(params)
        model.__dict__ = params
        X = json.loads(request.data)
        print(X)
        response = model.predict(X)
        return json.dumps(response, cls=NumpyEncoder)

@app.route("/predict-with-full-model", methods=['POST'])
def predict_with_full_model():
    with open("dt.pkl", "rb") as file:
        model: tree.DecisionTreeClassifier = pickle.load(file)
        X = json.loads(request.data)
        print(X)
        response = model.predict(X)
        return json.dumps(response, cls=NumpyEncoder)


app.run()
