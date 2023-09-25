from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
import pandas as pd
import json
import os
import pickle
import datetime


from src import config
from src import inference
from src import helpers
from src import preprocess


app = Flask(__name__)

@app.route("/ping", methods=["GET"])
def ping():
    return jsonify({"status": "ok", "datetime": str(datetime.datetime.now())})

@app.route('/predict', methods=['POST'])
def predict():
    X = request.json.get('X')
    if X is None:
        return jsonify({'error': 'No input data provided'})
    if not isinstance(X, list):
        return jsonify({'error': 'Input data must be a list of key-value items'})
    X = pd.DataFrame(X)
    if "loan_id" not in X.columns:
        return jsonify({'error': 'Input data must contain a loan_id column'})
    ref_job_id = helpers.get_latest_deployed_job_id()
    X = preprocess.preprocess_data(X, mode="inference", rescale=False, ref_job_id=ref_job_id)
    r = inference.make_predictions(X, predictors=config.PREDICTORS)
    # r = X[config.PREDICTORS].to_dict(orient="records")
    return jsonify(r)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5100, debug=True)