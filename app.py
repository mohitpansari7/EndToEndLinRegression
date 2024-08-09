import pickle
from flask import Flask, render_template, request, app, jsonify, url_for

import pandas as pd
import numpy as np

app = Flask(__name__)

reg_model = pickle.load(open('reg_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.json['data']
    print(data)
    print(np.array(list(data.values())).reshape(1,-1))
    new_data_scaled = scaler.transform(np.array(list(data.values())).reshape(1,-1))
    output = reg_model.predict(new_data_scaled)
    print(output[0])
    return jsonify(output[0])

@app.route('/predict', methods=['POST'])
def predict():
    data = [float(x) for x in request.form.values()]
    scaled_input = scaler.transform(np.array(data).reshape(1,-1))
    #predict(scaled_input)
    output = reg_model.predict(scaled_input)[0]
    return render_template("home.html", prediction_text="The House price predicted is {}".format(output))

if __name__ == "__main__":
    app.run(debug=True)



