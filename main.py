# -*- coding: utf-8 -*-
"""
Created on Sat Aug 27 17:18:46 2022

@author: Adithya
"""
from typing import Optional

import pandas as pd
from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__, template_folder='template')
data = pd.read_csv('cleaned_data.csv')
pipe = pickle.load(open("ridgeModel.pkl", 'rb'))


@app.route('/')
def index():
    cities = sorted(data['City'].unique())
    return render_template('webapp.html', cities=cities)


@app.route('/predict', methods=['POST'])
def predictor():
    city: Optional[str] = request.form.get('city')
    bhk = float(request.form.get('bhk'))
    bathroom = float(request.form.get('bathroom'))
    size = float(request.form.get('size'))

    print(city, bhk, bathroom)
    inputter = (pd.DataFrame([[city, bathroom, bhk, size]], columns=['City', 'Bathroom', 'BHK', 'Size']))
    prediction = pipe.predict(inputter)

    return str(np.round(prediction, 2))
    return ""


if __name__ == "__main__":
    app.run(port=5000, debug=True)
