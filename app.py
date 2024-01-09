import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import math

app = Flask(__name__)
model = pickle.load(open('model.pkl','rb'))

@app.route('/')
def home():
    return render_template('main.html')

@app.route('/predict', methods=['POST'])
def predict():
    int_features  = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    if prediction == 1:
        output ='You have heart problems'
    else:
        output ='Your heart is okay'
    
    return render_template('main.html',prediction_text="The resault is: {}".format(output))

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5003)