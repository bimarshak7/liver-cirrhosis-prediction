from flask import  Blueprint,render_template, request
import numpy as np
import joblib

bp = Blueprint('predict_routes', __name__)


# Load the saved model
scaler = joblib.load('../models/scaler.joblib')
model = joblib.load('../models/ada_best.joblib')
best_cols = np.array([ 0,  2,  4,  5,  6,  8,  9, 10, 11, 13, 15, 16])

@bp.route('/predict')
def index():
    return render_template('form.html')

# Define a route to handle incoming requests
@bp.route('/predict', methods=['POST'])
def predict():
    features = process(request.form).reshape(1,-1)
    features = scaler.transform(features)
    print(features)
    prediction = model.predict(features[:,best_cols])
    print("prediction ",prediction)
    
    return {"prediction":int(prediction[0])}

def process(form):
    n_days = int(request.form['N_Days'])
    drug = int(request.form['Drug'])
    age = float(request.form['Age'])
    sex = int(request.form['Sex'])
    ascites = int(request.form['Ascites'])
    hepatomegaly = int(request.form['Hepatomegaly'])
    spiders = int(request.form['Spiders'])
    edema = int(request.form['Edema'])
    bilirubin = float(request.form['Bilirubin'])
    cholesterol = float(request.form['Cholesterol'])
    albumin = float(request.form['Albumin'])
    copper = float(request.form['Copper'])
    alk_phos = float(request.form['Alk_phos'])
    sgot = float(request.form['SGOT'])
    tryglicerides = float(request.form['Tryglicerides'])
    platelets = float(request.form['Platelets'])
    prothrombin = float(request.form['Prothrombin'])

    features = np.array([n_days,drug,age,sex,ascites,hepatomegaly,spiders,edema,bilirubin,cholesterol,albumin,copper,alk_phos,sgot,tryglicerides,platelets,prothrombin])

    

    return features