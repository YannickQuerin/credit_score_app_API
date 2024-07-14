import pickle

import dask.dataframe as dd
import numpy as np
import pandas as pd
from flask import Flask, render_template, request
import secrets

model = pickle.load(open("C:\\Users\\yanni\\OneDrive\\Bureau\\P7_Modelisation_risque_defaut_credit\\pickle_files\\best_model.pickle", "rb"))

df = dd.read_parquet('C:\\Users\\yanni\\OneDrive\\Bureau\\P7_Modelisation_risque_defaut_credit\\train_set_without_target.parquet')
app = Flask(__name__, template_folder='templates')

# Load configuration from config.py if it exists
try:
    app.config.from_object('config')
except ImportError:
    # Fallback to direct configuration if config.py is not found
    app.config['DEBUG'] = True
    app.config['SECRET_KEY'] = secrets.token_hex(16)  # Génère une clé secrète si config.py est absent

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    '''For rendering results on HTML GUI'''

    all_id_client = list(df['SK_ID_CURR'].unique())

    seuil = 0.5

    ID = request.form['id_client']
    ID = int(ID)
    if ID not in all_id_client:
        prediction = "Ce client n'est pas répertorié"
        probability_default_payment = None
    else:
        X = df[df['SK_ID_CURR'] == ID]
        X = X.drop(['SK_ID_CURR'], axis=1)

        probability_default_payment = model.predict_proba(X)[:, 1][0]
        if probability_default_payment >= seuil:
            prediction = "Prêt NON Accordé, risque de défaut"
        else:
            prediction = "Prêt Accordé"

    return render_template('index.html', prediction_text=prediction, probability=probability_default_payment)


@app.route('/prediction_complete')
def pred_model():
    Xtot = df.drop(['SK_ID_CURR'], axis=1)
    seuil = 0.5
    y_pred = model.predict_proba(Xtot)[:, 1]
    y_seuil = y_pred >= seuil
    y_seuil = np.array(y_seuil > 0) * 1
    df_pred = df.copy()
    df_pred['Proba'] = y_pred
    df_pred['PREDICTION'] = y_seuil

    test_prediction = df_pred.to_json(orient='index')
    return test_prediction

if __name__ == '__main__':
    app.run()
