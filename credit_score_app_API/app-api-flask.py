import os
import pickle

import numpy as np
import pandas as pd
import dask.dataframe as dd

from flask import Flask, render_template, request
import secrets

# Load the model
model_path = "C:\\Users\\yanni\\OneDrive\\Bureau\\P7_Modelisation_risque_defaut_credit\\credit_score_app_API\\mlruns\\968964749954438347\\6e7635adf94c4fd79c6369a05600d325\\artifacts\\model\\model.pkl"

# Chargement du modèle
model = pickle.load(open(model_path, "rb"))

# Load the data
df = pd.read_parquet('C:\\Users\\yanni\\OneDrive\\Bureau\\P7_Modelisation_risque_defaut_credit\\train_set.parquet')
#df = pickle.load(open("C:\\Users\\yanni\\OneDrive\\Bureau\\P7_Modelisation_risque_defaut_credit\\pickle_files\\train_set.pickle", "rb"))

# Copie du jeu de données
train_set = df.copy()
# Passage de l'identifiant du client en index pour la modélisation
train_set.set_index('SK_ID_CURR')
# Suppression de la variable cible pour la modélisation
train_set = train_set.drop(['TARGET'], axis=1)

app = Flask(__name__, template_folder='templates')

# Load configuration from config.py if it exists
try:
    app.config.from_object('config')
except ImportError:
    # Fallback to direct configuration if config.py is not found
    app.config['DEBUG'] = True
    app.config['SECRET_KEY'] = secrets.token_hex(16)  # Generate a secret key if config.py is missing

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction_text = None
    probability = None
    if request.method == 'POST':
        client_id = request.form['id_client']
        all_id_client = list(train_set['SK_ID_CURR'].unique())

        seuil = 0.5

        try:
            ID = int(client_id)
            if ID not in all_id_client:
                prediction_text = "Ce client n'est pas répertorié"
            else:
                X = train_set[train_set['SK_ID_CURR'] == ID]
                X = X.drop(['SK_ID_CURR'], axis=1)

                probability_default_payment = model.predict_proba(X)[:, 1][0]
                probability = round(probability_default_payment, 2)
                if probability >= seuil:
                    prediction_text = "Prêt NON Accordé, risque de défaut"
                else:
                    prediction_text = "Prêt Accordé"
                #probability
                #= probability_default_payment
        except ValueError:
            prediction_text = "ID Client invalide. Veuillez entrer un nombre entier."

    return render_template('index.html', prediction_text=prediction_text, probability=probability)

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
    app.run(debug=True)
