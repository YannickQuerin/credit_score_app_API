

import streamlit as st

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import dask.dataframe as dd

import plotly.graph_objects as go
import shap
import pickle
from streamlit_shap import st_shap
from xplotter.insights import *

from PIL import Image


# Configuration de la page
st.set_page_config(
    page_title="Application de Crédit",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Style CSS
st.markdown("""
    <style>
    body {font-family:'Roboto Condensed';}
    h1 {font-family:'Roboto Condensed';}
    h2 {font-family:'Roboto Condensed';}
    p {font-family:'Roboto Condensed'; color:Gray; font-size:1.125rem;}
    .css-18e3th9 {padding-top: 1rem; padding-right: 1rem; padding-bottom: 1rem; padding-left: 1rem;}
    .css-184tjsw p {font-family:'Roboto Condensed'; color:Gray; font-size:1rem;}
    </style> 
    """,
    unsafe_allow_html=True
)

# Sidebar

# Centrage de l'image du logo dans la sidebar
col1, col2, col3 = st.columns([1,1,1])
with col1:
    st.sidebar.write("")
with col2:
    image = Image.open('P7_Modelisation_risque_defaut_credit/images/logo_proj7_credit.png')
    st.sidebar.image(image, use_column_width="always")
with col3:
    st.sidebar.write("")

st.sidebar.title("Navigation")
options = st.sidebar.radio("Menu", ["Accueil", "Analyse Client", "Profil Client", "Score Client"])

# Page d'accueil
if options == "Accueil":
    st.title("Bienvenue sur l'application de Crédit 🏠")
    st.write("""
        Cette application vous permet d'analyser le risque de défaut de paiement de vos clients, 
        de visualiser le profil de vos clients et de comprendre comment les scores de risque sont calculés.
        Utilisez le menu à gauche pour naviguer entre les différentes sections.
    """)

# Fonctionnalités partagées
@st.cache
def load_data(file_path):
    try:
        data = pd.read_csv(file_path)
        return data
    except pd.errors.ParserError:
        st.error(f"Erreur lors de la lecture du fichier {file_path}. Veuillez vérifier le format du fichier.")
        return None

@st.cache #mise en cache de la fonction pour exécution unique
def lecture_X_test_original():
    X_test_original = dd.read_parquet("P7_Modelisation_risque_defaut_credit/test_preprocess.parquet")
    X_test_original = X_test_original.compute()
    X_test_original['DUREE_REMBOURSEMENT'] = (X_test_original['AMT_CREDIT'] / X_test_original['AMT_ANNUITY']).round(0)
    X_test_original['TAUX_ENDETTEMENT'] = (X_test_original['AMT_ANNUITY'] / X_test_original['AMT_INCOME_TOTAL']) * 100
    return X_test_original

@st.cache
def lecture_X_test_clean():
    X_test_clean = dd.read_parquet("P7_Modelisation_risque_defaut_credit/test_set.parquet")
    X_test_clean = X_test_clean.compute()
    return X_test_clean



###########################
# Calcul des valeurs SHAP #
###########################
@st.cache
def calcul_valeurs_shap():
    model_LGBM = pickle.load(open("P7_Modelisation_risque_defaut_credit/pickle_files/best_model.pickle", "rb"))
    explainer = shap.TreeExplainer(model_LGBM)
    shap_values = explainer.shap_values(lecture_X_test_clean().drop(labels="SK_ID_CURR", axis=1))
    return shap_values


# Pages spécifiques
if options == "Analyse Client":
    st.title("Analyse Client 📊")
    # Contenu de Analyse_Client.py

    # Titre 1

    # Inclusion des polices Google
    st.markdown("""
    <link href="https://fonts.googleapis.com/css2?family=Lobster&family=Montserrat:ital@1&family=Open+Sans&family=Dancing+Script:ital@1&family=Source+Code+Pro&display=swap" rel="stylesheet">
    """, unsafe_allow_html=True)

    # Style CSS pour le contenu
    style = """
    <style>
    body {
        background-color: #F0F0F0;
    }
    h1 {
        font-family: 'Montserrat', sans-serif;
        color: #1E90FF;
        font-size: 2.3em;
        font-style: italic;
        font-weight: 700;
        margin: 0px;
    }
    p {
        font-family: 'Open Sans', sans-serif;
        color: #333333;
        font-style: normal;
        margin: 10px 0;
    }
    blockquote {
        font-family: 'Dancing Script', cursive;
        color: #50C878;
        background-color: #FFF8DC;
        padding: 10px;
        border-left: 5px solid #50C878;
        font-style: italic;
    }
    code {
        font-family: 'Source Code Pro', monospace;
        color: #0047AB;
        background-color: #F0F0F0;
        padding: 2px 4px;
        border-radius: 4px;
        font-style: normal;
    }
    </style>
    """

    # Appliquer le style CSS
    st.markdown(style, unsafe_allow_html=True)

    # Contenu de la page
    st.markdown("""
        <h1>1. Quelles sont les variables globalement les plus importantes pour comprendre la prédiction ?</h1>
        """, unsafe_allow_html=True)

    st.write("")

    st.write("L’importance des variables est calculée en moyennant la valeur absolue des valeurs de Shap. \
        Les caractéristiques sont classées de l'effet le plus élevé au plus faible sur la prédiction. \
        Le calcul prend en compte la valeur SHAP absolue, donc peu importe si la fonctionnalité affecte \
        la prédiction de manière positive ou négative.")

    st.write("Pour résumer, les valeurs de Shapley calculent l’importance d’une variable en comparant ce qu’un modèle prédit \
        avec et sans cette variable. Cependant, étant donné que l’ordre dans lequel un modèle voit les variables peut affecter \
        ses prédictions, cela se fait dans tous les ordres possibles, afin que les fonctionnalités soient comparées équitablement. \
        Cette approche est inspirée de la théorie des jeux.")

    st.write("*__Le diagramme d'importance des variables__* répertorie les variables les plus significatives par ordre décroissant.\
        Les *__variables en haut__* contribuent davantage au modèle que celles en bas et ont donc un *__pouvoir prédictif élevé__*.")

    st.write("On peut constater que les principales variables ayant le plus grand pouvoir prédictif pour le modèle sont représentés par \
        les variables *__EXT_SOURCE_1, EXT_SOURCE_2, EXT_SOURCE_3__* (scores normalisées issues de données externes: valeur entre 0 et 1) \
         suivie de la variable *__CREDIT_TO_ANNUITY_RATIO__* (ratio  montant credit/montant annuité, c'est-a-dire la proportion du montant \
         de crédit sur le montant de l'annuité), le sexe de la personne et autres.")


    fig = plt.figure()
    plt.title("Interprétation Globale :\n Diagramme d'Importance des Variables",
              fontname='Roboto Condensed',
              fontsize=20,
              fontstyle='italic')
    st_shap(shap.summary_plot(calcul_valeurs_shap()[1],
                              feature_names = lecture_X_test_clean().drop(labels="SK_ID_CURR", axis=1).columns,
                              plot_size=(12, 16),
                              color='orange',
                              plot_type="bar",
                              max_display=56,
                              show=False))
    plt.show()
    plt.close(fig)
    #style="color:#772b58;font-size:2.3em;font-style:italic;font-weight:700;margin:0px;"

    # Titre 2
    st.markdown("""
                    <h1>
                    2. Quel est l'Impact de chaque caractéristique sur la prédiction ?</h1>
                    """,
                unsafe_allow_html=True)
    st.write("")

    st.write("Le diagramme des valeurs SHAP ci-dessous indique également comment chaque caractéristique impacte la prédiction. \
                Les valeurs de Shap sont représentées pour chaque variable dans leur ordre d’importance. \
                Chaque point représente une valeur de Shap (pour un client).")
    st.write(
        "Les points rouges représentent des valeurs élevées de la variable et les points beiges des valeurs basses de la variable.")

    fig = plt.figure()
    plt.title("Interprétation Globale :\n Impact de chaque caractéristique sur la prédiction\n",
              fontname='Roboto Condensed',
              fontsize=20,
              fontstyle='italic')
    st_shap(shap.summary_plot(calcul_valeurs_shap()[1],
                              features=lecture_X_test_clean().drop(labels="SK_ID_CURR", axis=1),
                              feature_names=lecture_X_test_clean().drop(labels="SK_ID_CURR", axis=1).columns,
                              plot_size=(12, 16),
                              cmap='OrRd',
                              plot_type="dot",
                              max_display=56,
                              show=False))
    plt.show()
    plt.close(fig)

    st.write("14 variables ont un impact significatif sur la prédiction (Moyenne des valeurs absolues des valeurs de Shap >= 0.1). \
                La première est sans contexte le score normalisé à partir d'une source de données externes.")
    st.markdown("""
        1.  Plus la valeur du 'Score normalisé à partir d'une source de données externe' est faible (points de couleur beige),
            et plus la valeur Shap est élevée et donc plus le modèle prédit que le client aura des difficultés de paiement.<br>
        2.  Plus le ratio montant credit/montant annuité est faible (points de couleur beige), et plus la valeur Shap est faible et donc
            plus le modèle prédit que les difficultés de paiement du client diminue.<br>
        3.  Si le client est un homme, la valeur Shap est élevée et donc plus le modèle prédit qu'il aura des difficultés de paiement.<br>
        4.  Plus la part d'intéret (durée du crédit précédent pondérée avec la variation entre le montant de l'annuité et celui du crédit) \
            est elevée (points de couleur rouge), plus la valeur Shap est élevée et donc plus le modèle prédit qu'il aura des difficultés de paiement.<br>
        5.  Plus la date à laquelle le versement du crédit précédent était censé
            être payé (par rapport à la date de demande du prêt actuel) *__DAYS_PAYMENT_RATIO__* est élevé (points de couleur rouge),
            plus la valeur Shap est élevée et donc plus le modèle prédit qu'il aura des difficultés de paiement.<br>
        6.  Plus le client accumule des années d'emploi (points de couleur rouge),
            plus la valeur Shap est élevée et donc plus le modèle prédit qu'il aura des difficultés de paiement.<br>
        7.  Plus l'annuité du crédit est faible (points beige), plus la valeur Shap est faible et
            donc plus le modèle prédit que les difficultés de paiement du client diminue.<br>
        8.  Plus le montant de l'acompte prescrit du crédit précédent sur cet acompte est faible (points beige),
            la valeur Shap est élevée et donc plus le modèle pédit que le client aura des difficultés de paiement.<br>
        9.  Plus le client accumule des petits paiements (montants les plus faibles) différés (points rouge), plus la
            plus la valeur Shap est faible et donc plus le modèle prédit que les difficultés de paiement du client diminue.<br>
        10. Plus le montant final du crédit sur la demande précédente (pret) est élevé (points rouge),
            plus la valeur Shap est élevée et donc plus le modèle prédit qu'il aura des difficultés de paiement.<br>
        11. Plus l'age du demandeur (lors du pret) est élevé (points rouge),
            plus la valeur Shap est élevée et donc plus le modèle prédit qu'il aura des difficultés de paiement.<br>
        12. Plus le changement de version des versements d'un mois à l'autre (autre que la carte de crédit) est élevé
            (points de couleur rouge), plus la valeur Shap est faible et donc plus le modèle prédit que
            les difficultés de paiement du client diminue.<br>
        13. Plus le mois du solde par rapport à la date de la demande est élevé (l'information correspond à l'instantané mensuel le plus récent)
            (points de couleur rouge), plus la valeur Shap est faible et donc plus le modèle prédit que les difficultés de paiement du client diminue.<br>
        14. Plus le véhicule du client est recent (points de couleur beige), plus la valeur Shap est faible et
            donc plus le modèle prédit que les difficultés de paiement du client diminue.<br>.""",
                unsafe_allow_html=True)

    # Titre 2
    st.markdown("""
                    <h1>
                    2. Graphique de dépendance</h1>
                    """,
                unsafe_allow_html=True)
    st.write("Nous pouvons obtenir un aperçu plus approfondi de l'effet de chaque fonctionnalité \
                  sur l'ensemble de données avec un graphique de dépendance.")
    st.write("Le dependence plot permet d’analyser les variables deux par deux en suggérant une possiblité d’observation des interactions.\
                  Le scatter plot représente une dépendence entre une variable (en x) et les shapley values (en y) \
                  colorée par la variable la plus corrélées.")

    ################################################################################
    # Création et affichage du sélecteur des variables et des graphs de dépendance #
    ################################################################################
    liste_variables = lecture_X_test_clean().drop(labels="SK_ID_CURR", axis=1).columns.to_list()

    col1, col2 = st.columns(2)  # division de la largeur de la page en 2 pour diminuer la taille du menu déroulant
    with col1:
        ID_var = st.selectbox("*Veuillez sélectionner une variable à l'aide du menu déroulant 👇*",
                              (liste_variables))
        st.write("Vous avez sélectionné la variable :", ID_var)

    fig = plt.figure(figsize=(12, 4))
    ax1 = fig.add_subplot(121)
    shap.dependence_plot(ID_var,
                         calcul_valeurs_shap()[1],
                         lecture_X_test_clean().drop(labels="SK_ID_CURR", axis=1),
                         interaction_index=None,
                         alpha=0.5,
                         x_jitter=0.5,
                         title="Graphique de Dépendance",
                         ax=ax1,
                         show=False)
    ax2 = fig.add_subplot(122)
    shap.dependence_plot(ID_var,
                         calcul_valeurs_shap()[1],
                         lecture_X_test_clean().drop(labels="SK_ID_CURR", axis=1),
                         interaction_index='auto',
                         alpha=0.5,
                         x_jitter=0.5,
                         title="Graphique de Dépendance et Intéraction",
                         ax=ax2,
                         show=False)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)



if options == "Profil Client":
    st.title("Profil Client 🧾")
    # Contenu de Profil_Client.py
    # Ajouter ici les éléments spécifiques de Profil_Client.py

    # Inclusion des polices Google
    st.markdown("""
       <link href="https://fonts.googleapis.com/css2?family=Lobster&family=Montserrat:ital@1&family=Open+Sans&family=Dancing+Script:ital@1&family=Source+Code+Pro&display=swap" rel="stylesheet">
       """, unsafe_allow_html=True)

    # Style CSS pour le contenu
    style = """
       <style>
       body {
           background-color: #F0F0F0;
       }
       h1 {
           font-family: 'Montserrat', sans-serif;
           color: #1E90FF;
           font-size: 2.3em;
           font-style: italic;
           font-weight: 700;
           margin: 0px;
       }
       h2 {
           font-family: 'Open Sans', sans-serif;
           color: #333333;
           font-style: normal;
           margin: 10px 0;
       }
       blockquote {
           font-family: 'Dancing Script', cursive;
           color: #50C878;
           background-color: #FFF8DC;
           padding: 10px;
           border-left: 5px solid #50C878;
           font-style: italic;
       }
       code {
           font-family: 'Source Code Pro', monospace;
           color: #0047AB;
           background-color: #F0F0F0;
           padding: 2px 4px;
           border-radius: 4px;
           font-style: normal;
       }
       </style>
       """

    # Appliquer le style CSS
    st.markdown(style, unsafe_allow_html=True)

    if lecture_X_test_original() is not None and lecture_X_test_clean() is not None:
        # Titre 1
        # style="color:#772b58;font-size:2.3em;font-style:italic;font-weight:700;margin:0px;"
        st.markdown("""
                        <h1 >
                        1. Quel est le profil de votre client ?</h1>
                        """,
                    unsafe_allow_html=True)
        st.write("")

        ##########################################################
        # Création et affichage du sélecteur du numéro de client #
        ##########################################################
        liste_clients = list(lecture_X_test_original()['SK_ID_CURR'])
        col1, col2 = st.columns(2)  # division de la largeur de la page en 2 pour diminuer la taille du menu déroulant
        with col1:
            ID_client = st.selectbox("*Veuillez sélectionner le numéro de votre client à l'aide du menu déroulant 👇*",
                                     (liste_clients))
            st.write("Vous avez sélectionné l'identifiant n° :", ID_client)
        with col2:
            st.write("")

        #################################################
        # Lecture du modèle de prédiction et des scores #
        #################################################
        model_LGBM = pickle.load(
            open("P7_Modelisation_risque_defaut_credit/pickle_files/best_model.pickle", "rb"))
        y_pred_lgbm = model_LGBM.predict(
            lecture_X_test_clean().drop(labels="SK_ID_CURR", axis=1))  # Prédiction de la classe 0 ou 1
        y_pred_lgbm_proba = model_LGBM.predict_proba(
            lecture_X_test_clean().drop(labels="SK_ID_CURR", axis=1))  # Prédiction du % de risque

        # Récupération du score du client
        y_pred_lgbm_proba_df = pd.DataFrame(y_pred_lgbm_proba, columns=['proba_classe_0', 'proba_classe_1'])
        y_pred_lgbm_proba_df = pd.concat([y_pred_lgbm_proba_df['proba_classe_1'],
                                          lecture_X_test_clean()['SK_ID_CURR']], axis=1)
        # st.dataframe(y_pred_lgbm_proba_df)
        score = y_pred_lgbm_proba_df[y_pred_lgbm_proba_df['SK_ID_CURR'] == ID_client]
        score_value = score.proba_classe_1.iloc[0]

        st.write(f"Le client dont l'identifiant est **{ID_client}** a obtenu le score de **{score_value:.1%}**.")
        st.write(f"**Il y a donc un risque de {score_value:.1%} que le client ait des difficultés de paiement.**")


        ########################################################
        # Récupération et affichage des informations du client #
        ########################################################
        #df_info_client = pickle.load(open("P7_Modelisation_risque_defaut_credit/pickle_files/df_info_client.pickle", "rb"))
        df_info_client = pd.read_csv("P7_Modelisation_risque_defaut_credit/pickle_files/df_info_client.csv")
        df_info_client = df_info_client[df_info_client.SK_ID_CURR == ID_client]

        df_pret_client = pickle.load(open("P7_Modelisation_risque_defaut_credit/pickle_files/df_pret_client.pickle", "rb"))
        df_pret_client = df_pret_client[df_pret_client.SK_ID_CURR == ID_client]

        col1, col2 = st.columns(2)
        with col1:
            # Titre H2
            st.markdown("""
                            <h2>
                            Profil socio-économique</h2>
                            """,
                        unsafe_allow_html=True)
            st.write("")
            st.write(f"Sexe de l'individu : **{df_info_client['Sexe'].values[0]}**")
            st.write(f"Situation familiale : **{df_info_client['Statut familial'].values[0]}**")
            st.write(f"Nbre enfants : **{df_info_client['Nbre enfants'].values[0]}**")
            st.write(f"Niveau éducation : **{df_info_client['Niveau éducation'].values[0]}**")
            st.write(f"Type revenu : **{df_info_client['Type revenu'].values[0]}**")
            st.write(f"Ancienneté emploi : **{df_info_client['Ancienneté emploi'].values[0]} ANS**")
            st.write(f"Revenus: **{df_info_client['Revenus ($)'].values[0]} $**")


        with col2:
            # Titre H2
            st.markdown("""
                            <h2>
                            Profil emprunteur</h2>
                            """,
                        unsafe_allow_html=True)
            st.write("")
            st.write(f"Type de prêt : **{df_pret_client['Type de prêt'].values[0]}**")
            st.write(f"Montant du crédit: **{df_pret_client['Montant du crédit ($)'].values[0]} $**")
            st.write(f"Annuités: **{df_pret_client['Annuités ($)'].values[0]}$**")
            st.write(f"Montant du bien: **{df_pret_client['Montant du bien ($)'].values[0]} $**")
            st.write(f"Type de logement : **{df_pret_client['Type de logement'].values[0]}**")



        ###############################################################
        # Comparaison du profil du client à son groupe d'appartenance #
        ###############################################################

        # Titre 1
        st.markdown("""
                        <br>
                        <h1>
                        2. Comparaison du profil du client à celui des clients dont la probabilité de défaut de paiement est proche</h1>
                        """,
                    unsafe_allow_html=True)
        st.write("Pour la définition des groupes de clients, faites défiler la page vers le bas.")

        # Calcul des valeurs Shap
        explainer_shap = shap.TreeExplainer(model_LGBM)
        shap_values = explainer_shap.shap_values(lecture_X_test_clean().drop(labels="SK_ID_CURR", axis=1))
        shap_values_df = pd.DataFrame(data=shap_values[1],
                                      columns=lecture_X_test_clean().drop(labels="SK_ID_CURR", axis=1).columns)

        df_groupes = pd.concat([y_pred_lgbm_proba_df['proba_classe_1'], shap_values_df], axis=1)
        df_groupes['typologie_clients'] = pd.qcut(df_groupes.proba_classe_1,
                                                  q=5,
                                                  precision=1,
                                                  labels=['20%_et_moins',
                                                          '21%_30%',
                                                          '31%_40%',
                                                          '41%_60%',
                                                          '61%_et_plus'])

        # Titre H2

        st.markdown("""
                        <h2>
                        Comparaison de “la trajectoire” prise par la prédiction du client à celles des groupes de Clients</h2>
                        """,
                    unsafe_allow_html=True)
        st.write("")

        # Moyenne des variables par classe
        df_groupes_mean = df_groupes.groupby(['typologie_clients']).mean()
        df_groupes_mean = df_groupes_mean.rename_axis('typologie_clients').reset_index()
        df_groupes_mean["index"] = [1, 2, 3, 4, 5]
        df_groupes_mean.set_index('index', inplace=True)

        # récupération de l'index correspondant à l'identifiant du client
        idx = int(lecture_X_test_clean()[lecture_X_test_clean()['SK_ID_CURR'] == ID_client].index[0])

        # dataframe avec shap values du client et des 5 groupes de clients
        comparaison_client_groupe = pd.concat([df_groupes[df_groupes.index == idx],
                                               df_groupes_mean],
                                              axis=0)
        comparaison_client_groupe['typologie_clients'] = np.where(comparaison_client_groupe.index == idx,
                                                                  lecture_X_test_clean().iloc[idx, 0],
                                                                  comparaison_client_groupe['typologie_clients'])
        # transformation en array
        nmp = comparaison_client_groupe.drop(
            labels=['typologie_clients', "proba_classe_1"], axis=1).to_numpy()

        fig = plt.figure(figsize=(8, 20))
        st_shap(shap.decision_plot(explainer_shap.expected_value[1],
                                   nmp,
                                   feature_names=comparaison_client_groupe.drop(
                                       labels=['typologie_clients', "proba_classe_1"], axis=1).columns.to_list(),
                                   feature_order='importance',
                                   highlight=0,
                                   legend_labels=['Client', '20%_et_moins', '21%_30%', '31%_40%', '41%_60%',
                                                  '61%_et_plus'],
                                   plot_color='inferno_r',
                                   legend_location='center right',
                                   feature_display_range=slice(None, -57, -1),
                                   link='logit'))

        # Titre H2
        st.markdown("""
                        <h2>
                        Constitution de groupes de clients selon leur probabilité de défaut de paiement</h2>
                        """,
                    unsafe_allow_html=True)
        st.write("")

        col1, col2 = st.columns(2)
        with col1:
            fig1, ax1 = plt.subplots(figsize=(8, 6))
            plot_countplot(df=df_groupes,
                           col='typologie_clients',
                           order=False,
                           palette='cubehelix', ax=ax1, orient='v', size_labels=12)
            plt.title("Regroupement des Clients selon leur Probabilité de Défaut de Paiement\n",
                      loc="center", fontsize=16, fontstyle='italic', fontname='Roboto Condensed')
            fig1.tight_layout()
            st.pyplot(fig1)
        with col2:
            fig2, ax2 = plt.subplots(figsize=(8, 6))

            plot_aggregation(df=df_groupes,
                             group_col='typologie_clients',
                             value_col='proba_classe_1',
                             aggreg='mean',
                             palette="cubehelix", ax=ax2, orient='v', size_labels=12)

            plt.title("Probabilité Moyenne de Défaut de Paiement par Groupe de Clients\n",
                      loc="center", fontsize=18, fontstyle='italic')
            fig2.tight_layout()
            st.pyplot(fig2)


if options == "Score Client":
    st.title("Score Client 🏆")
    # Contenu de Score_Client.py

    # Inclusion des polices Google
    st.markdown("""
           <link href="https://fonts.googleapis.com/css2?family=Lobster&family=Montserrat:ital@1&family=Open+Sans&family=Dancing+Script:ital@1&family=Source+Code+Pro&display=swap" rel="stylesheet">
           """, unsafe_allow_html=True)

    # Style CSS pour le contenu
    style = """
           <style>
           body {
               background-color: #F0F0F0;
           }
           h1 {
               font-family: 'Montserrat', sans-serif;
               color: #1E90FF;
               font-size: 2.3em;
               font-style: italic;
               font-weight: 700;
               margin: 0px;
           }
           h2 {
               font-family: 'Open Sans', sans-serif;
               color: #333333;
               font-style: normal;
               margin: 10px 0;
           }
           blockquote {
               font-family: 'Dancing Script', cursive;
               color: #50C878;
               background-color: #FFF8DC;
               padding: 10px;
               border-left: 5px solid #50C878;
               font-style: italic;
           }
           code {
               font-family: 'Source Code Pro', monospace;
               color: #0047AB;
               background-color: #F0F0F0;
               padding: 2px 4px;
               border-radius: 4px;
               font-style: normal;
           }
           </style>
           """

    # Appliquer le style CSS
    st.markdown(style, unsafe_allow_html=True)


    # Lecture des fichiers de données
    X_test_original = dd.read_parquet("P7_Modelisation_risque_defaut_credit/test_preprocess.parquet")
    X_test_clean = dd.read_parquet("P7_Modelisation_risque_defaut_credit/test_set.parquet")
    description_variables = pd.read_csv("P7_Modelisation_risque_defaut_credit/description_variable.csv", sep=";")

    if X_test_original is not None and X_test_clean is not None:
        # Sélecteur d'identifiant de client
        liste_clients = list(X_test_original['SK_ID_CURR'])
        ID_client = st.selectbox("Veuillez sélectionner le numéro de votre client 👇", (liste_clients))
        st.write("Vous avez sélectionné l'identifiant n° :", ID_client)

        # Chargement du modèle et prédiction
        # model_LGBM = pickle.load(open("P7_Modelisation_risque_defaut_credit/pickle_files/best_model.pickle", "rb"))
        # y_pred_lgbm = model_LGBM.predict(X_test_clean.drop(labels="SK_ID_CURR", axis=1))
        # y_pred_lgbm_proba = model_LGBM.predict_proba(X_test_clean.drop(labels="SK_ID_CURR", axis=1))

        # # Récupération du score du client
        # y_pred_lgbm_proba_df = pd.DataFrame(y_pred_lgbm_proba, columns=['proba_classe_0', 'proba_classe_1'])
        # y_pred_lgbm_proba_df = pd.concat([y_pred_lgbm_proba_df['proba_classe_1'], X_test_clean['SK_ID_CURR']], axis=1)
        # score = y_pred_lgbm_proba_df[y_pred_lgbm_proba_df['SK_ID_CURR'] == ID_client]
        # score_value = round(score.proba_classe_1.iloc[0] * 100, 2)

        # Charger le modèle
        model_LGBM = pickle.load(open("P7_Modelisation_risque_defaut_credit/pickle_files/best_model.pickle", "rb"))
        
        # Prédictions
        X_test_clean_pandas = X_test_clean.compute() if hasattr(X_test_clean, 'compute') else X_test_clean
        y_pred_lgbm = model_LGBM.predict(X_test_clean_pandas.drop(labels="SK_ID_CURR", axis=1))
        y_pred_lgbm_proba = model_LGBM.predict_proba(X_test_clean_pandas.drop(labels="SK_ID_CURR", axis=1))
        
        # Récupération du score du client
        y_pred_lgbm_proba_df = pd.DataFrame(y_pred_lgbm_proba, columns=['proba_classe_0', 'proba_classe_1'])
        
        # Concaténation
        y_pred_lgbm_proba_df = pd.concat([y_pred_lgbm_proba_df['proba_classe_1'], X_test_clean_pandas['SK_ID_CURR']], axis=1)
        
        # Renommer les colonnes pour plus de clarté
        y_pred_lgbm_proba_df.columns = ['proba_classe_1', 'SK_ID_CURR']
        score = y_pred_lgbm_proba_df[y_pred_lgbm_proba_df['SK_ID_CURR'] == ID_client]
        score_value = round(score.proba_classe_1.iloc[0] * 100, 2)

        # Récupération de la décision
        # y_pred_lgbm_df = pd.DataFrame(y_pred_lgbm, columns=['prediction'])
        # y_pred_lgbm_df = pd.concat([y_pred_lgbm_df, X_test_clean['SK_ID_CURR']], axis=1)
        # y_pred_lgbm_df['client'] = np.where(y_pred_lgbm_df.prediction == 1, "non solvable", "solvable")
        # y_pred_lgbm_df['decision'] = np.where(y_pred_lgbm_df.prediction == 1, "refuser", "accorder")
        # solvabilite = y_pred_lgbm_df.loc[y_pred_lgbm_df['SK_ID_CURR'] == ID_client, "client"].values
        # decision = y_pred_lgbm_df.loc[y_pred_lgbm_df['SK_ID_CURR'] == ID_client, "decision"].values

         #Convertir X_test_clean en DataFrame Pandas si nécessaire
        X_test_clean_pandas = X_test_clean.compute() if hasattr(X_test_clean, 'compute') else X_test_clean
        
        # Prédictions
        y_pred_lgbm = model_LGBM.predict(X_test_clean_pandas.drop(labels="SK_ID_CURR", axis=1))
        
        # Création du DataFrame pour les prédictions
        y_pred_lgbm_df = pd.DataFrame(y_pred_lgbm, columns=['prediction'])
        
        # Concaténation avec SK_ID_CURR
        y_pred_lgbm_df = pd.concat([y_pred_lgbm_df, X_test_clean_pandas['SK_ID_CURR']], axis=1)
        
        # Ajout des colonnes client et decision
        y_pred_lgbm_df['client'] = np.where(y_pred_lgbm_df.prediction == 1, "non solvable", "solvable")
        y_pred_lgbm_df['decision'] = np.where(y_pred_lgbm_df.prediction == 1, "refuser", "accorder")
        
        # Extraction des informations pour un client spécifique
        solvabilite = y_pred_lgbm_df.loc[y_pred_lgbm_df['SK_ID_CURR'] == ID_client, "client"].values
        decision = y_pred_lgbm_df.loc[y_pred_lgbm_df['SK_ID_CURR'] == ID_client, "decision"].values

        # Affichage du score et du graphique de gauge
        col1, col2 = st.columns(2)
        with col2:
            st.write(f"Le client dont l'identifiant est **{ID_client}** a obtenu le score de **{score_value:.1f}%**.")
            st.write(f"**Il y a donc un risque de {score_value:.1f}% que le client ait des difficultés de paiement.**")
            st.write(f"Le client est donc considéré par *'Prêt à dépenser'* comme **{solvabilite[0]}** et décide de lui **{decision[0]}** le crédit.")
        with col1:
            fig = go.Figure(go.Indicator(
                domain={'x': [0, 1], 'y': [0, 1]},
                value=float(score_value),
                mode="gauge+number+delta",
                title={'text': "Score du client", 'font': {'size': 24}},
                delta={'reference': 35.2, 'increasing': {'color': "#3b203e"}},
                gauge={'axis': {'range': [None, 100], 'tickwidth': 3, 'tickcolor': 'darkblue'},
                       'bar': {'color': 'white', 'thickness': 0.3},
                       'bgcolor': 'white',
                       'borderwidth': 1,
                       'bordercolor': 'gray',
                       'steps': [{'range': [0, 20], 'color': '#e8af92'},
                                 {'range': [20, 40], 'color': '#db6e59'},
                                 {'range': [40, 60], 'color': '#b43058'},
                                 {'range': [60, 80], 'color': '#772b58'},
                                 {'range': [80, 100], 'color': '#3b203e'}],
                       'threshold': {'line': {'color': 'white', 'width': 8},
                                     'thickness': 0.8,
                                     'value': 35.2}}))

            fig.update_layout(paper_bgcolor='white',
                              height=400, width=500,
                              font={'color': '#772b58', 'family': 'Roboto Condensed'},
                              margin=dict(l=30, r=30, b=5, t=5))
            st.plotly_chart(fig, use_container_width=True)

        # Explication de la prédiction
        st.write("### Explication de la Prédiction")
        explainer_shap = shap.TreeExplainer(model_LGBM)
        shap_values = explainer_shap.shap_values(lecture_X_test_clean().drop(labels="SK_ID_CURR", axis=1))
        idx = int(lecture_X_test_clean()[lecture_X_test_clean()['SK_ID_CURR'] == ID_client].index[0])

        st.write("Le graphique suivant appelé `force-plot` permet de voir où se place la prédiction (f(x)) par rapport à la `base value`.")
        st.write("Nous observons également quelles sont les variables qui augmentent la probabilité du client d'être en défaut de paiement (en rouge) et celles qui la diminuent (en bleu), ainsi que l’amplitude de cet impact.")
        st_shap(shap.force_plot(explainer_shap.expected_value[1], shap_values[1][idx, :], lecture_X_test_clean().drop(labels="SK_ID_CURR", axis=1).iloc[idx, :], link='logit', figsize=(20, 8), ordering_keys=True, text_rotation=0, contribution_threshold=0.05))

        st.write("Le graphique ci-dessous appelé `decision_plot` est une autre manière de comprendre la prédiction. Comme pour le graphique précédent, il met en évidence l’amplitude et la nature de l’impact de chaque variable avec sa quantification ainsi que leur ordre d’importance. Mais surtout il permet d'observer “la trajectoire” prise par la prédiction du client pour chacune des valeurs des variables affichées.")
        st.write("Seules les 15 variables explicatives les plus importantes sont affichées par ordre décroissant.")
        st_shap(shap.decision_plot(explainer_shap.expected_value[1], shap_values[1][idx, :], lecture_X_test_clean().drop(labels="SK_ID_CURR", axis=1).iloc[idx, :], feature_names=lecture_X_test_clean().drop(labels="SK_ID_CURR", axis=1).columns.to_list(), feature_order='importance', feature_display_range=slice(None, -16, -1), link='logit'))


# Footer
st.markdown("""
    <hr>
    <footer style="text-align:center;">
        <p>© 2024 Prêt à dépenser. Tous droits réservés.</p>
    </footer>
    """,
    unsafe_allow_html=True)
