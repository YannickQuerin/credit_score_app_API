#!/usr/bin/env python
# coding: utf-8

# In[ ]:

# ====================================================================
# Outils Fonctions data -  projet 7 Openclassrooms
# Version : 0.0.0 - CRE LR 16/07/2021
# ====================================================================

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import missingno as msno 
import seaborn as sns
from IPython.display import display
from wordcloud import WordCloud
from statsmodels.graphics.gofplots import qqplot
import re
import pandas as pd
import numpy as np
import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
from sklearn.utils import check_random_state
from sklearn.inspection import permutation_importance
import eli5
from eli5.sklearn import PermutationImportance
import matplotlib.pyplot as plt
from IPython.display import display
from sklearn.feature_selection import RFECV
from pprint import pprint


# Version:
__version__ = '0.0.0'

# In[ ]:

# --------------------------------------------------------------------
# -- DESCRIPTION DES VARIABLES STATISTIQUES
# --------------------------------------------------------------------


def stat_descriptives(data,list_var):
    '''
    Fonction prenant un dataframe en entrée et retourne les variables, avec ses statistiques
    
    '''

    df = pd.DataFrame(columns=['Variable name', 'Mean', 'Median', 'Skew', 'Kurtosis', \
     'Variance', 'Stdev', 'min','25%','50%','75%','max'])
    
    for col in list_var:
        var_type = data[col].dtypes
        if var_type != 'object':       
            df = df.append(pd.DataFrame([[col, data[col].mean(),data[col].median(), \
            data[col].skew(),data[col].kurtosis(),data[col].var(ddof=0),data[col].std(ddof=0), \
            data[col].min(),data[col].quantile(0.25),data[col].quantile(0.5),data[col].quantile(0.75), \
            data[col].max()]], columns=['Variable name', 'Mean', 'Median', 'Skew', 'Kurtosis', \
            'Variance', 'Stdev', 'min','25%','50%','75%','max']))
    
    df = df.reset_index(drop=True)
    return df

########

def resume_datasets(dataframes, noms):
    '''
    Résumé rapide du contenu, nom, nombre de lignes, nombre de variables,
    valeurs manquantes, variable avec valeurs manquantes... pour plusieurs
    dataframes transmis en paramètres.
    Parameters
    ----------
    dataframes : les dataframes à résumer, obligatoire.
    noms : les noms des dataframes transmis, obligatoire.
    Returns
    -------
    df_resume : dataframe de résumé
    '''
    print(f'Les données se décomposent en {len(dataframes)} fichier(s).')

    # Creating a DataFrame with useful information about all datasets
    df_resume = pd.DataFrame({})
    df_resume['Jeu de données'] = noms
    df_resume['Nb lignes'] = [df.shape[0] for df in dataframes]
    df_resume['Nb variables'] = [df.shape[1] for df in dataframes]
    df_resume['Nb nan'] = [df.isnull().sum().sum() for df in dataframes]
    df_resume['% nan'] = [(df.isnull().sum().sum() * 100 / np.product(df.shape)) for df in dataframes]
    df_resume['Nb var avec nan'] = [len(
        [col for col, null in df.isnull().sum().items() if null > 0]) for df in dataframes]
    df_resume['Var avec nan'] = [', '.join(
        [col for col, null in df.isnull().sum().items() if null > 0]) for df in dataframes]

    return df_resume.style.hide_index()

########

def plot_type_analysis(data, contract_type_column):
    """
    Fonction pour analyser la variable NAME_CONTRACT_TYPE
    et créer des graphiques représentant le type de prêt en fonction de la target
    et la représentation du type de prêt.

    :param data: DataFrame contenant les données
    :param contract_type_column: Nom de la colonne contenant le type de prêt
    """

    # Analyse de la variable NAME_CONTRACT_TYPE
    print(data[contract_type_column].value_counts())

    # Création des graphiques
    fig, axs = plt.subplots(ncols=2, figsize=(16,8))

    # Graphique 1 : Représentation du type de prêt en fonction de la target
    sns.countplot(data=data, x=data['TARGET'], hue=data[contract_type_column], ax=axs[0], palette="pastel")
    axs[0].set_title('Représentation du type de prêt en fonction de la target', fontsize=16)
    axs[0].set_xlabel('Target', fontsize=14)
    axs[0].set_ylabel('Nature du prêt', fontsize=14)
    axs[0].legend(title='Nature du prêt', fontsize=12, title_fontsize=12)

    # Graphique 2 : Représentation de la situation familiale
    sns.countplot(data=data, x=data[contract_type_column], ax=axs[1], palette="pastel")
    axs[1].set_title('Représentation du type de prêt', fontsize=16)
    axs[1].set_xlabel('Nature du prêt', fontsize=14)
    axs[1].set_ylabel('Nombre de clients', fontsize=14)

    plt.tight_layout()
    plt.show()


########

def null_var(df, tx_seuil=50):
    null_tx = ((df.isnull().sum() / df.shape[0])*100).sort_values(ascending=False).reset_index()
    null_tx.columns = ['Variable','Taux_de_Null']
    high_null_tx = null_tx[null_tx.Taux_de_Null >= tx_seuil]
    return high_null_tx

#

def fill_var(df, tx_min, tx_max):
    fill_tx = (100 - (df.isnull().sum() / df.shape[0])*100).sort_values(ascending=False).reset_index()
    fill_tx.columns = ['Variable','Taux_de_remplissage']
    high_fill_tx = fill_tx[(fill_tx.Taux_de_remplissage >= tx_min) & (fill_tx.Taux_de_remplissage <= tx_max)]
    return high_fill_tx

#

# --------------------------------------------------------------------
# -- DESCRIPTION DES VARIABLES
# --------------------------------------------------------------------


def  get_nutri_col(data,cols_suppr=False):
        columns_nutri = ['energy_100g',
                             'nutrition_score_fr_100g',
                             'saturated_fat_100g',
                             'sugars_100g',
                             'proteins_100g',
                             'fat_100g',
                             'carbohydrates_100g',
                             'salt_100g',
                             'fiber_100g']
        if cols_suppr:                      
            return data[columns_nutri].drop(cols_suppr,axis=1).columns.to_list()
        else:
            return data[columns_nutri].columns.to_list()
        
#

def rempl_caracteres(data, anc_car, nouv_car):
    """
    Remplacer les caractères avant par les caractères après
    dans le nom des variables du dataframe
    Parameters
    ----------
    @param IN : dataframe : DataFrame, obligatoire
                car_avant : le caractère à remplacer
                car_apres : le caractère de remplacement
    @param OUT : dataframe modifié
    """
    # traces des variables à renommer
    cols_a_renom = data.columns[data.columns.str.contains(
        anc_car)]
    print(f'{len(cols_a_renom)} variables renommées \
          \'{anc_car}\' en \'{nouv_car}\' : \n\n {cols_a_renom.tolist()}')

    return data.columns.str.replace(anc_car, nouv_car)
        

# In[ ]:

def affichage_types_var(df_type, types, type_par_var, graph):
    """ Permet un aperçu du type des variables
    Parameters
    ----------
    @param IN : df_work : dataframe, obligatoire
                types : Si True lance dtypes, obligatoire
                type_par_var : Si True affiche tableau des types de
                               chaque variable, obligatoire
                graph : Si True affiche pieplot de répartition des types
    @param OUT :None.
    """

    if types:
        # 1. Type des variables
        print("-------------------------------------------------------------")
        print("Type de variable pour chacune des variables\n")
        display(df_type.dtypes)

    if type_par_var:
        # 2. Compter les types de variables
        #print("Répartition des types de variable\n")
        values = df_type.dtypes.value_counts()
        nb_tot = values.sum()
        percentage = round((100 * values / nb_tot), 2)
        table = pd.concat([values, percentage], axis=1)
        table.columns = [
            'Nombre par type de variable',
            '% des types de variable']
        display(table[table['Nombre par type de variable'] != 0]
                .sort_values('% des types de variable', ascending=False)
                .style.background_gradient('seismic'))

    if graph:
        # 3. Schéma des types de variable
        # print("\n----------------------------------------------------------")
        #print("Répartition schématique des types de variable \n")
        # Répartition des types de variables
        plt.figure(figsize=(5,5))
        df_type.dtypes.value_counts().plot.pie( autopct='%.0f%%', pctdistance=0.85, radius=1.2)
        #plt.pie(df_type.dtypes.value_counts(), labels = df_type.dtypes.unique(), autopct='%.0f%%', pctdistance=0.85, radius=1.2)
        centre_circle = plt.Circle((0, 0), 0.8, fc='white')
        fig = plt.gcf()
        fig.gca().add_artist(centre_circle)
        plt.title(label="Repartiton des types de variables",loc="left", fontstyle='italic')
        plt.show()

        
#

def get_val_manq(df_type, pourcentage, affiche_val_manq):
    """Indicateurs sur les variables manquantes
       @param in : df_work dataframe obligatoire
                   pourcentage : boolean si True affiche le nombre heatmap
                   affiche_heatmap : boolean si True affiche la heatmap
       @param out : none
    """

    # 1. Nombre de valeurs manquantes totales
    nb_nan_tot = df_type.isna().sum().sum()
    nb_donnees_tot = np.product(df_type.shape)
    pourc_nan_tot = round((nb_nan_tot / nb_donnees_tot) * 100, 2)
    print(
        f'Valeurs manquantes :{nb_nan_tot} NaN pour {nb_donnees_tot} données ({pourc_nan_tot} %)')

    if pourcentage:
        print("-------------------------------------------------------------")
        print("Nombre et pourcentage de valeurs manquantes par variable\n")
        # 2. Visualisation du nombre et du pourcentage de valeurs manquantes
        # par variable
        values = df_type.isnull().sum()
        percentage = 100 * values / len(df_type)
        table = pd.concat([values, percentage.round(2)], axis=1)
        table.columns = [
            'Nombres de valeurs manquantes',
            '% de valeurs manquantes']
        display(table[table['Nombres de valeurs manquantes'] != 0]
                .sort_values('% de valeurs manquantes', ascending=False))

    if affiche_val_manq:
        print("-------------------------------------------------------------")
        print("Heatmap de visualisation des valeurs manquantes")
        # 3. Heatmap de visualisation des valeurs manquantes
        msno.matrix(df_type)

#

def detail_type_var(data, type_var='all'):
    """
    Retourne la description des variables qualitatives/quantitatives
    ou toutes les variables du dataframe transmis :
    type, nombre de nan, % de nan et desc
    Parameters
    ----------
    @param IN : dataframe : DataFrame, obligatoire
                type_var = 'all' ==> tous les types de variables
                           'cat' ==> variables catégorielles
                           'num' ==> variables quantitative
    @param OUT : dataframe de description des variables
    """
    n_df = data.shape[0]

    if type_var == 'num':
        det_var = data.describe()
    elif type_var == 'cat':
        det_var = data.describe(exclude=[np.number])
    else:
        det_var = data.describe(include='all')
    
    det_type = pd.DataFrame(data[det_var.columns].dtypes, columns=['type']).T
    nb_nan = n_df - det_var.loc['count'].T
    pourcentage_nan = nb_nan * 100 / n_df
    det_nan = pd.DataFrame([nb_nan, pourcentage_nan], index=['nb_nan', '%_nan'])
    det_var = pd.concat([det_type, det_nan, det_var])
    
    return det_var



#
        
# --------------------------------------------------------------------
# -- SUPRESSION VARIABLES POUR UN TAUX DE NAN (%)
# --------------------------------------------------------------------
def clean_nan(data, taux_nan):
#     """
#     Supprime les variables à partir d'un taux en % de nan.
#     Affiche les variables supprimées et les variables conservées
#     ----------
#     @param IN : dataframe : DataFrame, obligatoire
#                 seuil : on conserve toutes les variables dont taux de nan <80%
#                         entre 0 et 100, integer
#     @param OUT : dataframe modifié
#     """
    qty_nan = round((data.isna().sum() / data.shape[0]) * 100, 2)
    cols = data.columns.tolist()
    
    # Conservation seulement des variables avec valeurs manquantes >= 80%
    cols_conservées = qty_nan[qty_nan.values < taux_nan].index.tolist()
    
    cols_suppr = [col for col in cols if col not in cols_conservées]

    data = data[qty_nan[qty_nan.values < taux_nan].index.tolist()]

    print(f'Liste des variables éliminées :\n{cols_suppr}\n')

    print(f'Liste des variables conservées :\n{cols_conservées}')

    return data
    

# In[ ]:

def trace_dispersion_boxplot_qqplot(dataframe, variable, titre, unite):
    """
    Suivi des dipsersions : boxplot
    ----------
    @param IN : dataframe : DataFrame, obligatoire
                variable : colonne dont on veut voir les outliers
                titre :titre pour les graphiques (str)
                unite : unité pour ylabel boxplot (str)
    @param OUT :None
    """
    # Boxplot + qqplot
    fig = plt.figure(figsize=(15, 6))

    data = dataframe[variable]

    ax1 = fig.add_subplot(1, 2, 1)
    box = sns.boxplot(data=data, color='violet', ax=ax1)
    box.set(ylabel=unite)

    plt.grid(False)

    ax2 = fig.add_subplot(1, 2, 2)
    ax2 = qqplot(data,
                 line='r',
                 **{'markersize': 5,
                    'mec': 'k',
                    'color': 'violet'},
                 ax=ax2)
    plt.grid(False)

    fig.suptitle(titre, fontweight='bold', size=14)
    plt.show()

# In[ ]:

def plot_var_filling (df, tx_min, tx_max, graph, axe, col):
    
    if graph:
            filling_var = fill_var(df, tx_min, tx_max)

            font_title = {'family': 'serif',
                          'color':  '#114b98',
                          'weight': 'bold',
                          'size': 18,
                         }
            
        
            sns.set(font_scale=1.2)
            sns.barplot(ax = axe, x="Taux_de_remplissage", y="Variable", data=filling_var, color = col)

    

# In[ ]:

def plot_columns_boxplots(data, columns=[], ncols=2, color="goldenrod"):
    if len(columns) == 0:
        columns = data.columns.values
        
    if len(columns) == 1:
        plt.figure(figsize=(9,3))
        sns.boxplot(x=data[columns[0]], color=color)
        
    else:
        fig, axs = plt.subplots(figsize=(20,20), ncols=ncols, nrows=math.ceil(len(columns) / ncols))
        for index, column in enumerate(columns):
            row_index = math.floor(index / ncols)
            col_index = index % ncols
            sns.boxplot(x=data[column], ax=axs[row_index][col_index], color=color)


# In[ ]:


# --------------------------------------------------------------------
# -- HISTPLOT BOXPLOT QQPLOT
# --------------------------------------------------------------------


def trace_histplot_boxplot_qqplot(dataframe, var):
    """
    Suivi des dipsersions : boxplot et qqplot
    ----------
    @param IN : dataframe : DataFrame, obligatoire
                var : colonne dont on veut voir les outliers
    @param OUT :None
    """
    # Boxplot + qqplot
    fig = plt.figure(figsize=(15, 6))
    fig.subplots_adjust(hspace=0.1, wspace=0.4)
    fig.suptitle('Distribution de ' + str(var), fontsize=16)

    data = dataframe[var]

    ax0 = fig.add_subplot(1, 3, 1)
    sns.histplot(data, kde=True, color='goldenrod', ax=ax0)
    plt.xticks(rotation=60)

    ax1 = fig.add_subplot(1, 3, 2)
    sns.boxplot(data=data, color='goldenrod', ax=ax1)
    plt.grid(False)

    ax2 = fig.add_subplot(1, 3, 3)
    qqplot(data,
           line='r',
           **{'markersize': 5,
              'mec': 'k',
              'color': 'orange'},
           ax=ax2)
    plt.grid(False)
    plt.show()


def trace_multi_histplot_boxplot_qqplot(dataframe, liste_var):
    """
    Suivi des dipsersions : boxplot et qqplot
    ----------
    @param IN : dataframe : DataFrame, obligatoire
                liste_var : colonnes dont on veut voir les outliers
    @param OUT :None
    """
    for col in liste_var:
        trace_histplot_boxplot_qqplot(dataframe, col)


def trace_histplot(
        dataframe,
        variable,
        col,
        titre,
        xlabel,
        xlim_bas,
        xlim_haut,
        ylim_bas,
        ylim_haut,
        kde=True,
        mean_median_mode=True,
        mean_median_zoom=False):
    """
    Histplot pour les variables quantitatives général + histplot
    ----------
    @param IN : dataframe : DataFrame, obligatoire
                variable : colonne dont on veut voir les histplot
                titre : titre du graphique (str)
                xlabel:légende des abscisses
                xlim_bas : limite du zoom supérieur bas(int)
                xlim_haut : limite du zoom supérieur haut(int)
                ylim_bas : limite du zoom inférieur bas(int)
                ylim_haut : limite du zoom inférieur haut(int)
                kde : boolean pour tracer la distribution normale
                mean_median_mode : boolean pour tracer la moyenne, médiane et mode
                mean_median_zoom : boolean pour tracer la moyenne et médiane sur le graphique zoomé
    @param OUT :None
    """
    # Distplot général + zoom
    
    fig = plt.figure(figsize=(15, 8))
    fig.suptitle(titre, fontsize=20, y=1.03)
    data = dataframe[variable]
    
    ax = fig.add_subplot(2, 1, 1)
    ax = sns.boxplot(x=data, color=col)
    ax.set_xlim(xlim_bas, xlim_haut)
    ax.set_ylim(ylim_bas, ylim_haut)
    plt.grid(False)
    plt.xticks([], [])
    

    ax = fig.add_subplot(2, 1, 2)
    ax = sns.histplot(data, kde=kde, color=col)

    if mean_median_mode:
        ax.vlines(data.mean(), *ax.get_ylim(), color='red', ls='-', lw=1.5)
        ax.vlines(
            data.median(),
            *ax.get_ylim(),
            color='green',
            ls='-.',
            lw=1.5)
        ax.vlines(
            data.mode()[0],
            *ax.get_ylim(),
            color='goldenrod',
            ls='--',
            lw=1.5)
    ax.legend(['mode', 'mean', 'median'])
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel('Nombre de produits', fontsize=12)
    plt.grid(False)
    
      
    plt.show()        
        
        

def trace_pieplot(dataframe, variable, titre, legende, liste_colors):
    """
    Suivi des dipsersions : bosplot et qqplot
    ----------
    @param IN : dataframe : DataFrame, obligatoire
                variable : colonne dont on veut voir les outliers (str)
                titre :titre pour les graphiques (str)
                legende : titre de la légende
                liste_colors : liste des couleurs
    @param OUT :None
    """

    plt.figure(figsize=(7, 7))
    plt.title(titre, size=16)
    nb_par_var = dataframe[variable].sort_values().value_counts()
    nb_par_var = nb_par_var.loc[sorted(nb_par_var.index)]
    explode = [0.1]
    for i in range(len(nb_par_var) - 1):
        explode.append(0)
    wedges, texts, autotexts = plt.pie(
        nb_par_var, labels=nb_par_var.index, autopct='%1.1f%%', colors=liste_colors, textprops={
            'fontsize': 16, 'color': 'black', 'backgroundcolor': 'w'}, explode=explode)
    axes = plt.gca()
    axes.legend(
        wedges,
        nb_par_var.index,
        title=legende,
        loc='center right',
        fontsize=14,
        bbox_to_anchor=(
            1,
            0,
            0.5,
            1))
    plt.show()

#    

def aff_eboulis_plot(pca):
    tx_var_exp = pca.explained_variance_ratio_
    scree = tx_var_exp * 100
    plt.bar(np.arange(len(scree)) + 1, scree, color='SteelBlue')
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    ax2.plot(np.arange(len(scree)) + 1, scree.cumsum(), c='green', marker='o')
    ax2.set_ylabel('Taux cumulatif de l\'inertie')
    ax1.set_xlabel('Rang de l\'axe d\'inertie')
    ax1.set_ylabel('Pourcentage d\'inertie')
    for i, p in enumerate(ax1.patches):
        ax1.text(
            p.get_width() /
            5 +
            p.get_x(),
            p.get_height() +
            p.get_y() +
            0.3,
            '{:.0f}%'.format(
                tx_var_exp[i] *
                100),
            fontsize=8,
            color='k')
    plt.title('Eboulis des valeurs propres')
    plt.gcf().set_size_inches(8, 4)
    plt.grid(False)
    plt.show(block=False)

 
    
    
def affiche_cercle(pcs, n_comp, pca, axis_ranks, labels=None, label_rotation=0, lims=None):
    for d1, d2 in axis_ranks: # On affiche les 3 premiers plans factoriels, donc les 6 premières composantes
        if d2 < n_comp:
 
            # initialisation de la figure
            fig, ax = plt.subplots(figsize=(7,6))
 
            # détermination des limites du graphique
            if lims is not None :
                xmin, xmax, ymin, ymax = lims
            elif pcs.shape[1] < 30 :
                xmin, xmax, ymin, ymax = -1, 1, -1, 1
            else :
                xmin, xmax, ymin, ymax = min(pcs[d1,:]), max(pcs[d1,:]), min(pcs[d2,:]), max(pcs[d2,:])
 
            # affichage des flèches
            # s'il y a plus de 30 flèches, on n'affiche pas le triangle à leur extrémité
            if pcs.shape[1] < 30 :
                plt.quiver(np.zeros(pcs.shape[1]), np.zeros(pcs.shape[1]),
                   pcs[d1,:], pcs[d2,:],
                   angles='xy', scale_units='xy', scale=1, color="black")
                # (voir la doc : https://matplotlib.org/api/_as_gen/matplotlib.pyplot.quiver.html)
            else:
                lines = [[[0,0],[x,y]] for x,y in pcs[[d1,d2]].T]
                ax.add_collection(LineCollection(lines, axes=ax, alpha=.1, color='black'))
             
            # affichage des noms des variables 
            if labels is not None: 
                for i,(x, y) in enumerate(pcs[[d1,d2]].T):
                    if x >= xmin and x <= xmax and y >= ymin and y <= ymax :
                        plt.text(x, y, labels[i], fontsize='14', ha='center', va='center', rotation=label_rotation, color="blue", alpha=0.5)
             
            # affichage du cercle
            circle = plt.Circle((0,0), 1, facecolor='none', edgecolor='b')
            plt.gca().add_artist(circle)
 
            # définition des limites du graphique
            plt.xlim(xmin, xmax)
            plt.ylim(ymin, ymax)
         
            # affichage des lignes horizontales et verticales
            plt.plot([-1, 1], [0, 0], color='grey', ls='--')
            plt.plot([0, 0], [-1, 1], color='grey', ls='--')
 
            # nom des axes, avec le pourcentage d'inertie expliqué
            plt.xlabel('F{} ({}%)'.format(d1+1, round(100*pca.explained_variance_ratio_[d1],1)))
            plt.ylabel('F{} ({}%)'.format(d2+1, round(100*pca.explained_variance_ratio_[d2],1)))
 
            plt.title("Cercle des corrélations (F{} et F{})".format(d1+1, d2+1))
            plt.show(block=False)

###


# --------------------------------------------------------------------
# -- AFFICHE LE PLAN FACTORIEL
# --------------------------------------------------------------------

def display_factorial_planes(X_projected, n_comp, pca, axis_ranks, labels=None, alpha=1, illustrative_var=None):
    for d1,d2 in axis_ranks:
        if d2 < n_comp:
 
            # initialisation de la figure       
            fig = plt.figure(figsize=(7,6))
        
            # affichage des points
            if illustrative_var is None:
                plt.scatter(X_projected[:, d1], X_projected[:, d2], alpha=alpha)
            else:
                illustrative_var = np.array(illustrative_var)
                for value in np.unique(illustrative_var):
                    selected = np.where(illustrative_var == value)
                    plt.scatter(X_projected[selected, d1], X_projected[selected, d2], alpha=alpha, label=value)
                plt.legend()

            # affichage des labels des points
            if labels is not None:
                for i,(x,y) in enumerate(X_projected[:,[d1,d2]]):
                    plt.text(x, y, labels[i],
                              fontsize='14', ha='center',va='center') 
                
            # détermination des limites du graphique
            boundary = np.max(np.abs(X_projected[:, [d1,d2]])) * 1.1
            plt.xlim([-boundary,boundary])
            plt.ylim([-boundary,boundary])
        
            # affichage des lignes horizontales et verticales
            plt.plot([-100, 100], [0, 0], color='grey', ls='--')
            plt.plot([0, 0], [-100, 100], color='grey', ls='--')

            # nom des axes, avec le pourcentage d'inertie expliqué
            plt.xlabel('F{} ({}%)'.format(d1+1, round(100*pca.explained_variance_ratio_[d1],1)))
            plt.ylabel('F{} ({}%)'.format(d2+1, round(100*pca.explained_variance_ratio_[d2],1)))

            plt.title("Projection des individus (sur F{} et F{})".format(d1+1, d2+1))
            plt.show(block=False)
    
    
# --------------------------------------------------------------------
# -- KDE PLOT graphe
# --------------------------------------------------------------------    
def plot_graph(df_work):
    """Graph densité pour 1 ou plusieurs colonne d'un dataframe
       @param in : df_work dataframe obligatoire
       @param out : none
    """

    plt.figure(figsize=(10, 5))
    axes = plt.axes()

    label_patches = []
    colors = ['Blue', 'SeaGreen', 'Sienna', 'DodgerBlue', 'Purple','Green']

    i = 0
    for col in df_work.columns:
        label = col
        sns.kdeplot(df_work[col], color=colors[i])
        label_patch = mpatches.Patch(
            color=colors[i],
            label=label)
        label_patches.append(label_patch)
        i += 1
    plt.xlabel('')
    plt.legend(
        handles=label_patches,
        bbox_to_anchor=(
            1.05,
            1),
        loc=2,
        borderaxespad=0.,
        facecolor='white')
    plt.grid(False)
    axes.set_facecolor('white')

    plt.show()    
    
    

####

def suppr_ponct(val):
    """
    Suppression de la ponctuation au texte transmis en paramètres.
    Parameters
    ----------
    val : texte dont on veut supprimer la ponctuation
    Returns
    -------
    Texte sans ponctuation
    """
    if isinstance(val, str):  # éviter les nan
        val = val.lower()
        val = re.compile('[éèêë]+').sub("e", val)
        val = re.compile('[àâä]+').sub("a", val)
        val = re.compile('[ùûü]+').sub("u", val)
        val = re.compile('[îï]+').sub("i", val)
        val = re.compile('[ôö]+').sub("o", val)
        return re.compile('[^A-Za-z" "]+').sub("", val)
    return val


################################################################################
## Function to plot continuous variables distribution (CDF, violin, displot, box)
################################################################################

                  #######  MULTIVARIEE  ##########

def plot_continuous_variables(data, column_name,
                              plots=['distplot', 'CDF', 'box', 'violin'], 
                              scale_limits=None, figsize=(20, 9),
                              log_scale=False, palette=['Gold', 'purple']):
    '''
    Function to plot continuous variables distribution
    Inputs:
        data: DataFrame
            The DataFrame from which to plot.
        column_name: str
            Column's name whose distribution is to be plotted.
        plots: list, default = ['distplot', 'CDF', box', 'violin']
            List of plots to plot for Continuous Variable.
        scale_limits: tuple (left, right), default = None
            To control the limits of values to be plotted in case of outliers.
        figsize: tuple, default = (20,8)
            Size of the figure to be plotted.
        log_scale: bool, default = False
            Whether to use log-scale for variables with outlying points.
    '''
    data_to_plot = data.copy()
    
    if scale_limits:
        data_to_plot = data_to_plot[(data_to_plot[column_name] > scale_limits[0]) & 
                                    (data_to_plot[column_name] < scale_limits[1])]
    
    plt.figure(figsize=figsize)
    sns.set_style('whitegrid')
    
    plot_functions = {
        'CDF': plot_cdf,
        'distplot': plot_distplot,
        'violin': plot_violin,
        'box': plot_box
    }

    for i, plot_type in enumerate(plots, start=1):
        plt.subplot(1, len(plots), i)
        plt.subplots_adjust(wspace=0.25)
        
        if plot_type in plot_functions:
            plot_functions[plot_type](data_to_plot, column_name, log_scale, palette)

    plt.show()

def plot_cdf(data, column_name, log_scale, palette):

    data['TARGET_LABEL'] = data['TARGET'].replace({0: 'Non-défaillants', 1: 'Défaillants'})
    
    for target_value, color in zip(data['TARGET_LABEL'].unique(), palette):
        subset = data[data['TARGET_LABEL'] == target_value]
        sns.ecdfplot(data=subset[column_name], complementary=True, label=f'Target: {target_value}',
                     color=color)
    plt.xlabel(column_name, fontsize=16)
    plt.ylabel('Probability', fontsize=16)
    plt.title(f'CDF of {column_name}', fontsize=18)
    plt.legend(fontsize='medium')
    if log_scale:
        plt.xscale('log')
        plt.xlabel(f'{column_name} - (log-scale)')

def plot_distplot(data, column_name, log_scale, palette):

    data['TARGET_LABEL'] = data['TARGET'].replace({0: 'Non-défaillants', 1: 'Défaillants'})
    
    for target_value, color in zip(data['TARGET_LABEL'].unique(), palette):
        subset = data[data['TARGET_LABEL'] == target_value]
        sns.kdeplot(data=subset[column_name], label=f'Target: {target_value}',
                    color=color, shade=True)
    plt.xlabel(column_name, fontsize=16)
    plt.ylabel('Probability Density', fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(fontsize=18)
    plt.title(f'Dist-Plot of {column_name}', fontsize=18)
    if log_scale:
        plt.xscale('log')
        plt.xlabel(f'{column_name} (log scale)', fontsize=16)

def plot_violin(data, column_name, log_scale, palette):

    data['TARGET_LABEL'] = data['TARGET'].replace({0: 'Non-défaillants' , 1: 'Défaillants'})
    
    sns.violinplot(x='TARGET_LABEL', y=column_name, data=data, palette=palette)
    plt.title(f'Violin-Plot of {column_name}', fontsize=18)
    if log_scale:
        plt.yscale('log')
        plt.ylabel(f'{column_name} (log Scale)')

def plot_box(data, column_name, log_scale, palette):

    data['TARGET_LABEL'] = data['TARGET'].replace({0: 'Non-défaillants' , 1: 'Défaillants'})
    sns.boxplot(x='TARGET_LABEL', y=column_name, data=data, palette=palette)
    plt.title(f'Box-Plot of {column_name}', fontsize=18)
    if log_scale:
        plt.yscale('log')
        plt.ylabel(f'{column_name} (log Scale)', fontsize=16)
    plt.xlabel('TARGET_LABEL', fontsize=16)
    plt.ylabel(column_name, fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)


def print_percentiles(data, column_name, percentiles=None):
    '''
    Function to print percentile values for given column
    Inputs:
        data: DataFrame
            The DataFrame from which to print percentiles
        column_name: str
            Column's name whose percentiles are to be printed
        percentiles: list, default = None
            The list of percentiles to print, if not given, default are printed
    '''
    print('-' * 79)
    print(f'Pecentiles de la variable {column_name}')
    if not percentiles:
        percentiles = list(range(0, 80, 25)) + list(range(90, 101, 2))
    for i in percentiles:
        
        print(
            f'Pecentile {i} = {np.percentile(data[column_name].dropna(), i)}')
    print("-" * 79)


                ########  UNIVARIEE  ##########

def plot_continuous_variable(data, column_name,
                              plots=['distplot', 'CDF', 'box', 'violin'], 
                              scale_limits=None, figsize=(20, 9),
                              log_scale=False, palette=['Gold', 'purple']):
    '''
    Function to plot continuous variables distribution
    Inputs:
        data: DataFrame
            The DataFrame from which to plot.
        column_name: str
            Column's name whose distribution is to be plotted.
        plots: list, default = ['distplot', 'CDF', box', 'violin']
            List of plots to plot for Continuous Variable.
        scale_limits: tuple (left, right), default = None
            To control the limits of values to be plotted in case of outliers.
        figsize: tuple, default = (20,8)
            Size of the figure to be plotted.
        log_scale: bool, default = False
            Whether to use log-scale for variables with outlying points.
    '''
    data_to_plot = data.copy()
    
    if scale_limits:
        data_to_plot = data_to_plot[(data_to_plot[column_name] > scale_limits[0]) & 
                                    (data_to_plot[column_name] < scale_limits[1])]
    
    plt.figure(figsize=figsize)
    sns.set_style('whitegrid')
    
    plot_functions = {
        'CDF': plot_cdf,
        'distplot': plot_distplot,
        'violin': plot_violin,
        'box': plot_box
    }

    for i, plot_type in enumerate(plots, start=1):
        plt.subplot(1, len(plots), i)
        plt.subplots_adjust(wspace=0.25)
        
        if plot_type in plot_functions:
            plot_functions[plot_type](data_to_plot, column_name, log_scale, palette)

    plt.show()

def plot_cdf_univ(data, column_name, log_scale, palette):

    subset = data[data[column_name] == target_value]
    sns.ecdfplot(data=subset[column_name], complementary=True, label=f'Target: {target_value}',
                 color=color)
    plt.xlabel(column_name, fontsize=16)
    plt.ylabel('Probability', fontsize=16)
    plt.title(f'CDF of {column_name}', fontsize=18)
    plt.legend(fontsize='medium')
    if log_scale:
        plt.xscale('log')
        plt.xlabel(f'{column_name} - (log-scale)')

def plot_distplot_univ(data, column_name, log_scale, palette):
    
    subset = data[data[column_name] == target_value]
    sns.kdeplot(data=subset[column_name], label=f'Target: {target_value}',
                color=color, shade=True)
    plt.xlabel(column_name, fontsize=16)
    plt.ylabel('Probability Density', fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(fontsize=18)
    plt.title(f'Dist-Plot of {column_name}', fontsize=18)
    if log_scale:
        plt.xscale('log')
        plt.xlabel(f'{column_name} (log scale)', fontsize=16)

def plot_violin_univ(data, column_name, log_scale, palette):
    
    sns.violinplot(x=column_name, data=data, palette=palette)
    plt.title(f'Violin-Plot of {column_name}', fontsize=18)
    if log_scale:
        plt.yscale('log')
        plt.ylabel(f'{column_name} (log Scale)')

def plot_box_univ(data, column_name, log_scale, palette):

    sns.boxplot(x=column_name, data=data, palette=palette)
    plt.title(f'Box-Plot of {column_name}', fontsize=18)
    if log_scale:
        plt.yscale('log')
        plt.ylabel(f'{column_name} (log Scale)', fontsize=16)
    plt.xlabel(column_name, fontsize=16)
    plt.xticks(fontsize=16)



####################################
## CARTE GRPAHIQUE DES CORRELATIONS
###################################

def plot_phik_matrix(data, categorical_columns, figsize=(20, 20),
                     mask_upper=True, tight_layout=True, linewidth=0.1,
                     fontsize=10, cmap='Blues', show_target_top_corr=True,
                     target_top_columns=10):
    data_for_phik = data[categorical_columns].astype('object')
    phik_matrix = data_for_phik.phik_matrix()
    mask_array = np.triu(np.ones(phik_matrix.shape)) if mask_upper else np.zeros(phik_matrix.shape)

    plt.figure(figsize=figsize, tight_layout=tight_layout)
    sns.heatmap(phik_matrix, annot=False, mask=mask_array, linewidth=linewidth, cmap=cmap)
    plt.xticks(rotation=90, fontsize=fontsize)
    plt.yticks(rotation=0, fontsize=fontsize)
    plt.title("Phi-K Correlation Heatmap des variables catégorielles", fontsize=fontsize+4)
    plt.show()

    if show_target_top_corr:
        phik_df = pd.DataFrame({'Variable': phik_matrix.TARGET.index[1:], 'Phik-Correlation': phik_matrix.TARGET.values[1:]})
        phik_df = phik_df.sort_values(by='Phik-Correlation', ascending=False)
        display(phik_df.head(target_top_columns).style.hide_index())

############################

def missing_values(df):
    '''
    Fonction qui calcule les valeurs manquantes d'un dataset
    '''
    # Nombre total de valeurs manquantes
    mis_val = df.isna().sum()
    
    # Pourcentage de valeurs manquantes
    mis_val_percent = (mis_val / len(df)) * 100
    
    # Construction d'un tableau avec les résultats
    mis_val_table = pd.DataFrame({'Valeurs manquantes': mis_val, '% total des valeurs': mis_val_percent})
    
    # On range de manière décroissante 
    mis_val_table = mis_val_table[mis_val_table['Valeurs manquantes'] != 0].sort_values('% total des valeurs', ascending=False).round(1)
    
    return mis_val_table



#############################

def description_dataset(data, name):
    '''
    Return the description of a dataset
    
        Parameters: 
            data: dataframe
            name: 
            
        Returns:
            the 3 first line of the dataframe
            the shape, the types, the number of null values,
            the number of unique values, the number of duplicated values
            
    '''
    print("On traite le dataset ", name)
    msno.bar(data)
    
    display(data.head(3))
    print(f'Taille :-------------------------------------------------------------- {data.shape}')
    
    print("--"*50)
    print("Valeurs manquantes par colonnes (%): ")
    table = missing_values(data)
    display(table.head(50))

    print("--"*50)
    print("Valeurs différentes par variables : ")
    for col in data:
        if data[col].nunique() < 30:
            print (f'{col :-<70} {data[col].unique()}')
        else : 
            print(f'{col :-<70} contient {data[col].nunique()} valeurs différentes')
    print("--"*50)
    print(f"Nombre de doublons : {data.duplicated().sum()}")


################################


def reduce_mem_usage(data, verbose=True):
    # source: https://www.kaggle.com/gemartin/load-data-reduce-memory-usage
    '''
    This function is used to reduce the memory usage by converting the datatypes of a pandas
    DataFrame withing required limits.
    '''

    start_mem = data.memory_usage().sum() / 1024**2
    if verbose:
        print('-' * 79)
        print('Memory usage du dataframe: {:.2f} MB'.format(start_mem))

    for col in data.columns:
        col_type = data[col].dtype

        #  Float et int
        if col_type != object:
            c_min = data[col].min()
            c_max = data[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(
                        np.int8).min and c_max < np.iinfo(
                        np.int8).max:
                    data[col] = data[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    data[col] = data[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    data[col] = data[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    data[col] = data[col].astype(np.int64)
            else:
                if c_min > np.finfo(
                        np.float16).min and c_max < np.finfo(
                        np.float16).max:
                    data[col] = data[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    data[col] = data[col].astype(np.float32)
                else:
                    data[col] = data[col].astype(np.float64)

        # # Boolean : pas à faire car pour machine learning il faut des int 0/1
        # et pas False/True
        # if list(data[col].unique()) == [0, 1] or list(data[col].unique()) == [1, 0]:
        #     data[col] = data[col].astype(bool)

    end_mem = data.memory_usage().sum() / 1024**2
    if verbose:
        print('Memory usage après optimization: {:.2f} MB'.format(end_mem))
        print('Diminution de {:.1f}%'.format(
            100 * (start_mem - end_mem) / start_mem))
        print('-' * 79)

    return data

###########################

def traduire_valeurs_variable(dataframe, colonne_a_traduire, dictionnaire):
    """
    Traduire les valeurs de la colonne du dataframe transmis par la valeur du dictionnaire
    ----------
    @param IN : dataframe : DataFrame, obligatoire
                colonne_a_traduire : colonne dont on veut traduire les valeurs obligatoire
                dictionnaire : dictionnaire clé=à remplacer,
                              valeur = le texte de remplacement oblgatoire
    @param OUT :None
    """
    dataframe[colonne_a_traduire] = dataframe[colonne_a_traduire].replace(dictionnaire)


############################

def cleaning_categories(col):
    '''
    Fonction qui nettoie une phrase de ses caractères spéciaux.
    Et la passe en majuscule( pour un passage en nom de colonne avec one hot)
    '''
    col = str(col).replace('-', '').replace('+', '').replace('/', ' ').replace(':', '_')
    col = str(col).upper()
    return col

#############################

def suppr_var_colineaire(dataframe, seuil=0.8):
    '''
    Récupération de la liste des variables fortement corrélées supérieur
    au seuil transmis.
    Parameters
    ----------
    dataframe : dataframe à analyser, obligatoire.
    seuil : le seuil de colinéarité entre les variables (0.8 par défaut).
    Returns
    -------
    cols_corr_a_supp : liste des variables à supprimer.
    '''
    
    # Matrice de corrélation avec valeur absolue pour ne pas avoir à gérer
    # les corrélations positives et négatives séparément
    corr = dataframe.corr().abs()
    # On ne conserve que la partie supérieur à la diagonale pour n'avoir
    # qu'une seule fois les corrélations prisent en compte (symétrie axiale)
    corr_triangle =  corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    
    # Variables avec un coef de Pearson > 0.8?
    cols_corr_a_supp = [var for var in corr_triangle.columns
                        if any(corr_triangle[var] > seuil)]
    print(f'{len(cols_corr_a_supp)} variables fortement corrélées à supprimer :\n')
    for var in cols_corr_a_supp:
        print(var)
        
    return cols_corr_a_supp



# --------------------------------------------------------------------
# -- PLAGE DE VALEURS MANQUANTES
# --------------------------------------------------------------------


def distribution_variables_plages_pourc_donnees(
        dataframe, variable, liste_bins):
    """
    Retourne les plages des pourcentages des valeurs pour le découpage transmis
    Parameters
    ----------
    @param IN : dataframe : DataFrame, obligatoire
                variable : variable à découper obligatoire
                liste_bins: liste des découpages facultatif int ou pintervallindex
    @param OUT : dataframe des plages de nan
    """
    nb_lignes = len(dataframe[variable])
    s_gpe_cut = pd.cut(
        dataframe[variable],
        bins=liste_bins).value_counts().sort_index()
    df_cut = pd.DataFrame({'Plage': s_gpe_cut.index,
                           'nb_données': s_gpe_cut.values})
    df_cut['%_données'] = [
        (row * 100) / nb_lignes for row in df_cut['nb_données']]

    return df_cut.style.hide_index()


#---------------------------------------------------------------------
# FONCTION DE REDUCTION DE MEMOIRE D'UN DATAFRAME
#---------------------------------------------------------------------


def reduce_mem_usage(data, verbose=True):
    # source: https://www.kaggle.com/gemartin/load-data-reduce-memory-usage
    '''
    This function is used to reduce the memory usage by converting the datatypes of a pandas
    DataFrame withing required limits.
    '''

    start_mem = data.memory_usage().sum() / 1024**2
    if verbose:
        print('-' * 79)
        print('Memory usage du dataframe: {:.2f} MB'.format(start_mem))

    for col in data.columns:
        col_type = data[col].dtype

        #  Float et int
        if col_type != object:
            c_min = data[col].min()
            c_max = data[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(
                        np.int8).min and c_max < np.iinfo(
                        np.int8).max:
                    data[col] = data[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    data[col] = data[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    data[col] = data[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    data[col] = data[col].astype(np.int64)
            else:
                if c_min > np.finfo(
                        np.float16).min and c_max < np.finfo(
                        np.float16).max:
                    data[col] = data[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    data[col] = data[col].astype(np.float32)
                else:
                    data[col] = data[col].astype(np.float64)

        # # Boolean : pas à faire car pour machine learning il faut des int 0/1
        # et pas False/True
        # if list(data[col].unique()) == [0, 1] or list(data[col].unique()) == [1, 0]:
        #     data[col] = data[col].astype(bool)

    end_mem = data.memory_usage().sum() / 1024**2
    if verbose:
        print('Memory usage après optimization: {:.2f} MB'.format(end_mem))
        print('Diminution de {:.1f}%'.format(
            100 * (start_mem - end_mem) / start_mem))
        print('-' * 79)

    return data

#------------------------------------------------------------------------

def agg_var_num(dataframe, group_var, dict_agg, prefix):
    """
    Aggregates the numeric values in a dataframe.
    This can be used to create features for each instance of the grouping variable.
    Parameters
    --------
        dataframe (dataframe): the dataframe to calculate the statistics on
        group_var (string): the variable by which to group df
        df_name (string): the variable used to rename the columns
    Return
    --------
        agg (dataframe): 
            a dataframe with the statistics aggregated for 
            all numeric columns. Each instance of the grouping variable will have 
            some statistics (mean, min, max, sum ...) calculated. 
            The columns are also renamed to keep track of features created.
    
    """
    # Remove id variables other than grouping variable
    for col in dataframe:
        if col != group_var and 'SK_ID' in col:
            dataframe = dataframe.drop(columns=col)

    group_ids = dataframe[group_var]
    numeric_df = dataframe.select_dtypes('number')
    numeric_df[group_var] = group_ids

    # Group by the specified variable and calculate the statistics
    agg = numeric_df.groupby(group_var).agg(dict_agg)

    # Ajout suffix mean, sum...
    agg.columns = ['_'.join(tup).strip().upper()
                   for tup in agg.columns.values]

    # Ajout du prefix bureau_balance pour avoir une idée du fichier
    agg.columns = [prefix + '_' + col
                   if col != group_var else col
                   for col in agg.columns]

    agg.reset_index(inplace=True)

    return agg

#----------------------------------------------------------------------------------


def agg_var_cat(dataframe, group_var, prefix):
    '''
        Aggregates the categorical features in a child dataframe
        for each observation of the parent variable.
        
        Parameters
        --------
        - dataframe        : pandas dataframe
                    The dataframe to calculate the value counts for.
            
        - parent_var : string
                    The variable by which to group and aggregate 
                    the dataframe. For each unique value of this variable, 
                    the final dataframe will have one row
            
        - prefix    : string
                    Variable added to the front of column names 
                    to keep track of columns

        Return
        --------
        categorical : pandas dataframe
                    A dataframe with aggregated statistics for each observation 
                    of the parent_var
                    The columns are also renamed and columns with duplicate values 
                    are removed.
    '''
    
    # Select the categorical columns
    categorical = pd.get_dummies(dataframe.select_dtypes('object'))

    # Make sure to put the identifying id on the column
    categorical[group_var] = dataframe[group_var]

    # Groupby the group var and calculate the sum and mean
    categorical = categorical.groupby(group_var).agg(['sum', 'count', 'mean'])
    
    column_names = []
    
    # Iterate through the columns in level 0
    for var in categorical.columns.levels[0]:
        # Iterate through the stats in level 1
        for stat in ['sum', 'count', 'mean']:
            # Make a new column name
            column_names.append('%s_%s_%s' % (prefix, var, stat))
    
    categorical.columns = column_names
    
    # Remove duplicate columns by values
    # _, idx = np.unique(categorical, axis = 1, return_index = True)
    # categorical = categorical.iloc[:, idx]
    
    return categorical


#-----------------------------------------------------------------------------

def agg_moy_par_pret(dataframe, group_var, prefix):
    """Aggregates the numeric values in a dataframe. This can
    be used to create features for each instance of the grouping variable.
    
    Parameters
    --------
        dataframe (dataframe): 
            the dataframe to calculate the statistics on
        group_var (string): 
            the variable by which to group df
        prefix (string): 
            the variable used to rename the columns
        
    Return
    --------
        agg (dataframe): 
            a dataframe with the statistics aggregated for 
            all numeric columns. Each instance of the grouping variable will have 
            the statistics (mean, min, max, sum; currently supported) calculated. 
            The columns are also renamed to keep track of features created.
    
    """
    # Remove id variables other than grouping variable
    for col in dataframe:
        if col != group_var and 'SK_ID' in col:
            dataframe = dataframe.drop(columns = col)
            
    group_ids = dataframe[group_var]
    numeric_df = dataframe.select_dtypes('number')
    numeric_df[group_var] = group_ids

    # Group by the specified variable and calculate the statistics
    agg = numeric_df.groupby(group_var).agg(['mean']).reset_index()

    # Need to create new column names
    columns = [group_var]

    # Iterate through the variables names
    for var in agg.columns.levels[0]:
        # Skip the grouping variable
        if var != group_var:
            # Iterate through the stat names
            for stat in agg.columns.levels[1][:-1]:
                # Make a new column name for the variable and stat
                columns.append('%s_%s_%s' % (prefix, var, stat))

    agg.columns = columns
    
    return agg


#----------------------------------------------------------------------
# FONCTIONS METRIQUES METIER
#----------------------------------------------------------------------


def custom_score(y_reel, y_pred, taux_tn=1, taux_fp=-5, taux_fn=-20, taux_tp=0):
    '''
    Métrique métier tentant de minimiser le risque d'accord prêt pour la
    banque en pénalisant les faux négatifs.
    Parameters
    ----------
    y_reel : classe réélle, obligatoire (0 ou 1).
    y_pred : classe prédite, obligatoire (0 ou 1).
    taux_tn : Taux de vrais négatifs, optionnel (1 par défaut),
              le prêt est remboursé : la banque gagne de l'argent.
    taux_fp : Taux de faux positifs, optionnel (-5 par défaut),
               le prêt est refusé par erreur : la banque perd les intérêts,
               manque à gagner mais ne perd pas réellement d'argent (erreur de
               type I).
    taux_fn : Taux de faux négatifs, optionnel (-20 par défaut),
              le prêt est accordé mais le client fait défaut : la banque perd
              de l'argent (erreur de type II)..
    taux_tp : Taux de vrais positifs, optionnel (0 par défaut),
              Le prêt est refusé à juste titre : la banque ne gagne ni ne perd
              d'argent.
    Returns
    -------
    score : gain normalisé (entre 0 et 1) un score élevé montre une meilleure
            performance
    '''
    # Vérification que les listes sont non vides et de même longueur
    if len(y_reel) != len(y_pred):
        raise ValueError("Les listes y_reel et y_pred doivent être de même longueur")
    
    # Calcul de la matrice de confusion
    tn, fp, fn, tp = confusion_matrix(y_reel, y_pred).ravel()
    
    # Gain total
    gain_tot = tn * taux_tn + fp * taux_fp + fn * taux_fn + tp * taux_tp
    
    # Gain maximum : toutes les prédictions sont correctes
    gain_max = (tn + fp) * taux_tn + (fn + tp) * taux_tp
    
    # Gain minimum : on accorde aucun prêt, la banque ne gagne rien
    gain_min = (tn + fp) * taux_fp + (fn + tp) * taux_fn
    
    score = (gain_tot - gain_min) / (gain_max - gain_min)
    return score


def custom_score_2(y_reel, y_pred, taux_tn=1, taux_fp=-1, taux_fn=-10, taux_tp=0):
    '''
    Métrique métier tentant de minimiser le risque d'accord prêt pour la
    banque en pénalisant les faux négatifs.
    Parameters
    ----------
    y_reel : classe réélle, obligatoire (0 ou 1).
    y_pred : classe prédite, obligatoire (0 ou 1).
    taux_tn : Taux de vrais négatifs, optionnel (1 par défaut),
              le prêt est remboursé : la banque gagne de l'argent ==>
              à encourager.
    taux_fp : Taux de faux positifs, optionnel (0 par défaut),
               le prêt est refusé par erreur : la banque perd les intérêts,
               manque à gagner mais ne perd pas réellement d'argent (erreur de
               type I) ==> à pénaliser.
    taux_fn : Taux de faux négatifs, optionnel (-10 par défaut),
              le prêt est accordé mais le client fait défaut : la banque perd
              de l'argent (erreur de type II). ==> à pénaliser
    taux_tp : Taux de vrais positifs, optionnel (1 par défaut),
              Le prêt est refusé à juste titre : la banque ne gagne ni ne perd
              d'argent.
    Returns
    -------
    score : gain normalisé (entre 0 et 1) un score élevé montre une meilleure
            performance
    '''
    # Matrice de Confusion
    (tn, fp, fn, tp) = confusion_matrix(y_reel, y_pred).ravel()
    # Gain total
    gain_tot = tn * taux_tn + fp * taux_fp + fn * taux_fn + tp * taux_tp
    # Gain maximum : toutes les prédictions sont correctes
    gain_max = (fp + tn) * taux_tn + (fn + tp) * taux_tp
    # Gain minimum : on accorde aucun prêt, la banque ne gagne rien
    gain_min = (fp + tn) * taux_fp + (fn + tp) * taux_fn
    
    custom_score = (gain_tot - gain_min) / (gain_max - gain_min)
    
    # Gain normalisé (entre 0 et 1) un score élevé montre une meilleure
    # performance
    return custom_score

def custom_score_3(y_reel, y_pred, taux_tn=0.2, taux_fp=-0.2, taux_fn=-0.7, taux_tp=0):
    '''
    Métrique métier tentant de minimiser le risque d'accord prêt pour la
    banque en pénalisant les faux négatifs.
    Les 2 précédentes n'ayant pas donner les résultats excomptés, on va
    raisonner en terme de coût pour la banque.
    TN : Les clients prédits non-défaillants et qui sont bien non-défaillants
         La banque accorde le prêt.
         Ils ont remboursé, la banque gagne S
    TP : Les clients prédits défaillants qui sont bien défaillants.
         La banque n'a pas accordé de prêt ==> pas de gain, ni gagné ni perdu
         d'argent.
    FN : Les clients prédits non-défaillants mais qui sont défaillants.
         La banque accorde le prêt.
         Ils n'ont pas tout remboursé, hypothèse en moyenne ils remboursent un
         tiers avant d'être défaillants ==> perte de 70% du montant du crédit.
    FP : Les clients prédits défaillants mais qui sont non-défaillants.
         La banque n'accorde pas le prêt ==> perte des 20% d'intérêt que les
         clients auraient remboursé.
    Donc le gain de la banque :
        gain = 
         
    Parameters
    ----------
    y_reel : classe réélle, obligatoire (0 ou 1).
    y_pred : classe prédite, obligatoire (0 ou 1).
    taux_tn : Taux de vrais négatifs, optionnel (1 par défaut),
              le prêt est remboursé : la banque gagne de l'argent ==>
              à encourager.
    taux_fp : Taux de faux positifs, optionnel (0 par défaut),
               le prêt est refusé par erreur : la banque perd les intérêts,
               manque à gagner mais ne perd pas réellement d'argent (erreur de
               type I) ==> à pénaliser.
    taux_fn : Taux de faux négatifs, optionnel (-10 par défaut),
              le prêt est accordé mais le client fait défaut : la banque perd
              de l'argent (erreur de type II). ==> à pénaliser
    taux_tp : Taux de vrais positifs, optionnel (1 par défaut),
              Le prêt est refusé à juste titre : la banque ne gagne ni ne perd
              d'argent.
    Returns
    -------
    score : gain normalisé (entre 0 et 1) un score élevé montre une meilleure
            performance
    '''
    # Matrice de Confusion
    (tn, fp, fn, tp) = confusion_matrix(y_reel, y_pred).ravel()
    # Gain total
    gain_tot = tn * taux_tn + fp * taux_fp + fn * taux_fn + tp * taux_tp
    # Gain maximum : toutes les prédictions sont correctes
    gain_max = (fp + tn) * taux_tn + (fn + tp) * taux_tp
    # Gain minimum : on accorde aucun prêt, la banque ne gagne rien
    gain_min = (fp + tn) * taux_fp + (fn + tp) * taux_fn
    
    custom_score = (gain_tot - gain_min) / (gain_max - gain_min)
    
    # Gain normalisé (entre 0 et 1) un score élevé montre une meilleure
    # performance
    return custom_score


# -----------------------------------------------------------------------
# -- MATRICE DE CONFUSION DE LA CLASSIFICATION BINAIRE
# -----------------------------------------------------------------------

def afficher_matrice_confusion(y_true, y_pred, title):

    plt.figure(figsize=(6, 4))

    cm = confusion_matrix(y_true, y_pred)
    
    labels = ['Non défaillants', 'Défaillants']
    
    sns.heatmap(cm,
                xticklabels=labels,
                yticklabels=labels,
                annot=True,
                fmt='d',
                cmap=plt.cm.Blues)
    plt.title(f'Matrice de confusion de : {title}')
    plt.ylabel('Classe réelle')
    plt.xlabel('Classe prédite')
    plt.show()   



def process_classification(model, X_train, X_valid, y_train, y_valid,
                           df_resultats, titre, affiche_res=True,
                           affiche_matrice_confusion=True):
    """
    Lance un modele de classification binaire, effectue cross-validation
    et sauvegarde des scores.
    Parameters
    ----------
    model : modèle de lassification initialisé, obligatoire.
    X_train : train set matrice X, obligatoire.
    X_valid : test set matrice X, obligatoire.
    y_train : train set vecteur y, obligatoire.
    y_valid : test set, vecteur y, obligatoire.
    df_resultats : dataframe sauvegardant les scores, obligatoire
    titre : titre à inscrire dans le tableau de sauvegarde, obligatoire.
    affiche_res : affiche le tableau de résultat (optionnel, True par défaut).
    Returns
    -------
    df_resultats : Le dataframe de sauvegarde des performances.
    y_pred : Les prédictions pour le modèle
    """
    # Top début d'exécution
    time_start = time.time()

    # Entraînement du modèle avec le jeu d'entraînement du jeu d'entrainement
    model.fit(X_train, y_train)

    # Sauvegarde du modèle de classification entraîné
    with open('modele_' + titre + '.pickle', 'wb') as f:
        pickle.dump(model, f, pickle.HIGHEST_PROTOCOL)
    
    # Top fin d'exécution
    time_end_train = time.time()
    
    # Prédictions avec le jeu de validation du jeu d'entraînement
    y_pred = model.predict(X_valid)

    # Top fin d'exécution
    time_end = time.time()

    # Probabilités
    y_proba = model.predict_proba(X_valid)[:, 1]
    
    # Calcul des métriques
    # Rappel/recall sensibilité
    recall = recall_score(y_valid, y_pred)
    # Précision
    precision = precision_score(y_valid, y_pred)
    # F-mesure ou Fbeta
    f1_score = fbeta_score(y_valid, y_pred, beta=1)
    f5_score = fbeta_score(y_valid, y_pred, beta=5)
    f10_score = fbeta_score(y_valid, y_pred, beta=10)
    # Score ROC AUC aire sous la courbe ROC
    roc_auc = roc_auc_score(y_valid, y_proba)
    # Score PR AUC aire sous la courbe précion/rappel
    pr_auc = average_precision_score(y_valid, y_proba)
    # Métrique métier
    banque_score = custom_score(y_valid, y_pred)

    # durée d'exécution d'entraînement
    time_exec_train = time_end_train - time_start
    # durée d'exécution entraînement + validation
    time_execution = time_end - time_start

    # cross validation
    scoring = ['roc_auc', 'recall', 'precision']
    scores = cross_validate(model, X_train, y_train, cv=10,
                            scoring=scoring, return_train_score=True)

    # Sauvegarde des performances
    df_resultats = df_resultats.append(pd.DataFrame({
        'Modèle': [titre],
        'Rappel': [recall],
        'Précision': [precision],
        'F1': [f1_score],
        'F5': [f5_score],
        'F10': [f10_score],
        'ROC_AUC': [roc_auc],
        'PR_AUC': [pr_auc],
        'Metier_score': [banque_score],
        'Durée_train': [time_exec_train],
        'Durée_tot': [time_execution],
        # Cross-validation
        'Train_roc_auc_CV': [scores['train_roc_auc'].mean()],
        'Train_roc_auc_CV +/-': [scores['train_roc_auc'].std()],
        'Test_roc_auc_CV': [scores['test_roc_auc'].mean()],
        'Test_roc_auc_CV +/-': [scores['test_roc_auc'].std()],
        'Train_recall_CV': [scores['train_recall'].mean()],
        'Train_recall_CV +/-': [scores['train_recall'].std()],
        'Test_recall_CV': [scores['test_recall'].mean()],
        'Test_recall_CV +/-': [scores['test_recall'].std()],
        'Train_precision_CV': [scores['train_precision'].mean()],
        'Train_precision_CV +/-': [scores['train_precision'].std()],
        'Test_precision_CV': [scores['test_precision'].mean()],
        'Test_precision_CV +/-': [scores['test_precision'].std()],
    }), ignore_index=True)

    # Sauvegarde du tableau de résultat
    # with open('../sauvegarde/modelisation/df_resultat_scores.pickle', 'wb') as df:
    #     pickle.dump(df_resultats, df, pickle.HIGHEST_PROTOCOL)
    
    if affiche_res:
        mask = df_resultats['Modèle'] == titre
        display(df_resultats[mask].style.hide_index())

    if affiche_matrice_confusion:
        afficher_matrice_confusion(y_valid, y_pred, titre)

    return df_resultats



# ------------------------------------------------------------------------
# -- SAUVEGARDE DES TAUX
# -- TN : vrais négatifs, TP : vrais positifs
# -- FP : faux positifs, FN : faux négatifs
# ------------------------------------------------------------------------

def sauvegarder_taux(titre_modele, FN, FP, TP, TN, df_taux):
    """
    Lance un modele de classification binaire, effectue cross-validation
    et sauvegarde des scores.
    Parameters
    ----------
    model : modèle de lassification initialisé, obligatoire.
    FN : nombre de faux négatifs, obligatoire.
    FP : nombre de faux positifs, obligatoire.
    TN : train set vecteur y, obligatoire.
    TP : test set, vecteur y, obligatoire.
    df_taux : dataframe sauvegardant les taux, obligatoire
    titre : titre à inscrire dans le tableau de sauvegarde, obligatoire.
    Returns
    -------
    df_taux : Le dataframe de sauvegarde des taux.
    """

    # Sauvegarde des performances
    df_taux = df_taux.append(pd.DataFrame({
        'Modèle': [titre_modele],
        'FN': [FN],
        'FP': [FP],
        'TP': [TP],
        'TN': [TN]
    }), ignore_index=True)

    # Sauvegarde du tableau de résultat
    with open('df_taux.pickle', 'wb') as df:
        pickle.dump(df_taux, df, pickle.HIGHEST_PROTOCOL)
    
    return df_taux


# Lancer la recherche avec BayesSearchCV
def lancer_optimise_baysearch(optimizer, X, y):
    optimizer.fit(X_train, y_train)
    d=pd.DataFrame(optimizer.cv_results_)
    best_score = optimizer.best_score_
    best_score_std = d.iloc[optimizer.best_index_].std_test_score
    best_params = optimizer.best_params_
    print(f'Best CV score : {best_score} +/- {best_score_std}')
    print('Best Hyperparamètres :\n')
    print(best_params)


#----------------------------------------------------------------------------------------------
# FONCTION DE SEUIL: La valeur du seuil de probabilité à 0.5 par défaut pourra également 
# être réglée pour tenter d'optimiser les performances du modèles pour notre métrique métier. 
# Le seuil optimal de bascule de la classe 0 à la classe 1 devra être déterminée pour chacun 
# des modèles entraînés avec la métrique métier comme score.
#---------------------------------------------------------------------------------------------


def determiner_seuil_probabilite(model, X_valid, y_valid, title, n=1):
    '''
    Déterminer le seuil de probabilité optimal pour la métrique métier.
    Parameters
    ----------
    model : modèle entraîné, obligatoire.
    y_valid : valeur réélle.
    X_valid : données à tester.
    title : titre pour graphique.
    n : gain pour la classe 1 (par défaut) ou 0.
    Returns
    -------
    None.
    '''
    seuils = np.arange(0, 1, 0.01)
    sav_gains = []
 
    for seuil in seuils:

        # Score du modèle : n = 0 ou 1
        y_proba = model.predict_proba(X_valid)[:, n]

        # Score > seuil de solvabilité : retourne 1 sinon 0
        y_pred = (y_proba > seuil)
        y_pred = np.multiply(y_pred, 1)
        
        # Sauvegarde du score de la métrique métier
        sav_gains.append(custom_score(y_valid, y_pred))
    
    df_score = pd.DataFrame({'Seuils' : seuils,
                             'Gains' : sav_gains})
    
    # Score métrique métier maximal
    gain_max = df_score['Gains'].max()
    print(f'Score métrique métier maximal : {gain_max}')
    # Seuil optimal pour notre métrique
    seuil_max = df_score.loc[df_score['Gains'].argmax(), 'Seuils']
    print(f'Seuil maximal : {seuil_max}')

    # Affichage du gain en fonction du seuil de solvabilité
    plt.figure(figsize=(12, 6))
    plt.plot(seuils, sav_gains, label='Gain en fonction du seuil')
    plt.axhline(y=gain_max, color='r', linestyle='--', label=f'Gain max: {gain_max:.2f}')
    plt.axvline(x=seuil_max, color='b', linestyle='--', label=f'Seuil optimal: {seuil_max:.2f}')
    plt.xlabel('Seuil de probabilité')
    plt.ylabel('Métrique métier')
    plt.title(title)
    plt.xticks(np.linspace(0, 1, 11))
    plt.legend()
    plt.show()




def process_classification_seuil(model, seuil, X_train, X_valid, y_train,
                                 y_valid, df_res_seuil, titre,
                                 affiche_res=True,
                                 affiche_matrice_confusion=True):
    """
    Lance un modele de classification binaire, effectue cross-validation
    et sauvegarde des scores.
    Parameters
    ----------
    model : modèle de lassification initialisé, obligatoire.
    seuil : seuil de probabilité optimal.
    X_train : train set matrice X, obligatoire.
    X_valid : test set matrice X, obligatoire.
    y_train : train set vecteur y, obligatoire.
    y_valid : test set, vecteur y, obligatoire.
    df_res_seuil : dataframe sauvegardant les scores, obligatoire
    titre : titre à inscrire dans le tableau de sauvegarde, obligatoire.
    affiche_res : affiche le tableau de résultat (optionnel, True par défaut).
    Returns
    -------
    df_resultats : Le dataframe de sauvegarde des performances.
    y_pred : Les prédictions pour le modèle
    """
    # Top début d'exécution
    time_start = time.time()

    # Entraînement du modèle avec le jeu d'entraînement du jeu d'entrainement
    model.fit(X_train, y_train)

    # Sauvegarde du modèle de classification entraîné
    with open('modele_' + titre + '.pickle', 'wb') as f:
        pickle.dump(model, f, pickle.HIGHEST_PROTOCOL)
    
    # Top fin d'exécution
    time_end_train = time.time()
    
    # Score du modèle : n = 0 ou 1
    # Probabilités
    y_proba = model.predict_proba(X_valid)[:, 1]

    # Prédictions avec le jeu de validation du jeu d'entraînement
    # Score > seuil de probabilité : retourne 1 sinon 0
    y_pred = (y_proba > seuil)
    y_pred = np.multiply(y_pred, 1)

    # Top fin d'exécution
    time_end = time.time()

    # Calcul des métriques
    # Rappel/recall sensibilité
    recall = recall_score(y_valid, y_pred)
    # Précision
    precision = precision_score(y_valid, y_pred)
    # F-mesure ou Fbeta
    f1_score = fbeta_score(y_valid, y_pred, beta=1)
    f5_score = fbeta_score(y_valid, y_pred, beta=5)
    f10_score = fbeta_score(y_valid, y_pred, beta=10)
    # Score ROC AUC aire sous la courbe ROC
    roc_auc = roc_auc_score(y_valid, y_proba)
    # Score PR AUC aire sous la courbe précion/rappel
    pr_auc = average_precision_score(y_valid, y_proba)
    # Métrique métier
    banque_score = custom_score(y_valid, y_pred)

    # durée d'exécution d'entraînement
    time_exec_train = time_end_train - time_start
    # durée d'exécution entraînement + validation
    time_execution = time_end - time_start

    # cross validation
    scoring = ['roc_auc', 'recall', 'precision']
    scores = cross_validate(model, X_train, y_train, cv=10,
                            scoring=scoring, return_train_score=True)

    # Sauvegarde des performances
    df_res_seuil = df_res_seuil.append(pd.DataFrame({
        'Modèle': [titre],
        'Rappel': [recall],
        'Précision': [precision],
        'F1': [f1_score],
        'F5': [f5_score],
        'F10': [f10_score],
        'ROC_AUC': [roc_auc],
        'PR_AUC': [pr_auc],
        'Metier_score': [banque_score],
        'Durée_train': [time_exec_train],
        'Durée_tot': [time_execution],
        # Cross-validation
        'Train_roc_auc_CV': [scores['train_roc_auc'].mean()],
        'Train_roc_auc_CV +/-': [scores['train_roc_auc'].std()],
        'Test_roc_auc_CV': [scores['test_roc_auc'].mean()],
        'Test_roc_auc_CV +/-': [scores['test_roc_auc'].std()],
        'Train_recall_CV': [scores['train_recall'].mean()],
        'Train_recall_CV +/-': [scores['train_recall'].std()],
        'Test_recall_CV': [scores['test_recall'].mean()],
        'Test_recall_CV +/-': [scores['test_recall'].std()],
        'Train_precision_CV': [scores['train_precision'].mean()],
        'Train_precision_CV +/-': [scores['train_precision'].std()],
        'Test_precision_CV': [scores['test_precision'].mean()],
        'Test_precision_CV +/-': [scores['test_precision'].std()],
    }), ignore_index=True)

    # Sauvegarde du tableau de résultat
    with open('df_res_seuil.pickle', 'wb') as df:
        pickle.dump(df_res_seuil, df, pickle.HIGHEST_PROTOCOL)
    
    if affiche_res:
        mask = df_res_seuil['Modèle'] == titre
        display(df_res_seuil[mask].style.hide_index())

    if affiche_matrice_confusion:
        afficher_matrice_confusion(y_valid, y_pred, titre)

    return df_res_seuil





def determiner_seuil_probabilite_F10(model, X_valid, y_valid, title, n=1):
    '''
    Déterminer le seuil de probabilité optimal pour la métrique métier.
    Parameters
    ----------
    model : modèle entraîné, obligatoire.
    y_valid : valeur réélle.
    X_valid : données à tester.
    title : titre pour graphique.
    n : gain pour la classe 1 (par défaut) ou 0.
    Returns
    -------
    None.
    '''
    seuils = np.arange(0, 1, 0.01)
    scores_F10 = []
 
    for seuil in seuils:

        # Score du modèle : n = 0 ou 1
        y_proba = model.predict_proba(X_valid)[:, n]

        # Score > seuil de solvabilité : retourne 1 sinon 0
        y_pred = (y_proba > seuil)
        y_pred = np.multiply(y_pred, 1)
        
        # Sauvegarde du score de la métrique métier
        scores_F10.append(fbeta_score(y_valid, y_pred, beta=10))
    
    df_score = pd.DataFrame({'Seuils': seuils,
                             'Gains': scores_F10})
    
    # Score métrique métier maximal
    gain_max = df_score['Gains'].max()
    print(f'Score F10 maximal : {gain_max}')
    # Seuil optimal pour notre métrique
    seuil_max = df_score.loc[df_score['Gains'].argmax(), 'Seuils']
    print(f'Seuil maximal : {seuil_max}')

    # Affichage du gain en fonction du seuil de solvabilité
    plt.figure(figsize=(12, 6))
    plt.plot(seuils, scores_F10, label='Score F10 en fonction du seuil')
    plt.axhline(y=gain_max, color='r', linestyle='--', label=f'Score F10 max: {gain_max:.2f}')
    plt.axvline(x=seuil_max, color='b', linestyle='--', label=f'Seuil optimal: {seuil_max:.2f}')
    plt.xlabel('Seuil de probabilité')
    plt.ylabel('Score F10')
    plt.title(title)
    plt.xticks(np.linspace(0, 1, 11))
    plt.legend()
    plt.show()



def plot_features_importance(features_importance, nom_variables,
                             figsize=(6, 5)):
    '''
    Affiche le liste des variables avec leurs importances par ordre décroissant.
    Parameters
    ----------
    features_importance: les features importances, obligatoire
    nom_variables : nom des variables, obligatoire
    figsize : taille du graphique
    Returns
    -------
    None.
    '''
    df_feat_imp = pd.DataFrame({'feature': nom_variables,
                                'importance': features_importance})
    df_feat_imp_tri = df_feat_imp.sort_values(by='importance')
    
    # BarGraph de visalisation
    plt.figure(figsize=figsize)
    plt.barh(df_feat_imp_tri['feature'], df_feat_imp_tri['importance'])
    plt.yticks(fontsize=20)
    plt.xlabel('Feature Importances (%)')
    plt.ylabel('Variables', fontsize=18)
    plt.title('Comparison des Features Importances', fontsize=30)
    plt.show()



def plot_feature_importances(df, threshold = 0.9):
    """
    Plots 15 most important features and the cumulative importance of features.
    Prints the number of features needed to reach threshold cumulative importance.
    
    Parameters
    --------
    df : dataframe
        Dataframe of feature importances. Columns must be feature and importance
    threshold : float, default = 0.9
        Threshold for prining information about cumulative importances
        
    Return
    --------
    df : dataframe
        Dataframe ordered by feature importances with a normalized column (sums to 1)
        and a cumulative importance column
    
    """
    
    plt.rcParams['font.size'] = 18
    
    # Sort features according to importance
    df = df.sort_values('importance', ascending = False).reset_index()
    
    # Normalize the feature importances to add up to one
    df['importance_normalized'] = df['importance'] / df['importance'].sum()
    df['cumulative_importance'] = np.cumsum(df['importance_normalized'])

    # Make a horizontal bar chart of feature importances
    plt.figure(figsize = (10, 6))
    ax = plt.subplot()
    
    # Need to reverse the index to plot most important on top
    ax.barh(list(reversed(list(df.index[:15]))), 
            df['importance_normalized'].head(15), 
            align = 'center', edgecolor = 'k')
    
    # Set the yticks and labels
    ax.set_yticks(list(reversed(list(df.index[:15]))))
    ax.set_yticklabels(df['feature'].head(15))
    
    # Plot labeling
    plt.xlabel('Normalized Importance'); plt.title('Feature Importances')
    plt.show()
    
    # Cumulative importance plot
    plt.figure(figsize = (8, 6))
    plt.plot(list(range(len(df))), df['cumulative_importance'], 'r-')
    plt.xlabel('Number of Features'); plt.ylabel('Cumulative Importance'); 
    plt.title('Cumulative Feature Importance');
    plt.show();
    
    importance_index = np.min(np.where(df['cumulative_importance'] > threshold))
    print('%d features required for %0.2f of cumulative importance' % (importance_index + 1, threshold))
    
    return df

def identify_zero_importance_features(train, train_labels, iterations = 2):
    """
    Identify zero importance features in a training dataset based on the 
    feature importances from a gradient boosting model. 
    
    Parameters
    --------
    train : dataframe
        Training features
        
    train_labels : np.array
        Labels for training data
        
    iterations : integer, default = 2
        Number of cross validation splits to use for determining feature importances
    """
    
    # Initialize an empty array to hold feature importances
    feature_importances = np.zeros(train.shape[1])

    # Create the model with several hyperparameters
    model = lgb.LGBMClassifier(objective='binary', boosting_type = 'goss', n_estimators = 10000, class_weight = 'balanced',
                           verbose = 200,
                           early_stopping_rounds = 100)
    
    # Fit the model multiple times to avoid overfitting
    for i in range(iterations):

        # Split into training and validation set
        train_features, valid_features, train_y, valid_y = train_test_split(train, train_labels, test_size = 0.25, random_state = i)

        # Train using early stopping
        model.fit(train_features, train_y, eval_set = [(valid_features, valid_y)], 
                  eval_metric = 'auc')

        # Record the feature importances
        feature_importances += model.feature_importances_ / iterations
    
    feature_importances = pd.DataFrame({'feature': list(train.columns), 'importance': feature_importances}).sort_values('importance', ascending = False)
    
    # Find the features with zero importance
    zero_features = list(feature_importances[feature_importances['importance'] == 0.0]['feature'])
    print('\nThere are %d features with 0.0 importance' % len(zero_features))
    
    return zero_features, feature_importances


def tracer_features_importance(dataframe, df_features_importance, jeu, methode):
    """
    Affiche l'étape puis nombre de lignes et de variables pour le dataframe transmis
    Parameters
    ----------
    @param IN : dataframe : DataFrame, obligatoire
                df_features_importance : dataframe de suivi des dimensions,
                                         obligatoire
                jeu : jeu de données train_set, train set avec imputation 1...
                methode : titre du modèle de feature sélection
    @param OUT : dataframe de suivi des dimensions
    """
    # Nombre de variables retenues lors de la feature selection
    n_features = dataframe.shape[0]
    print(f'{jeu} - {methode} : {n_features} variables importantes conservées')

    df_features_importance = \
        df_features_importance.append({'Jeu_données': jeu,
                                       'Méthode': methode,
                                       'Nb_var_importante': n_features},
                                       ignore_index=True)

    # Suivi dimensions
    return df_features_importance

def plot_permutation_importance_eli5(model, x_test, y_test):
    '''
    Affiche les SHAPE VALUES.
    Parameters
    ----------
    model: le modèle de machine learning, obligatoire
    x_test :le jeu de test de la matrice X, obligatoire
    y_test :le jeu de test de la target, obligatoire
    perm : permutation importance
    -------
    None.
    '''
    perm = PermutationImportance(model, random_state=21).fit(x_test, y_test)
    display(eli5.show_weights(perm, feature_names=x_test.columns.tolist()))
    
    return perm


def plot_permutation_importance(model, x_test, y_test, figsize=(6, 6)):
    '''
    Affiche les SHAPE VALUES.
    Parameters
    ----------
    model: le modèle de machine learning, obligatoire
    x_test :le jeu de test de la matrice X, obligatoire
    y_test :le jeu de test de la target, obligatoire
    Returns
    -------
    perm_importance : permutation importance
    '''
    perm_importance = permutation_importance(model, x_test, y_test)

    sorted_idx = perm_importance.importances_mean.argsort()
    plt.figure(figsize=figsize)
    plt.barh(x_test.columns[sorted_idx],
             perm_importance.importances_mean[sorted_idx])
    plt.xlabel("Permutation Importance (%)")
    plt.show()    
    
    return perm_importance 


# --------------------------------------------------------------------
# -- PLAGE DE VALEURS MANQUANTES
# --------------------------------------------------------------------


def distribution_variables_plages(
        dataframe, variable, liste_bins):
    """
    Retourne les plages des pourcentages des valeurs pour le découpage transmis
    Parameters
    ----------
    @param IN : dataframe : DataFrame, obligatoire
                variable : variable à découper obligatoire
                liste_bins: liste des découpages facultatif int ou pintervallindex
    @param OUT : dataframe des plages de nan
    """
    nb_lignes = len(dataframe[variable])
    s_gpe_cut = pd.cut(
        dataframe[variable],
        bins=liste_bins).value_counts().sort_index()
    df_cut = pd.DataFrame({'Plage': s_gpe_cut.index,
                           'nb_données': s_gpe_cut.values})
    df_cut['%_données'] = [
        (row * 100) / nb_lignes for row in df_cut['nb_données']]

    return df_cut.style.hide_index()





