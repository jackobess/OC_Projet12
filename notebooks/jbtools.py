# Les JBTools

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def _minifig(row):      #  _▄█
    lafourchette=row['max']-row['min']
    if lafourchette==0: return("----")
    lequartfourchu=lafourchette/4
    laminifig=''
    
    for latranche in [(row['25%']-row['min']), (row['50%']-row['25%']),(row['75%']-row['50%']),(row['max']-row['75%'])]:
        if latranche==0 : laminifig=laminifig+'█'
        elif lequartfourchu/latranche >2 : laminifig=laminifig+'█'
        elif lequartfourchu/latranche >=0.5 : laminifig=laminifig+'▄'
        else: laminifig=laminifig+'_'
    
    return(laminifig)

def _skewscore(row):
    lamed=row['50%'] ; lemin=row['min'] ; lemax=row['max']
    if lemin<0: lamed=lamed-lemin ; lemax=lemax-lemin; lemin=0
    if lamed==0:
        if lemin==0 and lemax==0: leskewscore=0
        elif lemin==0 and lemax>0: leskewscore='+'
        else: leskewscore='?'
    else:
        min2med=round((lamed-lemin)/lamed,2)
        max2med=round((lemax-lamed)/lamed,2)
        if max2med==min2med: leskewscore=0
        elif min2med==0: leskewscore='+'
        elif max2med==0: leskewscore='-'
        elif max2med>min2med: leskewscore=round(max2med/min2med,1)-1
        elif max2med<min2med: leskewscore=round(-min2med/max2med,1)+1

    return(leskewscore)

def describoost(the_df, the_name="Dataset", the_round=2, sorting_column='Remarque'):
    nb_lignes_total = len(the_df)
    
    pd.set_option('display.expand_frame_repr', False) # On bloque le wrapping pour avoir des tableaux bien lisibles
    pd.set_option('display.float_format', '{:.2f}'.format)
    pd.set_option('display.max_rows', 300)
 
    print(f"\n{'='*20} 🚀 DESCRIBOOST : {the_name} {'='*20}")
    print(f"Nbre de lignes  : {nb_lignes_total} ")
    print(f"Nbre de colonnes: {the_df.shape[1]} ")
    nb_doublons = the_df.duplicated().sum()
    if nb_doublons == 0: print("Aucun doublon trouvé \n")
    else: print(f"Nbre de doublons: {nb_doublons} *** \n")

    # --- 1. BLOC NUMÉRIQUE ---
    cols_num = the_df.select_dtypes(include='number').columns
    if not cols_num.empty:
        df_num = the_df[cols_num].describe().T.round(the_round)
        df_num['skewsc'] = df_num.apply(_skewscore, axis=1)
        df_num['mean/md'] =  df_num.apply(lambda x: round(x['mean'] / x['50%'], 1) if x['50%'] > 0 else '.', axis=1)

        df_num['Minidist'] = df_num.apply(_minifig, axis=1)
        df_num['NaNs'] = nb_lignes_total - df_num['count']
        df_num['Nan%'] = (df_num['NaNs'] / nb_lignes_total * 100).round(1)
        df_num['Remarque'] = "-"
        df_num['Suggestion'] = "-"

        # --- Remarque & Suggestion
        mean_md_num = pd.to_numeric(df_num['mean/md'], errors='coerce')
        skew_num = pd.to_numeric(df_num['skewsc'], errors='coerce')

        df_num.loc[mean_md_num >= 1.4, ['Remarque', 'Suggestion']] = ["R-skewed", "scaling/log?"]
        df_num.loc[mean_md_num <= 0.7, ['Remarque', 'Suggestion']] = ["L-skewed", "scaling/??"]

        df_num.loc[skew_num >= 3, ['Remarque', 'Suggestion']] = ["R-outlier(s)", "log?/suppr?"]
        df_num.loc[skew_num <= -3, ['Remarque', 'Suggestion']] = ["L-outlier(s)", "keep/suppr?"]

        df_num.loc[df_num['NaNs'] > 0, ['Remarque', 'Suggestion']] = ["Présence de NaN", "Imputer/Suppr"]

        df_num.loc[df_num['min'] == df_num['max'], ['Remarque', 'Suggestion']] = ["Valeur Unique", "A supprimer"]        
        
        print("--- ANALYSE NUMÉRIQUE ---")
        if sorting_column == 'index': print(df_num.sort_index())
        elif sorting_column and sorting_column in df_num.columns: print(df_num.sort_values(sorting_column, ascending=True,key=lambda x: x.astype(str)))
        else: print(df_num)
    else:
        print("--- ANALYSE NUMÉRIQUE : Aucune colonne trouvée ---")

    # --- 2. BLOC NON-NUMÉRIQUE ---
    cols_obj = the_df.select_dtypes(exclude='number').columns
    if not cols_obj.empty:
        df_obj = the_df[cols_obj].describe().T
        # Conversion en float pour éviter le plantage du calcul de %
        df_obj['freq%'] = (df_obj['freq'].astype(float) / nb_lignes_total * 100).round(1)
        df_obj['cat.mean'] = (nb_lignes_total / df_obj['unique'].astype(float)).round(0)
        df_obj['cat.tp/mean'] = (df_obj['freq'].astype(float)/df_obj['cat.mean'].astype(float)).round(2)
        df_obj['NaNs'] = nb_lignes_total - df_obj['count'].astype(int)
        df_obj['Nan%'] = (df_obj['NaNs'] / nb_lignes_total * 100).round(1)
        df_obj['Remarque'] = "-"
        df_obj['Suggestion'] = "-"

        # Détection remarque & Suggestion
        df_obj.loc[df_obj['cat.tp/mean'].astype(float) > np.sqrt(df_obj['unique'].astype(float)), ['Remarque', 'Suggestion']] = ["déséquilibré", "-"]
        df_obj.loc[df_obj['unique'].astype(float) > 50, ['Remarque', 'Suggestion']] = ["haute cardinalité", "-"]
        df_obj.loc[df_obj['unique'].astype(float) == nb_lignes_total, ['Remarque', 'Suggestion']] = ["unique=nbrow  id?", "Supprimer?"]

        df_obj.loc[df_obj['NaNs'] > 0, ['Remarque', 'Suggestion']] = ["Présence de NaN", "Imputer/Suppr"]
        df_obj.loc[df_obj['unique'] == 1, ['Remarque', 'Suggestion']] = ["Valeur Unique", "A supprimer"]

        print("\n--- ANALYSE CATEGORIELLE ---")
        if sorting_column == 'index': print(df_obj.sort_index())
        elif sorting_column and sorting_column in df_obj.columns:  print(df_obj.sort_values(sorting_column, ascending=True,key=lambda x: x.astype(str)))
        else: print(df_obj)

    else:
        print("\n--- ANALYSE CATEGORIELLE : Aucune colonne trouvée ---")
    
    print(f"\n{'='*60}")

def showpearson(df, cols_to_exclude=[]):
    cols_num = df.select_dtypes(include='number').columns.difference(cols_to_exclude)
    corr = df[cols_num].corr(method='pearson')
    
    plt.figure(figsize=(14, 10))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0, vmin=-1, vmax=1)
    plt.title('Matrice de corrélation de Pearson')
    plt.tight_layout()
    plt.show()