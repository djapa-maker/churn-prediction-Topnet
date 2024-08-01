import pandas as pd
import numpy as np

def preprocess_data(data, datafacture):
    # Convert date columns to datetime
    data['New_DateDerniereConnexion'] = pd.to_datetime(data['New_DateDerniereConnexion'])
    data['new_dateresiliationsouhaite'] = pd.to_datetime(data['new_dateresiliationsouhaite'])
    data['New_Datedebutducontrat'] = pd.to_datetime(data['New_Datedebutducontrat'])
    data['New_DateFinducontrat'] = pd.to_datetime(data['New_DateFinducontrat'])

    # Calculate additional features
    data['daysrc'] = (data['new_dateresiliationsouhaite'] - data['New_DateDerniereConnexion']).dt.days
    data['governorat_mapped'] = pd.factorize(data['gouvernorat'])[0] + 1
    data['type_abonnement'] = (data['New_DateFinducontrat'] - data['New_Datedebutducontrat']).dt.days
    data['type_abonnement'] = round(data['type_abonnement'] / 365)

    # Convert date columns in facture data
    datafacture['Echeance'] = pd.to_datetime(datafacture['Echeance'])
    datafacture['CreatedOn'] = pd.to_datetime(datafacture['CreatedOn'])
    
    # Merge client data and facture data
    merged_data = pd.merge(data, datafacture, left_on='new_codeclient', right_on='unique_codesclient', how='left')
    
    # Calculate additional features based on facture data
    merged_data['nb_facture'] = merged_data.groupby('new_codeclient')['new_numerofacture'].transform('count')
    merged_data['is_facture_payed_on_time'] = (merged_data['Echeance'] - merged_data['CreatedOn']).dt.days >= 0
    merged_data['is_payed_on_time'] = merged_data.groupby('new_codeclient')['is_facture_payed_on_time'].transform('all').astype(int)
    merged_data['total_restepayer'] = merged_data.groupby('new_codeclient')['New_restepayer'].transform('sum')
    merged_data['is_paied'] = (merged_data['total_restepayer'] <= 20).astype(int)
    
    # Drop unnecessary columns
    merged_data.drop(columns=['new_numerofacture', 'is_facture_payed_on_time', 'total_restepayer'], inplace=True)
    merged_data.drop_duplicates(subset='new_codeclient', inplace=True)
    
    # Map specific motifs
    specific_motifs = [
        "Migration vers smart ADSL",
        "Migration vers RAPIDO",
        "Résilliation IP fixe avec réservation",
        "Migration vers GPON",
        "Résiliation suite SWAP GPON",
        "Autres",
        "Migration vers TTBox"
    ]
    merged_data['motif'] = merged_data['Motif resilliation'].apply(lambda x: 0 if x in specific_motifs else 1)
    
    # Select relevant columns and create the target variable
    selected_columns = ['governorat_mapped', 'type_abonnement', 'nb_facture', 'is_payed_on_time', 'is_paied', 'daysrc', 'motif']
    final_data = merged_data[selected_columns]

    # Define conditions for prediction
    conditions = [
        (final_data['nb_facture'] > 2) & (final_data['is_payed_on_time'] == 0),
        (final_data['daysrc'] < 0) & (final_data['is_paied'] == 0),
        (final_data['motif'] == 1) & (final_data['is_payed_on_time'] == 0)
    ]
    values = [1, 1, 1]
    final_data['resiliation'] = np.select(conditions, values, default=0)

    return final_data






