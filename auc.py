##
import os
os.chdir("/home/local/CORNELL/esh76/final")
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.utils import resample
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from numpy import mean##
import os
os.chdir("/home/local/CORNELL/esh76/final")
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.utils import resample
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from numpy import mean
from numpy import std
from sklearn.ensemble import RandomForestClassifier
import joblib
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

##
#read in text files
uniprot2seq = pd.read_csv('UniProt2Seq.txt', sep="\t")
features =  pd.read_csv('Expasy_AA_Scales.txt', sep="\t")
features= features.rename(columns={"Amino Acid": "AA"}) #rename column for merging
ref = pd.merge(features, uniprot2seq, how='outer') #for generation of new features later

#calculate averages in ref file again
ref=ref.sort_values(by=['UniProt', 'Position'])

#list of features
cols = list(ref.columns)
cols.remove("UniProt")
cols.remove("Position")
cols.remove("AA")

#calculate averages using a window of 5 ([pos-2, pos+2]) and create new columns in ref dataset
#group by UniProt
ref_grouped = ref.groupby("UniProt")
for col in cols:
    new_name = col + "_avg"
    ref[new_name] = ref_grouped[col].transform(lambda x: x.rolling(5, center=True, min_periods=5).mean())

#drop original columns
ref=ref.drop(cols, axis=1)

#variables needed that were calculated earlier
cols_to_drop1= ['beta_sheet__Deleage_&_Roux_avg', 'Hphob__Black_avg', 'Hphob__Miyazawa_et_al_avg', 'Hphob_HPLC_pH75__Cowan_avg', 'Hphob__Chothia_avg', 'AA_composition_avg', 'alpha_helix__Deleage_&_Roux_avg', 'Transmembrane_tendency_avg', 'Polarity__Grantham_avg', 'Hphob__Eisenberg_et_al_avg']
cols_to_keep = ['UniProt', 'Position', 'Label', 'AA', 'HPLC__retention_pH_74_avg', 'beta_sheet__Levitt_avg', 'Hphob__Wolfenden_et_al_avg', 'beta_turn__Deleage_&_Roux_avg', 'Bulkiness_avg', 'Number_of_codons_avg', 'Hphob__Bull_&_Breese_avg', 'Hphob__Hopp_&_Woods_avg', 'AA_comp_in_Swiss_Prot_avg', 'beta_turn__Levitt_avg', 'Refractivity_avg', 'Relative_mutability_avg', 'Molecular_weight_avg', 'alpha_helix__Levitt_avg', 'alpha_helix__Chou_&_Fasman_avg', 'Total_beta_strand_avg', 'Antiparallel_beta_strand_avg', 'Hphob__Guy_avg', 'Ratio_hetero_endside_avg', 'Hphob_HPLC__Parker_&_al_avg', 'Hphob__Tanford_avg', '%_buried_residues_avg', 'Recognition_factors_avg', '%_accessible_residues_avg']
cols_to_drop= ['UniProt', "AA", "Label"]
final_model = joblib.load("final_model.pkl")

def return_auc(filename):
    labels_test = pd.read_csv(filename, sep="\t")
    temp = pd.merge(labels_test, uniprot2seq)  # merge into one dataset
    dataset = pd.merge(temp, features)
    # drop original columns
    copy = dataset.copy()
    test_data = pd.merge(copy, ref)  # merge with averages
    test_data = test_data.dropna() #drop rows with na
    test_data = test_data.drop(cols, axis=1)  # drop original columns
    test_data = test_data.drop(cols_to_drop1, axis=1)  # drop highly correlated columns
    test_data = test_data[cols_to_keep]  # keep important features found by RFE
    df_test_norm = test_data.copy()
    df_test_num = df_test_norm.select_dtypes(include='number')
    numerical_cols = list(df_test_num)
    # remove label and position from numerical columns
    numerical_cols.remove('Position')
    numerical_cols.remove('Label')
    #normalize data
    for c in numerical_cols:
        df_test_norm[c] = (test_data[c] - test_data[c].min()) / (test_data[c].max() - test_data[c].min())
    #split data into labels and features
    features_test = df_test_norm.drop(cols_to_drop, axis=1)
    label_test = df_test_norm.Label
    y_score = final_model.predict_proba(features_test)[:, 1]
    fpr_t, tpr_t, _ = roc_curve(label_test, y_score)
    auc_test = roc_auc_score(label_test, y_score)
    return auc_test
