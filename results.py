x##
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
#save prediction results for UniProt2Seq in a file
#read in text files
uniprot2seq = pd.read_csv('UniProt2Seq.txt', sep="\t")
features =  pd.read_csv('Expasy_AA_Scales.txt', sep="\t")
features= features.rename(columns={"Amino Acid": "AA"}) #rename column for merging
ref = pd.merge(features, uniprot2seq) #for generation of new features later

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
new_cols=[]
for col in cols:
    new_name = col + "_avg"
    new_cols.append(new_name)
    ref[new_name] = ref_grouped[col].transform(lambda x: x.rolling(5, center=True).mean())


#replace NA values with original value from original column
for new_col in new_cols:
    original_col = new_col[:len(new_col)-4]
    ref.loc[ref[new_col].isna(), new_col] = ref[original_col]


#drop rows with NA (should be 0 rows dropped)
ref = ref.dropna()

#variables needed that were calculated earlier
cols_to_keep = ['UniProt', 'Position', 'AA', 'HPLC__retention_pH_74_avg', 'beta_sheet__Levitt_avg', 'Hphob__Wolfenden_et_al_avg', 'beta_turn__Deleage_&_Roux_avg', 'Bulkiness_avg', 'Number_of_codons_avg', 'Hphob__Bull_&_Breese_avg', 'Hphob__Hopp_&_Woods_avg', 'AA_comp_in_Swiss_Prot_avg', 'beta_turn__Levitt_avg', 'Refractivity_avg', 'Relative_mutability_avg', 'Molecular_weight_avg', 'alpha_helix__Levitt_avg', 'alpha_helix__Chou_&_Fasman_avg', 'Total_beta_strand_avg', 'Antiparallel_beta_strand_avg', 'Hphob__Guy_avg', 'Ratio_hetero_endside_avg', 'Hphob_HPLC__Parker_&_al_avg', 'Hphob__Tanford_avg', '%_buried_residues_avg', 'Recognition_factors_avg', '%_accessible_residues_avg']
final_model = joblib.load("final_model.pkl")
results = ref[cols_to_keep]  # keep important features found by RFE

##
results_norm = results.copy()
results_num = results_norm.select_dtypes(include='number')
num_cols = list(results_num)
# remove label and position from numerical columns
num_cols.remove('Position')
#normalize data
for c in num_cols:
    results_norm[c] = (results[c] - results[c].min()) / (results[c].max() - results[c].min())
results_norm=results_norm.sort_values(by=['UniProt', 'Position'])
#get features
cols_to_drop= ['UniProt', "AA"]
results_features = results_norm.drop(cols_to_drop, axis=1)
##
#load final model and make predictions
final_model = joblib.load("final_model.pkl")
results_preds = final_model.predict(results_features)
results_series= pd.Series(results_preds)
output = results_norm[['UniProt', 'Position', 'AA']]
output['Label'] = results_series #save predictions as column
output.to_csv('results.txt', header=True, index=None, sep="\t")


