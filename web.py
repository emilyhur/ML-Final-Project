##
import os
os.chdir("/home/local/CORNELL/esh76/final")
from collections import defaultdict
import pandas as pd
import joblib


seq = 'MSSCI'
AA_list=[]
for char in seq:
    AA_list.append(char)
Position_list=[]
counter = 1
for item in AA_list:
    Position_list.append(counter)
    counter +=1

df = pd.DataFrame(columns =['Position', 'AA'])
df['Position']=Position_list
df['AA']=AA_list

features =  pd.read_csv('Expasy_AA_Scales.txt', sep="\t")
features= features.rename(columns={"Amino Acid": "AA"}) #rename column for merging
ref = pd.merge(features, df) #for generation of new features later

#calculate averages in ref file again
ref=ref.sort_values(by=['Position'])

#list of features
cols = list(ref.columns)
cols.remove("Position")
cols.remove("AA")

#calculate averages using a window of 5 ([pos-2, pos+2]) and create new columns in ref dataset
#group by UniProt
new_cols=[]
for col in cols:
    new_name = col + "_avg"
    new_cols.append(new_name)
    ref[new_name] = ref[col].transform(lambda x: x.rolling(5, center=True).mean())

#replace NA values with original value from original column
for new_col in new_cols:
    original_col = new_col[:len(new_col)-4]
    ref.loc[ref[new_col].isna(), new_col] = ref[original_col]

#drop rows with NA (should be 0 rows dropped)
ref = ref.dropna()

#variables needed that were calculated earlier
cols_to_keep = ['Position', 'AA', 'HPLC__retention_pH_74_avg', 'beta_sheet__Levitt_avg', 'Hphob__Wolfenden_et_al_avg', 'beta_turn__Deleage_&_Roux_avg', 'Bulkiness_avg', 'Number_of_codons_avg', 'Hphob__Bull_&_Breese_avg', 'Hphob__Hopp_&_Woods_avg', 'AA_comp_in_Swiss_Prot_avg', 'beta_turn__Levitt_avg', 'Refractivity_avg', 'Relative_mutability_avg', 'Molecular_weight_avg', 'alpha_helix__Levitt_avg', 'alpha_helix__Chou_&_Fasman_avg', 'Total_beta_strand_avg', 'Antiparallel_beta_strand_avg', 'Hphob__Guy_avg', 'Ratio_hetero_endside_avg', 'Hphob_HPLC__Parker_&_al_avg', 'Hphob__Tanford_avg', '%_buried_residues_avg', 'Recognition_factors_avg', '%_accessible_residues_avg']
final_model = joblib.load("final_model.pkl")
results = ref[cols_to_keep]  # keep important features found by RFE

results_norm = results.copy()
results_num = results_norm.select_dtypes(include='number')
num_cols = list(results_num)
# remove label and position from numerical columns
num_cols.remove('Position')
#normalize data
for c in num_cols:
    results_norm[c] = (results[c] - results[c].min()) / (results[c].max() - results[c].min())
#get features
cols_to_drop= ["AA"]
results_features = results_norm.drop(cols_to_drop, axis=1)

#load final model and make predictions
final_model = joblib.load("final_model.pkl")
results_preds = list(final_model.predict(results_features))
positions = list(results_features['Position'])

seq_rows=[]
for i in range(len(results_preds)):
    temp1 = positions[i]
    temp2 = results_preds[i]
    temp_list=[temp1, temp2]
    seq_rows.append(temp_list)

##
import os
os.chdir("/home/local/CORNELL/esh76/final")
import pandas as pd
import joblib
seq = 'MSSCI'
seq_exist = False
seq_rows=[]
if seq != 0:
    seq_exist = True
    AA_list=[]
    for char in seq:
        AA_list.append(char)
    Position_list=[]
    counter = 1
    for item in AA_list:
        Position_list.append(counter)
        counter +=1
    df = pd.DataFrame(columns =['Position', 'AA'])
    df['Position']=Position_list
    df['AA']=AA_list
    features =  pd.read_csv('Expasy_AA_Scales.txt', sep="\t")
    features= features.rename(columns={"Amino Acid": "AA"})
    ref = pd.merge(features, df)
    ref=ref.sort_values(by=['Position'])
    cols = list(ref.columns)
    cols.remove("Position")
    cols.remove("AA")
    new_cols=[]
    for col in cols:
        new_name = col + "_avg"
        new_cols.append(new_name)
        ref[new_name] = ref[col].transform(lambda x: x.rolling(5, center=True).mean())
    for new_col in new_cols:
        original_col = new_col[:len(new_col)-4]
        ref.loc[ref[new_col].isna(), new_col] = ref[original_col]
    ref = ref.dropna()
    #variables needed that were calculated earlier
    cols_to_keep = ['Position', 'AA', 'HPLC__retention_pH_74_avg', 'beta_sheet__Levitt_avg', 'Hphob__Wolfenden_et_al_avg', 'beta_turn__Deleage_&_Roux_avg', 'Bulkiness_avg', 'Number_of_codons_avg', 'Hphob__Bull_&_Breese_avg', 'Hphob__Hopp_&_Woods_avg', 'AA_comp_in_Swiss_Prot_avg', 'beta_turn__Levitt_avg', 'Refractivity_avg', 'Relative_mutability_avg', 'Molecular_weight_avg', 'alpha_helix__Levitt_avg', 'alpha_helix__Chou_&_Fasman_avg', 'Total_beta_strand_avg', 'Antiparallel_beta_strand_avg', 'Hphob__Guy_avg', 'Ratio_hetero_endside_avg', 'Hphob_HPLC__Parker_&_al_avg', 'Hphob__Tanford_avg', '%_buried_residues_avg', 'Recognition_factors_avg', '%_accessible_residues_avg']
    final_model = joblib.load("final_model.pkl")
    results = ref[cols_to_keep]
    results_norm = results.copy()
    results_num = results_norm.select_dtypes(include='number')
    num_cols = list(results_num)
    for c in num_cols:
        results_norm[c] = (results[c] - results[c].min()) / (results[c].max() - results[c].min())
    cols_to_drop= ["AA"]
    results_features = results_norm.drop(cols_to_drop, axis=1)
    final_model = joblib.load("final_model.pkl")
    results_preds = list(final_model.predict(results_features))
    positions = list(results['Position'])
    for i in range(len(results_preds)):
        temp1 = positions[i]
        temp2 = results_preds[i]
        temp_list=[temp1, temp2]
        seq_rows.append(temp_list)