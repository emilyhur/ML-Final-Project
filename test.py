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
#read in all three text files
labels = pd.read_csv('Labels.txt', sep="\t")
uniprot2seq = pd.read_csv('UniProt2Seq.txt', sep="\t")
features =  pd.read_csv('Expasy_AA_Scales.txt', sep="\t")
features= features.rename(columns={"Amino Acid": "AA"}) #rename column for merging
new_data=pd.merge(labels, uniprot2seq) #merge into one dataset
df=pd.merge(new_data, features) #drops missing value
ref = pd.merge(features, uniprot2seq) #for generation of new features later

##
# downsample data
alpha= df[df.Label==1]
not_alpha= df[df.Label==0]
downsample = resample(not_alpha,
                          replace=False,
                          n_samples=len(alpha),
                          random_state=0)
downsampled_data=pd.concat([alpha, downsample])

##
#split downsampled data into training and test sets
train_set, test_set = train_test_split(
	downsampled_data,
	test_size=0.2,
	random_state=0)
trained_copy = train_set.copy()

##
#calculate averages in ref file
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
    ref[new_name] = ref_grouped[col].transform(lambda x: x.rolling(5, center=True).mean())

#drop original columns
ref=ref.drop(cols, axis=1)
##
#merge original columns and new generated features
df_updated = pd.merge(trained_copy, ref)
df_updated=df_updated.dropna() #drop rows with na
df_updated= df_updated.drop(cols, axis=1) #drop original features
##
#high correlation filter (it's redundant to have 2 highly correlated columns)
corr_matrix1 = df_updated.corr().abs() #take absolute value of correlation matrix
#Reduce matrix to lower triangular shape
mask1 = np.triu(np.ones_like(corr_matrix1, dtype=bool))
df_triangular1 = corr_matrix1.mask(mask1)
# Look at combinations where correlation threshold >.95 and drop one of the variables
cols_to_drop1=[]
for c in df_triangular1.columns:
	if any (df_triangular1[c]>.95):
		cols_to_drop1.append(c)
#new data frame with one of each highly correlated pair of columns dropped
new_df1 = df_updated.drop(cols_to_drop1, axis=1)

##
#look at correlations with Label
corr_matrix = new_df1.corr()
correlations=(corr_matrix["Label"].sort_values(ascending=False)) #correlations with label
#correlations range from .3 to -.3 for downsampled dataset

##
#low variance filter

df_normalized1 = new_df1.copy()
df_numerical = df_normalized1.select_dtypes(include='number')
col_names=list(df_numerical.columns)
#remove label and position from numerical columns
col_names.remove('Position')
col_names.remove('Label')

#normalize data to calculate variances
for c in col_names:
	df_normalized1[c]= (new_df1[c] - new_df1[c].min()) / (new_df1[c].max() - new_df1[c].min())
variance1 = df_normalized1.var()

filtered_cols1 = [ ]
columns= new_df1.columns
for i in range(0,len(variance1)):
    if variance1[i]>=0.006: #threshold for variance is 1%, don't want to include constant variables
        filtered_cols1.append(columns[i])
# filtered_cols1 = columns, so none of the remaining columns are constant with a variance of 0
##
#Recursive Feature Elimination– Greedy Method
#starts with all features, then eliminates the least important ones
model = LogisticRegression(max_iter=1000)
selector1=RFE(estimator=model, n_features_to_select=24, step=1) #chose arbitrarily to select half of current number of features

#split normalized dataset into features and labels
Y1=df_normalized1['Label']
X1=df_normalized1.drop(['Label', 'UniProt', 'AA', 'Position'], axis=1)

#run RFE
rfe1 = selector1.fit_transform(X1, Y1)
filter1=(selector1.get_support())
##
#filter columns using data from RFE
filter1=list(filter1)
current_cols = list(X1.columns)

#figure out which columns to keep
important_cols=[]
for index in range(len(filter1)):
    if (filter1[index])==True:
        important_cols.append(current_cols[index])

##
#drop least important features from the dataset
cols_to_keep = ['UniProt', "Position", 'Label', 'AA']
cols_to_keep = cols_to_keep + important_cols
df_filtered1= df_normalized1[cols_to_keep] #averaged, features scaled/normalized

##
#prepare dataset for models
cols_to_drop= ['UniProt', "AA", "Label"]

#split into features and target
features1 = df_filtered1.drop(cols_to_drop, axis=1)
target1 = df_filtered1.Label

##
#Models and cross validation
#Model1: Logistic Regression
#useful for binary classification, like our label column
#good for a simple first model as it assumes linearity (also a disadvantage)
LR = LogisticRegression(max_iter=1000)
cv = KFold(n_splits=10, random_state=0, shuffle=True)
result = cross_val_score(LR, features1, target1, cv=cv, scoring='accuracy')
print('Accuracy: %.3f (%.3f)' % (mean(result), std(result)))
#Accuracy: 0.666 (0.004)

##
#Model2: random forest; suitable for classifying larger datasets
#(as opposed to decision trees)
#This builds on top of other models (ensemble learning)
RF = RandomForestClassifier()
# evaluate the model
cv2 = KFold(n_splits=10, random_state=1, shuffle=True)
result2 = cross_val_score(RF, features1, target1, cv=cv2, scoring='accuracy')
# report performance
print('Accuracy: %.3f (%.3f)' % (mean(result2), std(result2)))
#Accuracy: 0.634 (0.005)
##
#Model3 Naive Bayes
#fast model for classification that's easy to train
#not very sensitive to noise or to overfitting
#however, assumes features are independent, which is rarely the case
NB = GaussianNB()
cv3 = KFold(n_splits=10, random_state=0, shuffle=True)
result3 = cross_val_score(NB, features1, target1, cv=cv3, scoring='accuracy')
print('Accuracy: %.3f (%.3f)' % (mean(result3), std(result3)))
#Accuracy: 0.643 (0.003)
##hyperparameter tuning
#the logistic regression model gave the highest accuracy with a small SD,
#so it's the one I selected. All of the accuracies were close, however.
h_model=LogisticRegression(max_iter=1000)
grid = dict()
grid['solver'] = ['newton-cg', 'lbfgs', 'liblinear']
grid['penalty'] = ['l1', 'l2', 'none']
grid['C'] = np.logspace(-4, 4, 10) #range of values to test

#5 folds for 90 candidates
grid_search = GridSearchCV(h_model, grid, cv = 5, scoring = 'accuracy',n_jobs=-1, verbose=True)
final= grid_search.fit(features1, target1)
print('Best Score: %s' % final.best_score_)
print('Best Hyperparameters: %s' % final.best_params_)
#Best Score: 0.6686708391786416
#Best Hyperparameters: {'C': 0.3593813663804626, 'penalty': 'l1', 'solver': 'liblinear'}
final_model = final.best_estimator_
#final_model = LogisticRegression(C=0.3593813663804626, max_iter=1000, penalty='l1', solver='liblinear')

##model persistence
joblib.dump(final_model, "final_model.pkl")

##evaluating test set
cols_to_keep = ['UniProt', 'Position', 'Label', 'AA', 'HPLC__retention_pH_74_avg', 'beta_sheet__Levitt_avg', 'Hphob__Wolfenden_et_al_avg', 'beta_turn__Deleage_&_Roux_avg', 'Bulkiness_avg', 'Number_of_codons_avg', 'Hphob__Bull_&_Breese_avg', 'Hphob__Hopp_&_Woods_avg', 'AA_comp_in_Swiss_Prot_avg', 'beta_turn__Levitt_avg', 'Refractivity_avg', 'Relative_mutability_avg', 'Molecular_weight_avg', 'alpha_helix__Levitt_avg', 'alpha_helix__Chou_&_Fasman_avg', 'Total_beta_strand_avg', 'Antiparallel_beta_strand_avg', 'Hphob__Guy_avg', 'Ratio_hetero_endside_avg', 'Hphob_HPLC__Parker_&_al_avg', 'Hphob__Tanford_avg', '%_buried_residues_avg', 'Recognition_factors_avg', '%_accessible_residues_avg']
test_copy= test_set.copy()
df_test = pd.merge(test_copy, ref) #merge with averages
df_test=df_test.dropna() #drop rows with na
df_test= df_test[cols_to_keep] #keep important features found by RFE
df_test_normalized= df_test.copy()
df_test_numerical = df_test_normalized.select_dtypes(include='number')
current_columns=list(df_test_numerical)
#remove label and position from numerical columns
current_columns.remove('Position')
current_columns.remove('Label')

for c in current_columns:
	df_test_normalized[c]= (df_test[c] - df_test[c].min()) / (df_test[c].max() - df_test[c].min())
##
#prepare dataset by splitting into features and label
cols_to_drop= ['UniProt', "AA", "Label"]
X_test = df_test_normalized.drop(cols_to_drop, axis=1)
Y_test= df_test_normalized.Label
##
y_preds = final_model.predict_proba(X_test)[:,1]
fpr, tpr, _ = roc_curve(Y_test,  y_preds)
auc = roc_auc_score(Y_test, y_preds)
##
#create figure for ROC curve

plt.plot(fpr,tpr,label="AUC="+str(auc))
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.legend(loc=4)
plt.savefig('roc_curve.png')
##
#plot pr curve
precision, recall, thresholds = precision_recall_curve(Y_test, y_preds)
plt.plot(recall, precision)
plt.ylabel('Precision')
plt.xlabel('Recall')
plt.savefig('pr.png')

##
#creating a function for test set 2 called return_auc
#first, repeated code for all variables needed

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
for col in cols:
    new_name = col + "_avg"
    ref[new_name] = ref_grouped[col].transform(lambda x: x.rolling(5, center=True).mean())

#drop original columns
ref=ref.drop(cols, axis=1)

#variables needed that were calculated earlier
cols_to_drop= ['UniProt', "AA", "Label"]
cols_to_keep = ['UniProt', 'Position', 'Label', 'AA', 'HPLC__retention_pH_74_avg', 'beta_sheet__Levitt_avg', 'Hphob__Wolfenden_et_al_avg', 'beta_turn__Deleage_&_Roux_avg', 'Bulkiness_avg', 'Number_of_codons_avg', 'Hphob__Bull_&_Breese_avg', 'Hphob__Hopp_&_Woods_avg', 'AA_comp_in_Swiss_Prot_avg', 'beta_turn__Levitt_avg', 'Refractivity_avg', 'Relative_mutability_avg', 'Molecular_weight_avg', 'alpha_helix__Levitt_avg', 'alpha_helix__Chou_&_Fasman_avg', 'Total_beta_strand_avg', 'Antiparallel_beta_strand_avg', 'Hphob__Guy_avg', 'Ratio_hetero_endside_avg', 'Hphob_HPLC__Parker_&_al_avg', 'Hphob__Tanford_avg', '%_buried_residues_avg', 'Recognition_factors_avg', '%_accessible_residues_avg']
final_model = joblib.load("final_model.pkl")

def return_auc(filename):
    labels_test = pd.read_csv(filename, sep="\t")
    temp = pd.merge(labels_test, uniprot2seq)  # merge into one dataset
    dataset = pd.merge(temp, features)
    # drop original columns
    copy = dataset.copy()
    test_data = pd.merge(copy, ref)  # merge with averages
    test_data = test_data.dropna() #drop rows with na
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
    auc_test = roc_auc_score(label_test, y_score)
    return auc_test

##
#save prediction results for UniProt2Seq in a file



