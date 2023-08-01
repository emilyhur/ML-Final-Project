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

##
#import and merge text files
labels = pd.read_csv('Labels.txt', sep="\t")
uniprot2seq = pd.read_csv('UniProt2Seq.txt', sep="\t")
features =  pd.read_csv('Expasy_AA_Scales.txt', sep="\t")
features= features.rename(columns={"Amino Acid": "AA"}) #rename column for merging
new_data=pd.merge(labels, uniprot2seq)
df=pd.merge(new_data, features) #drops missing value

##
#create training and test sets
train_set, test_set = train_test_split(
	df,
	test_size=0.2,
	random_state=0)
trained_copy = train_set.copy()
##
#fix imbalanced data with downsampling
alpha= train_set[train_set.Label==1]
not_alpha= train_set[train_set.Label==0]
downsample = resample(not_alpha,
                          replace=False,
                          n_samples=len(alpha),
                          random_state=0)
downsampled_data=pd.concat([alpha, downsample])
##
#feature engineering– Look at correlations with Label
df_downsampled= downsampled_data.copy()
corr_matrix = df_downsampled.corr()
correlations=(corr_matrix["Label"].sort_values(ascending=False)) #correlations with label
#correlations range from .18 to -.18 for downsampled dataset

##
#high correlation filter (it's redundant to have 2 highly correlated columns)
corr_matrix1 = df_downsampled.corr().abs() #take absolute value of correlation matrix
#Reduce matrix to lower triangular shape
mask1 = np.triu(np.ones_like(corr_matrix1, dtype=bool))
df_triangular1 = corr_matrix1.mask(mask1)
# Look at combinations where correlation threshold >.95 and drop one of the variables
cols_to_drop1=[]
for c in df_triangular1.columns:
	if any (df_triangular1[c]>.95):
		cols_to_drop1.append(c)

#new data frame with one of each highly correlated pair of columns dropped
new_df1 = df_downsampled.drop(cols_to_drop1, axis=1)
##
#low variance filter

df_normalized1 = new_df1.copy()
df_numerical = df_normalized1.select_dtypes(include='number')
col_names=list(df_numerical.columns)

#normalize data to calculate variances
for c in col_names:
	df_normalized1[c]= (new_df1[c] - new_df1[c].min()) / (new_df1[c].max() - new_df1[c].min())
variance1 = df_normalized1.var()

filtered_cols1 = [ ]
cols= new_df1.columns
for i in range(0,len(variance1)):
    if variance1[i]>=0.006: #threshold for variance is 1%, don't want to include constant variables
        filtered_cols1.append(cols[i])
# filtered_cols = cols, so none of the remaining columns are constant with a variance of 0

##
#Recursive Feature Elimination– Greedy Method
#starts with all features, then eliminates the least important ones
model = LogisticRegression(max_iter=1000)
selector1=RFE(estimator=model, n_features_to_select=24, step=1) #chose arbitrarily to select half of current number of features

#split dataset into features and labels
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
#want to work with original training set that's not downsampled for sequence context
trained_copy=trained_copy.drop(cols_to_drop1, axis=1) #drop columns using high correlation

##
#sort alphabetically, then by position
trained_copy=trained_copy.sort_values(by=['UniProt', 'Position'])
#create list of features to cosnider
columns_list = list(trained_copy.columns)
columns_list.remove("UniProt")
columns_list.remove("Position")
columns_list.remove("Label")
columns_list.remove("AA")

##
#Group data by UniProt ID
grouped_df = trained_copy.groupby("UniProt")
#using sliding window of size 5 ([pos+2, pos-2]), calculate average of each feature
for col in columns_list:
    new_name = col + "_avg"
    trained_copy[new_name] = grouped_df[col].transform(lambda x: x.rolling(5, center=True, min_periods=5).mean())
df_averaged=trained_copy.drop(columns_list, axis=1) #drop original columns
df_averaged=df_averaged.dropna() #drop rows with na

#rename columns back to original names
new_names=['UniProt', "Position", 'Label', 'AA']
new_names=new_names+columns_list
df_averaged.columns=new_names
##
#downsample averaged data
alpha2= df_averaged[df_averaged.Label==1]
not_alpha2= df_averaged[df_averaged.Label==0]
downsample2 = resample(not_alpha2,
                          replace=False,
                          n_samples=len(alpha2),
                          random_state=0)
downsampled_data2=pd.concat([alpha2, downsample2])
df_averaged=downsampled_data2.copy()

##normalize averaged data
df_normalized2 = df_averaged.copy()
df_numerical2 = df_normalized2.select_dtypes(include='number')
col_names2=list(df_numerical2.columns)

#normalize data to prepare for RFE
for c in col_names2:
	df_normalized2[c]= (df_averaged[c] - df_averaged[c].min()) / (df_averaged[c].max() - df_averaged[c].min())
##
#perform RFE again with dataset of averaged, normalized values
model = LogisticRegression(max_iter=1000)
new_selector=RFE(estimator=model, n_features_to_select=23, step=1) #again, select half of the features

#separate into features and labels
new_Y=df_normalized2['Label']
new_X=df_normalized2.drop(['Label', 'UniProt', 'AA', 'Position'], axis=1)

#run RFE again
rfe2 = new_selector.fit_transform(new_X, new_Y)
filter2=(new_selector.get_support())

##
#filter columns using data from RFE
filter2=list(filter2)
current_columns = list(new_X.columns)

#figure out which columns to drop
important_cols2=[]
for index in range(len(filter2)):
    if (filter2[index])==True:
        important_cols2.append(current_columns[index])

## compare top 23 features from both RFEs and find the ones in common
important_features=[]
for item in important_cols:
    if item in important_cols2:
        important_features.append(item)
#outputs list of 17 "most important" features common to both filtered lists

##
#drop other columns from both averaged and non-averaged datasets
labels = ['UniProt', "Position", 'Label', 'AA']
labels = labels + important_features
df_filtered1= df_normalized1[labels] #not averaged, features scaled/normalized
df_filtered2= df_normalized2[labels] #averaged, features scaled/normalized

##
#Models and cross validation
#Model1: Logistic Regression
#useful for binary classification, like our label column
#good for a simple first model as it assumes linearity (also a disadvantage)
cols_to_drop= ['UniProt', "AA", "Label"]

#produced better results with the averaged dataset
features1 = df_filtered1.drop(cols_to_drop, axis=1)
target1 = df_filtered1.Label
features2 = df_filtered2.drop(cols_to_drop, axis=1)
target2 = df_filtered2.Label
##
LR = LogisticRegression(max_iter=1000)
cv = KFold(n_splits=10, random_state=0, shuffle=True)
result = cross_val_score(LR, features1, target1, cv=cv, scoring='accuracy')
print('Accuracy: %.3f (%.3f)' % (mean(result), std(result)))
#Accuracy: 0.660 (0.004)

##
#Model2: random forest; suitable for classifying larger datasets
#(as opposed to decision trees)
#This builds on top of other models (ensemble learning)
RF = RandomForestClassifier()
# evaluate the model
cv2 = KFold(n_splits=10, random_state=1, shuffle=True)
result2 = cross_val_score(RF, features2, target2, cv=cv2, scoring='accuracy')
# report performance
print('Accuracy: %.3f (%.3f)' % (mean(result2), std(result2)))
#Accuracy: 0.630 (0.004)
##
#Model3
#fast model for classification that's easy to train
#not very sensitive to noise or to overfitting
NB = GaussianNB()
cv3 = KFold(n_splits=10, random_state=0, shuffle=True)
result3 = cross_val_score(NB, features1, target1, cv=cv3, scoring='accuracy')
print('Accuracy: %.3f (%.3f)' % (mean(result3), std(result3)))
#Accuracy: 0.639 (0.009)

##hyperparameter tuning
#the logistic regression model gave the highest accuracy with a small SD,
#so it's the one I selected. All of the accuracies were close, however.
from sklearn.model_selection import GridSearchCV
h_model=LogisticRegression(max_iter=1000)
grid = dict()
grid['solver'] = ['newton-cg', 'lbfgs', 'liblinear']
grid['penalty'] = ['l1', 'l2', 'none']
grid['C'] = np.logspace(-4, 4, 10) #range of values to test
grid_search = GridSearchCV(h_model, grid, cv = 5, scoring = 'accuracy',n_jobs=-1, verbose=True)
final= grid_search.fit(features2, target2)
print('Best Score: %s' % final.best_score_)
print('Best Hyperparameters: %s' % final.best_params_)
final_model = final.best_estimator_
##model persistence
joblib.dump(final_model, "final_model.pkl")
