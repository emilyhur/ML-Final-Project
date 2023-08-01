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
labels = pd.read_csv('Labels.txt', sep="\t")
uniprot2seq = pd.read_csv('UniProt2Seq.txt', sep="\t")
features =  pd.read_csv('Expasy_AA_Scales.txt', sep="\t")
features= features.rename(columns={"Amino Acid": "AA"}) #rename column for merging
new_data=pd.merge(labels, uniprot2seq)
df=pd.merge(new_data, features) #drops missing value
##
df=df.sort_values(by=['UniProt', 'Position'])
test_cols = list(df.columns)
test_cols.remove("UniProt")
test_cols.remove("Position")
test_cols.remove("Label")
test_cols.remove("AA")
test_grouped = df.groupby("UniProt")
df_test=df.copy()
for col in test_cols:
    new_name = col + "_avg"
    df_test[new_name] = df[col].transform(lambda x: x.rolling(5, center=True, min_periods=5).mean())
df_test=df_test.drop(test_cols, axis=1) #drop original columns
df_test=df_test.dropna() #drop rows with na
##
train_set, test_set = train_test_split(
	df_test,
	test_size=0.2,
	random_state=0)
trained_copy = train_set.copy()
##
alpha= train_set[train_set.Label==1]
not_alpha= train_set[train_set.Label==0]
downsample = resample(not_alpha,
                          replace=False,
                          n_samples=len(alpha),
                          random_state=0)
downsampled_data=pd.concat([alpha, downsample])
##
#high correlation filter (it's redundant to have 2 highly correlated columns)
corr_matrix1 = downsampled_data.corr().abs() #take absolute value of correlation matrix
#Reduce matrix to lower triangular shape
mask1 = np.triu(np.ones_like(corr_matrix1, dtype=bool))
df_triangular1 = corr_matrix1.mask(mask1)
# Look at combinations where correlation threshold >.95 and drop one of the variables
cols_to_drop1=[]
for c in df_triangular1.columns:
	if any (df_triangular1[c]>.95):
		cols_to_drop1.append(c)

#new data frame with one of each highly correlated pair of columns dropped
new_df1 = downsampled_data.drop(cols_to_drop1, axis=1)

##
df_normalized1 = new_df1.copy()
df_numerical = df_normalized1.select_dtypes(include='number')
col_names=list(df_numerical.columns)

#normalize data to calculate variances
for c in col_names:
	df_normalized1[c]= (new_df1[c] - new_df1[c].min()) / (new_df1[c].max() - new_df1[c].min())
##
#Recursive Feature Elimination– Greedy Method
model = LogisticRegression(max_iter=1000)
selector1=RFE(estimator=model, n_features_to_select=23, step=1) #chose arbitrarily to select half of current number of features

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

labels = ['UniProt', "Position", 'Label', 'AA']
labels = labels + important_cols
df_filtered1= df_normalized1[labels]
##
cols_to_drop= ['UniProt', "AA", "Label"]
features1 = df_filtered1.drop(cols_to_drop, axis=1)
target1 = df_filtered1.Label
LR = LogisticRegression(max_iter=1000)
kf = KFold(n_splits=10, random_state=0, shuffle=True)
result = cross_val_score(LR, features1, target1, cv=kf, scoring='accuracy')
print('Accuracy: %.3f (%.3f)' % (mean(result), std(result)))
##
NB = GaussianNB()
result3 = cross_val_score(NB, features1, target1, cv=10, scoring='accuracy')
print('Accuracy: %.3f (%.3f)' % (mean(result3), std(result3)))
##
RF = RandomForestClassifier()
# evaluate the model
cv = KFold(n_splits=10, random_state=1, shuffle=True)
result2 = cross_val_score(RF, features1, target1, cv=cv, scoring='accuracy')