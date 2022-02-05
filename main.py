# below data is adapted from https://towardsdatascience.com/building-a-logistic-regression-in-python-step-by-step-becd4d56c9c8
# by Clara Mingyu Wan
# 4 February 2022

import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt

plt.rc("font", size=14)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import seaborn as sns
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)

###########read the data
#data = pd.read_csv('pos_data.csv',header=0) # try pos feature
#data = pd.read_csv('bows.csv',header=0) # try bow feature
#data = pd.read_csv('w2v.csv',header=0) # try w2v feature
data = pd.read_csv('epa.csv',header=0) # try epa feature

data_final = data.dropna()

#print(data.shape) #(41188, 21)
#print(data.head()) #
#print(data['education'].unique()) #['basic.4y' 'unknown' 'university.degree' 'high.school' 'basic.9y'  'professional.course' 'basic.6y' 'illiterate']
'''
###########creat dummy variables
cat_vars=['job','marital','education','default','housing','loan','contact','month','day_of_week','poutcome']
for var in cat_vars:
    cat_list='var'+'_'+var
    cat_list = pd.get_dummies(data[var], prefix=var)
    data1=data.join(cat_list)
    data=data1

###########data selection
cat_vars=['job','marital','education','default','housing','loan','contact','month','day_of_week','poutcome']
data_vars=data.columns.values.tolist()
to_keep=[i for i in data_vars if i not in cat_vars]

data_final=data[to_keep]
#print(data_final.columns.values)
#print(data.columns.values)

###########one sampling using SMOTE
'''
X = data_final.loc[:, data_final.columns != 'Truth']
y = data_final.loc[:, data_final.columns == 'Truth']
from imblearn.over_sampling import SMOTE
os = SMOTE(random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
columns = X_train.columns
os_data_X,os_data_y=os.fit_resample(X_train, y_train)
os_data_X = pd.DataFrame(data=os_data_X,columns=columns )
os_data_y= pd.DataFrame(data=os_data_y,columns=['y'])
# we can Check the numbers of our data
# print("length of oversampled data is ",len(os_data_X))
# print("Number of no subscription in oversampled data",len(os_data_y[os_data_y['y']==0]))
# print("Number of subscription",len(os_data_y[os_data_y['y']==1]))
# print("Proportion of no subscription data in oversampled data is ",len(os_data_y[os_data_y['y']==0])/len(os_data_X))
# print("Proportion of subscription data in oversampled data is ",len(os_data_y[os_data_y['y']==1])/len(os_data_X))

###########Implementing the model
#import statsmodels.api as sm
#logit_model=sm.Logit(y,X)
#result=logit_model.fit()
#print(result.summary2())

###########Model Fitting
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier as rfc

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
logreg = LogisticRegression()
logreg.fit(X_train, y_train)

#svmclasifier = svm.SVC()
#svmclasifier.fit(X_train, y_train)

#rfcclassifier = rfc()
#rfcclassifier.fit(X_train, y_train)

y_pred = logreg.predict(X_test)
#y_pred = svmclasifier.predict(X_test)
#y_pred = rfcclassifier.predict(X_test)
print('Accuracy of classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))
#print('Accuracy of classifier on test set: {:.2f}'.format(svmclasifier.score(X_test, y_test)))
#print('Accuracy of classifier on test set: {:.2f}'.format(rfcclassifier.score(X_test, y_test)))

###########Confusion Matrix
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)

###########Compute precision, recall, F-measure and support
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

