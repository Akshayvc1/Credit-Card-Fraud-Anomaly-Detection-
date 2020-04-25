import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost

data = pd.read_csv('creditcard.csv')
x = data.iloc[:,:-1].values
y = data.iloc[:,-1].values

from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=0)

sm = SMOTE(random_state=1)
x_train_sm, y_train_sm = sm.fit_sample(x_train, y_train)
clf = xgboost.XGBClassifier()
clf.fit(x_train_sm,y_train_sm)
y_pred = clf.predict(x_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)




