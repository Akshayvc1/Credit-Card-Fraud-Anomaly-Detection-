import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix

data = pd.read_csv('creditcard.csv')
x = data.iloc[:,:-1].values
y = data.iloc[:,-1].values

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)

classifiers = []
c1 = xgboost.XGBClassifier()
classifiers.append(c1)

# c2 = SVC(kernel='rbf')
# classifiers.append(c2)

# c3 = GaussianNB()
# classifiers.append(c3)

# c4 = DecisionTreeClassifier(criterion='entropy')
# classifiers.append(c4)

# c5 = RandomForestClassifier(n_estimators=100, criterion='entropy')
# classifiers.append(c5)

# c6 = GradientBoostingClassifier()
# classifiers.append(c6)


for clf in classifiers:
    clf.fit(x_train,y_train)
    y_pred = clf.predict(x_test)
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    print()
    


    




