import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.cross_validation import train_test_split
# remove comment from the line below if using ipython notebook
#%matplotlib inline

data = pd.read_csv('input/exl_train.csv')

# remove IND_BURGLAR_ALARM  and IND_SPRINKLER_SYSTEM
data['PREMIUM_AMOUNT'] = data['PREMIUM_AMOUNT'].fillna(data.mean()['PREMIUM_AMOUNT'])
data = data.drop(['IND_BURGLAR_ALARM','IND_SPRINKLER_SYSTEM'],axis=1)

# set state id for states
mapping = {'TX':1,'NJ':2,'CA':3,'IL':4}
data = data.replace({'STATE':mapping})

agency_mapping = {'A':1,'B':2,'C':3,'D':4,'E':5}
data = data.replace({'AGENCY_NAME':agency_mapping})

# quote date might be included in analysis
y = data['IND_QUOTE_CONVERSION']
X = data.drop(['IND_QUOTE_CONVERSION','QUOTE_NUM','ORIGINAL_QUOTE_DATE'],axis=1)

# from my gist https://gist.github.com/light94/95e07a651b82bbfdb28e

X_train, X_test, y_train, y_test  = train_test_split(X, y, test_size=0.4)

k_range = range(1,26)
scores = []
for k in k_range:
  knn = KNeighborsClassifier(n_neighbors=k)
  knn.fit(X_train,y_train)
  y_pred = knn.predict(X_test)
  scores.append(metrics.accuracy_score(y_test,y_pred))

plt.plot(k_range,scores)
plt.xlabel('Value for k in knn')
plt.ylabel('Testing accuracy')