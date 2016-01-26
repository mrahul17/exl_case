import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.cross_validation import train_test_split,cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
# remove comment from the line below if using ipython notebook
#%matplotlib inline

def read_data():

	train = pd.read_csv('input/exl_train.csv')
	test = pd.read_csv('input/exl_evaluation.csv')
	features = train.columns.tolist()
	return (train,test,features)
	# remove IND_BURGLAR_ALARM  and IND_SPRINKLER_SYSTEM

def clean_data(data):
	data['PREMIUM_AMOUNT'] = data['PREMIUM_AMOUNT'].fillna(data.mean()['PREMIUM_AMOUNT'])
	data = data.drop(['IND_BURGLAR_ALARM','IND_SPRINKLER_SYSTEM'],axis=1)

	# set state id for states
	mapping = {'TX':1,'NJ':2,'CA':3,'IL':4}
	data = data.replace({'STATE':mapping})

	agency_mapping = {'A':1,'B':2,'C':3,'D':4,'E':5}
	data = data.replace({'AGENCY_NAME':agency_mapping})

	# might remove in future
	
	data = data.drop(['QUOTE_NUM','ORIGINAL_QUOTE_DATE'],axis=1)

	return data
# quote date might be included in analysis
#y = data['IND_QUOTE_CONVERSION']
#X = data.drop(['IND_QUOTE_CONVERSION','QUOTE_NUM','ORIGINAL_QUOTE_DATE'],axis=1)

# from my gist https://gist.github.com/light94/95e07a651b82bbfdb28e




def knn_classifier():
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

def k_fold_cross_validation():
	k_range = range(1,31)
	k_scores = []
	for k in k_range:
		knn = KNeighborsClassifier(n_neighbors=k)
		scores = cross_val_score(knn,X,y,cv=10,scoring='accuracy')
		k_scores.append(scores.mean())
	print k_scores

	plt.plot(k_range,k_scores)
	plt.xlabel('Value for k in knn')
	plt.ylabel('Cross-validated accuracy')
	print k_scores.mean()


def logistic_regression(X,y):
	logreg = LogisticRegression()
	print cross_val_score(logreg,X,y,cv=10,scoring='accuracy').mean()


def gradient_boosting(X_train,y_train):
	clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,max_depth=1, random_state=0)
	clf.fit(X_train,y_train)
	return clf

if __name__ == '__main__':
	train,test,features = read_data()
	data = clean_data(train)
	y = data['IND_QUOTE_CONVERSION']
	X = data.drop(['IND_QUOTE_CONVERSION'],axis=1)
	X_train, X_test, y_train, y_test  = train_test_split(X, y, test_size=0.4)
	#print X.shape
	#print y.shape
	logreg = logistic_regression(X,y)
	clf = gradient_boosting(X_train,y_train)
	print clf.score(X_test,y_test)
	#logistic_regression(X,y)