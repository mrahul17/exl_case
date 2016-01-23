import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

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

