import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import neighbors, metrics
from sklearn.preprocessing import LabelEncoder

# KNN Implementation Car data

data = pd.read_csv('car.data')
print(data.head())

X = data[[
    'buying',
    'maint',
    'safty'
]].values
y = data[['class']]

print(X,y)

# converting the data
#X
Le = LabelEncoder()
for i in range (len(X[0])):
    X[:, i] = Le.fit_transform(X[:, i])
print(X)

#y
label_mapping = {
    'unacc':0,
    'acc':1,
    'good':2,
    'vgood':3
}
y['class'] = y['class'].map(label_mapping)
y = np.array(y)
print(y)

# model

KNN = neighbors.KNeighborsClassifier(n_neighbors=25, weights='uniform')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

KNN.fit(X_train,y_train)

prediction = KNN.predict(X_test)

accuracy = metrics.accuracy_score(y_test, prediction)

print('''





''')
print(accuracy)
print('actual value = ', y[1724])
print('pridected value = ', KNN.predict(X)[1724])