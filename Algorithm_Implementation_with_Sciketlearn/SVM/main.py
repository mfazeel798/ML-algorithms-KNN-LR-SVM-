from sklearn import datasets
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# SVM implementation to detect Flower Name

iris = datasets.load_iris()                                 # loading data from dataset

# getting data in X and Y
X = iris.data
y = iris.target

classes = ['Iris Setosa', 'Iris Versicolor', 'Iris Veirginicaa']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


model = svm.SVC()
model.fit(X_train, y_train)
pred = model.predict(X_test)
acc = accuracy_score(y_test, pred)

print('predictions : ', pred)
print('accuracy : ', acc)


for i in range(len(pred)):
    print(classes[pred[i]])
