# k-nearest neighbor classification with breast cancer dataset

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import pandas as pd
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

# load and prepare standard dataset
dataset = load_breast_cancer()
X = pd.DataFrame(dataset.data, columns=dataset.feature_names)
y = pd.Categorical.from_codes(dataset.target, dataset.target_names)

# create dataframe
diagnosis_value = []
for diagnosis in y:
    if diagnosis == 'malignant':
        diagnosis_value.append(1)
    else:
        diagnosis_value.append(0)

y = diagnosis_value
#df = X
#df['diagnosis'] = y
#df['diagnosis_value'] = diagnosis_value

# split, train, test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

knn = KNeighborsClassifier(n_neighbors=13, metric='manhattan')
knn.fit(X_train, y_train)

# %% use the model to predict values
y_prediction = knn.predict(X_test)

# performance of model
print('Confusion Matrix:', metrics.confusion_matrix(y_test, y_prediction))
print('Accuracy:', metrics.accuracy_score(y_test,y_prediction))
print(metrics.classification_report(y_test, y_prediction, digits=2))
print('Prediction Score:', knn.score(X_test, y_test))

# performing cross validation
neighbors = []
cv_scores = []

# perform 10 fold cross validation
for k in range(1, 51, 2):
    neighbors.append(k)
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(
        knn, X_train, y_train, cv=10, scoring='accuracy')
    cv_scores.append(scores.mean())
MSE = [1 - x for x in cv_scores]

# determining the best k
optimal_k = neighbors[MSE.index(min(MSE))]
print('The optimal number of neighbors is % d ' % optimal_k)

# plot misclassification error versus k
plt.figure(figsize = (10, 6))
plt.plot(neighbors, MSE)
plt.xlabel('Number of neighbors')
plt.ylabel('Misclassification Error')
plt.show()


