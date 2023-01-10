# Naive Bayes using a diabetes dataset obtained through kaggle

import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler

# import data
# dataset contains 3 columns (glucose, blood pressure, diabetes) and 995 entries
# https://www.kaggle.com/datasets/himanshunakrani/naive-bayes-classification-data?resource=download
df = pd.read_csv('/.../diabetes_dataset.csv')

# split into X and y variable
X = df.drop(columns=['diabetes'])
y = df.diabetes
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# normalize features with min-max-scaler into range [0,1]
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# specify and fit the model
model = GaussianNB()
model.fit(X_train, y_train)

# cross validation
scores = cross_val_score(model, X_train, y_train, cv=5)
print('Cross Validation Scores:', scores)
print('Mean CrossVal Score:', scores.mean())

# prediction
train_prediction = model.predict(X_train)
test_prediction = model.predict(X_test)

# performance with training data
print('Prediction Score - Training:', model.score(X_train, y_train))
print('Confusion Matrix:', metrics.confusion_matrix(y_train, train_prediction))
print('Accuracy:', metrics.accuracy_score(y_train,train_prediction))
print(metrics.classification_report(y_train, train_prediction, digits=2))

# performance with testing data
print('Prediction Score - Testing:', model.score(X_test, y_test))
print('Confusion Matrix:', metrics.confusion_matrix(y_test, test_prediction))
print('Accuracy:', metrics.accuracy_score(y_test,test_prediction))
print(metrics.classification_report(y_test, test_prediction, digits=2))


