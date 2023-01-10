# Linear & Logistic Regression with the Iris Flowers dataset

from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn import metrics
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# load standard dataset
iris = datasets.load_iris(as_frame=True)

# convert to pd dataframe
iris = pd.DataFrame(
    data= np.c_[iris['data'], iris['target']],
    columns= iris['feature_names'] + ['target']
    )

### LINEAR REGRESSION
# predict sepal length (cm) of the iris flowers
X = iris.drop(labels= 'sepal length (cm)', axis= 1) # everything beside the y we want to predict
y = iris['sepal length (cm)'] # the thing we want to predict

# split the data into training and testing
# test_size (between 0-1): represents proportion of dataset to include in test split
# random_state: results are reproducible; every time code is run, the same instances will be included; it doesn't matter which number is chosen
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# use only the training data set for model training
lr = linear_model.LinearRegression()
model = lr.fit(X_train, y_train)
prediction = lr.predict(X_test)
original = y_test

# the intercept and coefficients
intercept = lr.intercept_
coefficient = lr.coef_

# Evaluating Model's Performance
print('Mean Absolute Error (MAE):', mean_absolute_error(y_test, prediction))
print('Mean Squared Error (MSE):', mean_squared_error(y_test, prediction, squared=True))
print('Mean Root Squared Error (RMSE):', mean_squared_error(y_test, prediction, squared=False))
print('Mean Root Squared Error (RMSE):', np.sqrt(mean_squared_error(y_test, prediction)))
print('Coefficient of Determination R² - Testing:', lr.score(X_test, y_test)) # how much more accurate line is compared to mean
print('Cross Validation Score:', cross_val_score(model, X_train,y_train, cv=5))
print('Aggregated CV Score / R² - Training:', cross_val_score(model, X_train,y_train, cv=5).mean())

# Visualising the Prediction Result
result_df = original.to_frame(name='original')
result_df['prediction'] = prediction
result_df['absolute deviation'] = result_df.original - result_df.prediction
result_df['num'] = list(range(len(result_df)))

scatter1 = plt.scatter(result_df.num, result_df.original, label='original', color='#20E09C')
scatter2 = plt.scatter(result_df.num, result_df.prediction, label='prediction', color='#044EA1')
plt.title('Original vs. Predicted Values for Sepal Length (cm)')
plt.xlabel('Data Points')
plt.ylabel('Sepal Length (cm)')
plt.legend(loc='upper right')
plt.show()


### LOGISTIC REGRESSION
# predict target class (flower species) based on sepal and petal length and width
X_logreg = iris.drop(labels= 'target', axis= 1) # everything beside the y we want to predict
y_logreg = iris['target'] # the thing we want to predict

# split the data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X_logreg, y_logreg, test_size=0.5, random_state=42)

# use only the training data set for model training
logreg = linear_model.LogisticRegression()
logreg.fit(X_train, y_train)
training_prediction = logreg.predict(X_train)
test_prediction = logreg.predict(X_test)
test_prediction2 =logreg.predict_proba(X_test)
# If using predict() instead of predict_proba(),
# then only class that the model considers to be more probable is shown
original = y_test

# performance in training
#print(metrics.confusion_matrix(y_train, training_prediction))
#print(metrics.classification_report(y_train, training_prediction, digits=3))

# performance in testing
print('Confusion Matrix:', metrics.confusion_matrix(y_test, test_prediction))
print('Accuracy:', metrics.accuracy_score(y_test,test_prediction))
print('Recall:', metrics.recall_score(y_test,test_prediction, average='micro'))
print('Precision:', metrics.precision_score(y_test,test_prediction, average='micro'))
print(metrics.classification_report(y_test, test_prediction, digits=2))

