# SVM with the Iris Flowers dataset

from sklearn import datasets # for dataset
import pandas as pd # for data processing
from sklearn import svm # for ML prediction
from sklearn.model_selection import train_test_split
import plotly.express as px
from sklearn import metrics

# import and prepare dataset
iris = datasets.load_iris(as_frame=True)
df = pd.DataFrame(iris.data, columns=iris.feature_names)

flower_value = []
for i in iris.target:
    flower_value.append(i)

flower_names = []
for i in range(len(flower_value)):
    if flower_value[i] == 0:
        flower_names.append('setosa')
    elif flower_value[i] == 1:
        flower_names.append('versicolor')
    else:
        flower_names.append('virginica')

df['flower value'] = flower_value
df['flower name'] = flower_names
# print(df)

# Exploratory Data Analysis
df_distribution = df['flower value'].value_counts()
df_percentage_distribution = df['flower value'].value_counts()/float(len(df))
missing_values = df.isnull().sum()
# print(df_distribution)
# print(df_percentage_distribution)
# print(missing_values)

# plot histograms to check distribution
hist_plot_slength = px.histogram(data_frame=df, nbins=20, x=df['sepal length (cm)'], color=df['flower name'],
                         title='Distribution of Sepal Length by Iris Flower')
hist_plot_swidth = px.histogram(data_frame=df, nbins=20, x=df['sepal width (cm)'], color=df['flower name'],
                         title='Distribution of Sepal Width by Iris Flower')
hist_plot_plength = px.histogram(data_frame=df, nbins=20, x=df['petal length (cm)'], color=df['flower name'],
                         title='Distribution of Petal Length by Iris Flower')
hist_plot_pwidth = px.histogram(data_frame=df, nbins=20, x=df['petal width (cm)'], color=df['flower name'],
                         title='Distribution of Petal Width by Iris Flower')
# hist_plot_slength.show()
# hist_plot_swidth.show()
# hist_plot_plength.show()
# hist_plot_pwidth.show()

# declare feature vector and target variable
X = df.drop(labels=['flower value', 'flower name'], axis= 1)
y = df['flower value']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)
# print(X_train.shape, X_test.shape)

# instantiate classifier with default hyperparameters & fit classifier to training set & make predictions on test set
svc_default = svm.SVC()
svc_default.fit(X_train,y_train)
y_pred_default = svc_default.predict(X_test)

# instantiate classifier with rbf kernel and C=100 & fit classifier to training set & make predictions on test set
svc_rbf = svm.SVC(C=100.0)
svc_rbf.fit(X_train,y_train)
y_pred_rbf = svc_rbf.predict(X_test)

# instantiate classifier with linear kernel and C=1.0
svc_linear = svm.SVC(kernel='linear', C=1.0)
svc_linear.fit(X_train,y_train)
y_pred_linear = svc_rbf.predict(X_test)

# instantiate classifier with polynomial kernel and C=1.0
svc_poly = svm.SVC(kernel='poly', C=1.0)
svc_poly.fit(X_train,y_train)
y_pred_poly = svc_rbf.predict(X_test)

# compute and print accuracy score
print('Model accuracy score with default hyperparameters: {0:0.4f}'.format(metrics.accuracy_score(y_test, y_pred_default)))
print('Model accuracy score with rbf kernel and C=100: {0:0.4f}'.format(metrics.accuracy_score(y_test, y_pred_rbf)))
print('Model accuracy score with linear kernel and C=1.0: {0:0.4f}'.format(metrics.accuracy_score(y_test, y_pred_linear)))
print('Model accuracy score with polynomial kernel and C=1.0: {0:0.4f}'.format(metrics.accuracy_score(y_test, y_pred_poly)))

print('Confusion Matrix with default hyperparameters:', metrics.confusion_matrix(y_test, y_pred_default))
print('Confusion Matrix with rbf kernel and C=100:', metrics.confusion_matrix(y_test, y_pred_rbf))
print('Confusion Matrix with linear kernel and C=1.0:', metrics.confusion_matrix(y_test, y_pred_linear))
print('Confusion Matrix with polynomial kernel and C=1.0:', metrics.confusion_matrix(y_test, y_pred_poly))