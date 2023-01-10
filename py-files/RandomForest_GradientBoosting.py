# Random Forest and Gradient Boosting with Breast Cancer Dataset

from sklearn.datasets import load_breast_cancer
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import plotly.express as px

# load and prepare standard dataset
dataset = load_breast_cancer()
df = pd.DataFrame(dataset.data, columns=dataset.feature_names)

diagnosis_value = []
for i in dataset.target:
    diagnosis_value.append(i)
df['diagnosis value'] = diagnosis_value

diagnosis_names = []
for i in range(len(diagnosis_value)):
    if diagnosis_value[i] == '1':
        diagnosis_names.append('malignant')
    else:
        diagnosis_names.append('benign')
df['diagnosis name'] = diagnosis_names

# Exploratory Data Analysis
df_distribution = df['diagnosis value'].value_counts()
df_percentage_distribution = df['diagnosis value'].value_counts()/float(len(df))
missing_values = df.isnull().sum()
#print(df_distribution)
#print(df_percentage_distribution)
#print(missing_values)

## RANDOM FOREST
# declare feature vector and target variable
X = df.drop(labels=['diagnosis value', 'diagnosis name'], axis= 1)
y = df['diagnosis value']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42, shuffle=True)
# print(X_train.shape, X_test.shape)

# specify and train the model
classifier = RandomForestClassifier(n_estimators=400, max_depth=3, min_samples_leaf=10, random_state=42)
classifier.fit(X_train, y_train)

# use the model to predict values
y_pred_testing = classifier.predict(X_test)
print('Accurarcy Score for Testing Data:', metrics.accuracy_score(y_test, y_pred_testing))

conf_mat = metrics.confusion_matrix(y_test, y_pred_testing)
heatmap_conf = px.imshow(conf_mat, text_auto=True, labels=dict(x='True Values', y='Predicted Values', color='Count'),
                         x=['Actual Positive', 'Actual Negative'], y=['Predicted Positive', 'Predicted Negative'],
                         color_continuous_scale='blues')
heatmap_conf.update_xaxes(side="top")
heatmap_conf.show()

# extract feature importances
feature_scores = pd.Series(classifier.feature_importances_, index=X_train.columns).sort_values(ascending=False)
print('Feature Scores:', feature_scores)

# Make a horizontal bar chart to visualize feature importantance
bar_features = px.bar(y=feature_scores.index, x=feature_scores)
bar_features.show()


## GRADIENT BOOSTING
from sklearn.ensemble import GradientBoostingClassifier
# specify the classifier to be trained
clf = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)
clf.fit(X_train, y_train)

# use the model to predict values
y_pred_gb = clf.predict(X_test)

# print the confusion matrix
print('Confusion Matrix:')
print(metrics.confusion_matrix(y_test, y_pred_gb))
conf_mat_gb = metrics.confusion_matrix(y_test, y_pred_gb)
heatmap_conf_gb = px.imshow(conf_mat_gb, text_auto=True, labels=dict(x='True Values', y='Predicted Values', color='Count'),
                         x=['Actual Positive', 'Actual Negative'], y=['Predicted Positive', 'Predicted Negative'],
                         color_continuous_scale='oranges')
heatmap_conf_gb.update_xaxes(side="top")
heatmap_conf_gb.show()

# print accuracy
print('Accurarcy Score', metrics.accuracy_score(y_test, y_pred_gb))

# extract feature importances
feature_scores_gb = pd.Series(clf.feature_importances_,
    index=X_train.columns).sort_values(ascending=False)
print('Feature Scores:', feature_scores_gb)

# Make a horizontal bar chart to visualize feature importantance
color_scale = ['#ffa500']
bar_features_gb = px.bar(y=feature_scores_gb.index, x=feature_scores_gb, color_discrete_sequence=color_scale)
bar_features_gb.show()