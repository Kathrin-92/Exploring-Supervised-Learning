# Linear, Ridge, Lasso and Elastic Net Regression + GLM with Salary Dataset

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

# load standard data
years_experience = [1.1, 1.3, 1.5, 2.0, 2.2, 2.9, 3.0, 3.2, 3.2, 3.7, 3.9, 4.0, 4.0, 4.1, 4.5, 4.9, 5.1, 5.3, 5.9, 6.0,
                    6.8, 7.1, 7.9, 8.2, 8.7, 9.0, 9.5, 9.6, 10.3, 10.5, 1, 1.2, 1.4, 1.6, 1.8, 2, 2.2, 2.4, 2.6, 2.8, 3,
                    3.2, 3.4, 3.6, 3.8, 4, 4.2, 4.4, 4.6, 4.8, 5, 5.2, 5.4, 5.6, 5.8, 6, 6.2, 6.4, 6.6, 6.8, 7, 7.2,
                    7.4, 7.6, 7.8, 8, 8.2, 8.4, 8.6, 8.8, 9, 9.2, 9.4, 9.6, 9.8, 10, 10.2, 10.4, 10.6, 10.8, 11, 11.2,
                    11.4, 11.6, 11.8, 12]
age = [19, 19, 20, 20, 21, 22, 24, 24, 25, 26, 28, 30, 31, 28, 30, 31, 33, 34, 35, 36, 37, 38, 38, 37, 29, 30, 40, 41,
       42, 45, 19, 19, 20, 20, 21, 22, 24, 24, 25, 26, 28, 30, 31, 28, 30, 31, 33, 34, 35, 36, 37, 38, 38, 37, 29, 30,
       40, 41, 42, 45, 39, 39, 38, 41, 42, 29, 45, 45, 46, 47, 49, 50, 51, 49, 51, 50, 48, 39, 38, 37, 50, 51, 52, 49,
       38, 39]
salary = [39343.00, 46205.00, 37731.00, 43525.00, 39891.00, 56642.00, 60150.00, 54445.00, 64445.00, 57189.00, 63218.00,
          55794.00, 56957.00, 57081.00, 61111.00, 67938.00, 66029.00, 83088.00, 81363.00, 93940.00, 91738.00, 98273.00,
          101302.00, 113812.00, 109431.00, 105582.00, 116969.00, 112635.00, 122391.00, 121872.00, 30243, 31812, 33904,
          34427, 35996, 36519, 37042, 38611, 39134, 39657, 40180, 42795, 43318, 43841, 45933, 49594, 50117, 52732,
          53255, 55870, 56393, 56916, 57439, 60577, 61100, 64761, 65284, 65807, 68422, 68945, 72083, 72606, 73129, 73652,
          78359, 78882, 79405, 79928, 84112, 84635, 85158, 88296, 88819, 89342, 89865, 90388, 95618, 96141, 96664, 97187,
          97710, 98233, 104509, 105032, 105555, 109216]
data = list(zip(years_experience, age, salary))
df = pd.DataFrame(data, columns=['years_experience', 'age', 'salary'])

# show distribution of original data
plt.scatter(df.years_experience, df.salary)
plt.scatter(df.age, df.salary)

# predict the salary based on years of experience
X = df.drop(labels= 'salary', axis= 1) # everything beside the y we want to predict
y = df['salary'] # the thing we want to predict

# split the data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


### LINEAR REGRESSION
# train and fit
lr = LinearRegression()
linear_model = lr.fit(X_train, y_train)
linear_prediction = lr.predict(X_test)
original = y_test

# Evaluating Model's Performance
print('LINEAR REGRESSION:')
print('Mean Absolute Error (MAE):', mean_absolute_error(y_test, linear_prediction))
print('Mean Squared Error (MSE):', mean_squared_error(y_test, linear_prediction, squared=True))
print('Mean Root Squared Error (RMSE):', mean_squared_error(y_test, linear_prediction, squared=False))


###RIDGE REGRESSION
# train and fit
ridge_reg = Ridge(alpha=1) # alpha multiplies the L2 term, controlling regularization strength
ridge_model = ridge_reg.fit(X_train, y_train)
ridge_prediction = ridge_reg.predict(X_test)

# Evaluating Model's Performance
print('RIDGE REGRESSION:')
print('Mean Absolute Error (MAE):', mean_absolute_error(y_test, ridge_prediction))
print('Mean Squared Error (MSE):', mean_squared_error(y_test, ridge_prediction, squared=True))
print('Mean Root Squared Error (RMSE):', mean_squared_error(y_test, ridge_prediction, squared=False))


### LASSO REGRESSION
# train and fit
lasso_reg = Lasso(alpha=0.5) # alpha multiplies the L1 term; = 0: same coefficients as simple linear regression
lasso_model = lasso_reg.fit(X_train, y_train)
lasso_prediction = lasso_reg.predict(X_test)

# Evaluating Model's Performance
print('LASSO REGRESSION:')
print('Mean Absolute Error (MAE):', mean_absolute_error(y_test, lasso_prediction))
print('Mean Squared Error (MSE):', mean_squared_error(y_test, lasso_prediction, squared=True))
print('Mean Root Squared Error (RMSE):', mean_squared_error(y_test, lasso_prediction, squared=False))


### ELASTIC NET
elastic_net = ElasticNet(alpha=0.1, l1_ratio=0.5)
elastic_model = elastic_net.fit(X_train,y_train)
elastic_prediction = elastic_net.predict(X_test)

print('ELASTIC NET REGRESSION:')
print('Mean Absolute Error (MAE):', mean_absolute_error(y_test, elastic_prediction))
print('Mean Squared Error (MSE):', mean_squared_error(y_test, elastic_prediction, squared=True))
print('Mean Root Squared Error (RMSE):', mean_squared_error(y_test, elastic_prediction, squared=False))


### GENERALIZED LINEAR MODEL - GLM
# pecify exogeneous and endogeneous variables
exog, endog = sm.add_constant(X), y

# specify the model
glm_model = sm.GLM(endog, exog, family=sm.families.Poisson(link=sm.families.links.log()))

# fit the model
res = glm_model.fit()

# %% print model summary
print(res.summary())