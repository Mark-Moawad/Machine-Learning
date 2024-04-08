import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import linear_model

df = pd.read_csv("../FuelConsumptionCo2.csv")

# Print a glimpse of the data
print(df.head())

# Summarize the data
print(df.describe())

cdf = df[
    ['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_CITY', 'FUELCONSUMPTION_HWY', 'FUELCONSUMPTION_COMB', 'CO2EMISSIONS']]
print(cdf.head(9))

# Create a scatter plot of ENGINESIZE vs CO2EMISSIONS
plt.scatter(cdf.ENGINESIZE, cdf.CO2EMISSIONS, color='blue')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show()

# Create a mask to select random rows, split dataset into 80% training set and 20% testing set
msk = np.random.rand(len(cdf)) < 0.8
train_data = cdf[msk]
test_data = cdf[~msk]

X_train = np.asanyarray(train_data[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB']])
y_train = np.asanyarray(train_data[['CO2EMISSIONS']])

# Check the dimensions of the arrays before fitting the model
print("X_train shape: ", X_train.shape)
print("y_train shape: ", y_train.shape)

regr = linear_model.LinearRegression()
regr.fit(X_train, y_train)

# Print the coefficients
print('Coefficients: ', regr.coef_)

# Calculate the mean squared error (MSE) and explained variance score for the model
X_test = np.asanyarray(test_data[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB']])
y_test = np.asanyarray(test_data[['CO2EMISSIONS']])

y_pred = regr.predict(X_test)

print("Mean Squared Error (MSE) : %.2f" % np.mean((y_pred - y_test) ** 2))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % regr.score(X_test, y_test))

# Repeat the process using the features ENGINESIZE, CYLINDERS, FUELCONSUMPTION_CITY, and FUELCONSUMPTION_HWY
# to compare the accuracy
X_train = np.asanyarray(train_data[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_CITY', 'FUELCONSUMPTION_HWY']])
y_train = np.asanyarray(train_data[['CO2EMISSIONS']])

# Check the dimensions of the arrays before fitting the model
print("X_train shape: ", X_train.shape)
print("y_train shape: ", y_train.shape)

regr = linear_model.LinearRegression()
regr.fit(X_train, y_train)

# Print the coefficients of the regression model
print('Coefficients: ', regr.coef_)

# Calculate the mean squared error (MSE) and explained variance score for the model
X_test = np.asanyarray(test_data[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_CITY', 'FUELCONSUMPTION_HWY']])
y_test = np.asanyarray(test_data[['CO2EMISSIONS']])

y_pred = regr.predict(X_test)

print("Mean Squared Error (MSE) : %.2f" % np.mean((y_pred - y_test) ** 2))
print('Variance score: %.2f' % regr.score(X_test, y_test))
