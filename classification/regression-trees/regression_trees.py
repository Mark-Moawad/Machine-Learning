"""
A real estate company is planning to invest in Boston real estate. Information about various areas of Boston has been
collected. This code aims to train a regression tree model that can predict the median price of houses for that area,
so it can be used to make offers.

The dataset has information on areas/towns not individual houses, the features are:

-CRIM: Crime per capita
-ZN: Proportion of residential land zoned for lots over 25,000 sq.ft.
-INDUS: Proportion of non-retail business acres per town
-CHAS: Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
-NOX: Nitric oxides concentration (parts per 10 million)
-RM: Average number of rooms per dwelling
-AGE: Proportion of owner-occupied units built prior to 1940
-DIS: Weighted distances to five Boston employment centers
-RAD: Index of accessibility to radial highways
-TAX: Full-value property-tax rate per $10,000
-PTRAIO: Pupil-teacher ratio by town
-LSTAT: Percent lower status of the population
-MEDV: Median value of owner-occupied homes in $1000s
"""

# Pandas allows the creation of a dataframe of the data, so it can be used and manipulated
import pandas as pd
# Regression Tree Algorithm
from sklearn.tree import DecisionTreeRegressor
# Split the data into a training and testing data
from sklearn.model_selection import train_test_split


# Importing Data
data = pd.read_csv('real_estate_data.csv')

# Inspecting Data
print(data.head())  # Show first 5 rows of the dataset
print(data.shape)  # Show the shape of the dataset
print(data.isna().sum())  # Show the count of invalid (N/A) values of each feature

# Data Pre-processing
data.dropna(inplace=True)  # Remove invalid values
print(data.isna().sum())  # Show the count of invalid (N/A) values of each feature

# Splitting the data into independent features and target feature
X = data.drop(columns=["MEDV"])
Y = data["MEDV"]

print(X.head())
print(Y.head())

# Splitting the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.2, random_state=1)

"""
The important parameters of 'DecisionTreeRegressor' are:

'criterion': {'absolute_error', 'squared_error', 'poisson', 'friedman_mse'} - The function used to measure error

'max_depth' - The max depth the tree can be

'min_samples_split' - The minimum number of samples required to split a node

'min_samples_leaf' - The minimum number of samples that a leaf can contain

'max_features': {"auto", "sqrt", "log2"} - The number of features to be examined, looking
for the best one (used to speed up training)
"""
# Model Creation using mean square error (MSE)
regression_tree = DecisionTreeRegressor(criterion='squared_error')

# Training Model
regression_tree.fit(X_train, Y_train)

# Evaluation
print(regression_tree.score(X_test, Y_test))
prediction = regression_tree.predict(X_test)
print("$", (prediction - Y_test).abs().mean() * 1000)


# Model Creation using mean absolute error (MAE)
regression_tree = DecisionTreeRegressor(criterion='absolute_error')

# Training Model
regression_tree.fit(X_train, Y_train)

# Evaluation
print(regression_tree.score(X_test, Y_test))
prediction = regression_tree.predict(X_test)
print("$", (prediction - Y_test).abs().mean() * 1000)
