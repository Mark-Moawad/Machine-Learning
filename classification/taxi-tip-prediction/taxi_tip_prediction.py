"""
A real dataset is used to train a Regression Tree model. The dataset includes information about taxi tip
and was collected and provided to the NYC Taxi and Limousine Commission (TLC) by technology providers authorized
under the Taxicab & Livery Passenger Enhancement Programs (TPEP/LPEP). The trained model will be used to predict
the amount of tip paid.

The model will be created in two ways, using the Scikit-Learn Python interface, and the Python API offered by
the Snap Machine Learning (Snap ML) library. Snap ML is a high-performance IBM library for ML modeling.
It provides highly-efficient CPU/GPU implementations of linear models and tree-based models.
Snap ML not only accelerates ML algorithms through system awareness, but it also offers novel ML algorithms
with best-in-class accuracy.
"""
from __future__ import print_function

import gc
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
# import the Decision Tree Regression Model from scikit-learn
from sklearn.tree import DecisionTreeRegressor as decisionTR
# import the Decision Tree Regressor Model from Snap ML
from snapml import DecisionTreeRegressor

warnings.filterwarnings('ignore')

# Dataset Analysis
# Importing Data
raw_data = pd.read_csv('yellow_tripdata_2019-06.csv')
print("There are " + str(len(raw_data)) + " observations in the dataset.")
print("There are " + str(len(raw_data.columns)) + " variables in the dataset.")

# Inspecting Data
print(raw_data.head())

# Data Cleaning
# some trips report 0 tip. it is assumed that these tips were paid in cash.
# for this study all these rows will be dropped
raw_data = raw_data[raw_data['tip_amount'] > 0]

# Removing some outliers, namely those where the tip was larger than the fare cost
raw_data = raw_data[(raw_data['tip_amount'] <= raw_data['fare_amount'])]

# Removing trips with very large fare cost
raw_data = raw_data[((raw_data['fare_amount'] >= 2) & (raw_data['fare_amount'] < 200))]

# Dropping variables that include the target variable in it, namely the total_amount
clean_data = raw_data.drop(['total_amount'], axis=1)

# Releasing memory occupied by raw_data as it is not needed anymore
# Dealing with a large dataset, not running out of memory has to be assured
del raw_data
gc.collect()

# print the number of trips left in the dataset
print("There are " + str(len(clean_data)) + " observations in the dataset.")
print("There are " + str(len(clean_data.columns)) + " variables in the dataset.")

plt.hist(clean_data.tip_amount.values, 16, histtype='bar', facecolor='g')
plt.show()

print("Minimum amount value is ", np.min(clean_data.tip_amount.values))
print("Maximum amount value is ", np.max(clean_data.tip_amount.values))
print("90% of the trips have a tip amount less or equal than ", np.percentile(clean_data.tip_amount.values, 90))

# display first rows in the dataset
print(clean_data.head())

# Data Pre-processing
# Convert 'tpep_dropoff_datetime' and 'tpep_pickup_datetime' columns to datetime objects
clean_data['tpep_dropoff_datetime'] = pd.to_datetime(clean_data['tpep_dropoff_datetime'])
clean_data['tpep_pickup_datetime'] = pd.to_datetime(clean_data['tpep_pickup_datetime'])

# Extract pickup and dropoff hour
clean_data['pickup_hour'] = clean_data['tpep_pickup_datetime'].dt.hour
clean_data['dropoff_hour'] = clean_data['tpep_dropoff_datetime'].dt.hour

# Extract pickup and dropoff day of the week (0 = Monday, 6 = Sunday)
clean_data['pickup_day'] = clean_data['tpep_pickup_datetime'].dt.weekday
clean_data['dropoff_day'] = clean_data['tpep_dropoff_datetime'].dt.weekday

# Calculate trip time in seconds
clean_data['trip_time'] = (clean_data['tpep_dropoff_datetime'] - clean_data['tpep_pickup_datetime']).dt.total_seconds()

# Ideally the full dataset has to be used
# However, to avoid running into out-of-memory issues due to the data size,
# size will be reduced.
# For instance, in this example, only the first 200,000 samples are used.
first_n_rows = 200000
clean_data = clean_data.head(first_n_rows)

# drop the pickup and dropoff datetimes
clean_data = clean_data.drop(['tpep_pickup_datetime', 'tpep_dropoff_datetime'], axis=1)

# some features are categorical, so the need to be encoded
# to encode categorical features, one-hot encoding from the Pandas package is used
get_dummy_col = ["VendorID", "RatecodeID", "store_and_fwd_flag", "PULocationID", "DOLocationID", "payment_type",
                 "pickup_hour", "dropoff_hour", "pickup_day", "dropoff_day"]
proc_data = pd.get_dummies(clean_data, columns=get_dummy_col)

# Releasing memory occupied by raw_data as it is not needed anymore
# Dealing with a large dataset, not running out of memory has to be assured
del clean_data
gc.collect()

# extract the labels from the dataframe
y = proc_data[['tip_amount']].values.astype('float32')

# drop the target variable from the feature matrix
proc_data = proc_data.drop(['tip_amount'], axis=1)

# get the feature matrix used for training
X = proc_data.values

# normalize the feature matrix
X = normalize(X, axis=1, norm='l1', copy=False)

# print the shape of the features matrix and the labels vector
print('X.shape=', X.shape, 'y.shape=', y.shape)

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print('X_train.shape=', X_train.shape, 'Y_train.shape=', y_train.shape)
print('X_test.shape=', X_test.shape, 'Y_test.shape=', y_test.shape)

# for reproducible output across multiple function calls, set random_state to a given integer value
sklearn_dt = decisionTR(max_depth=8, random_state=35)

# training a Decision Tree Regressor using scikit-learn
t0 = time.time()
sklearn_dt.fit(X_train, y_train)
sklearn_time = time.time() - t0
print("[Scikit-Learn] Training time (s):  {0:.5f}".format(sklearn_time))

# in contrast to sklearn's Decision Tree, Snap ML offers multi-threaded CPU/GPU training
# to use the GPU, one needs to set the use_gpu parameter to True
# snapml_dt = DecisionTreeRegressor(max_depth=4, random_state=45, use_gpu=True)

# to set the number of CPU threads used at training time, one needs to set the n_jobs parameter
# for reproducible output across multiple function calls, set random_state to a given integer value
snapml_dt = DecisionTreeRegressor(max_depth=8, random_state=45, n_jobs=4)

# train a Decision Tree Regressor model using Snap ML
t0 = time.time()
snapml_dt.fit(X_train, y_train)
snapml_time = time.time() - t0
print("[Snap ML] Training time (s):  {0:.5f}".format(snapml_time))

# Model Evaluation and Comparison
# Snap ML vs Scikit-Learn training speedup
training_speedup = sklearn_time / snapml_time
print('[Decision Tree Regressor] Snap ML vs. Scikit-Learn speedup : {0:.2f}x '.format(training_speedup))

# run inference using the sklearn model
sklearn_pred = sklearn_dt.predict(X_test)

# evaluate mean squared error on the test dataset
sklearn_mse = mean_squared_error(y_test, sklearn_pred)
print('[Scikit-Learn] MSE score : {0:.3f}'.format(sklearn_mse))

# run inference using the Snap ML model
snapml_pred = snapml_dt.predict(X_test)

# evaluate mean squared error on the test dataset
snapml_mse = mean_squared_error(y_test, snapml_pred)
print('[Snap ML] MSE score : {0:.3f}'.format(snapml_mse))

# Training another Decision Tree Regressor with the 'max_depth' parameter set to '12',
# 'random_state' set to '45', and 'n_jobs' set to '4' and comparing its Mean Squared Error
# to the decision tree regressor trained previously
tree = DecisionTreeRegressor(max_depth=12, random_state=45, n_jobs=4)
tree.fit(X_train, y_train)
pred = tree.predict(X_test)
print("MSE: ", mean_squared_error(y_test, pred))
