"""
In this code a Machine Learning model will be created for a telecommunication company,
to predict when its customers will leave for a competitor, so that they can take some action to retain the customers.

About the Dataset:
A telecommunications dataset will be used for predicting customer churn.
This is a historical customer dataset where each row represents one customer.
The data is relatively easy to understand, and insights that can be used immediately may be drawn upon first glance.

Typically it is less expensive to keep customers than acquire new ones, so the focus of this analysis is to
predict the customers who will stay with the company.

This dataset provides information to help predict what behavior will help retain customers.
All relevant customer data can be analyzed to develop focused customer retention programs.

The dataset includes information about:

- Customers who left within the last month – the column is called Churn
- Services that each customer has signed up for – phone, multiple lines, internet, online security, online backup,
device protection, tech support, and streaming TV and movies
- Customer account information – how long they had been a customer, contract, payment method, paperless billing,
monthly charges, and total charges
- Demographic info about customers – gender, age range, and if they have partners and dependents
"""

import itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import jaccard_score
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split

# Dataset Analysis
# Importing Data
churn_df = pd.read_csv("ChurnData.csv")
print(churn_df.head())

# Data pre-processing and selection
churn_df = churn_df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip', 'callcard', 'wireless', 'churn']]
churn_df['churn'] = churn_df['churn'].astype('int')
print(churn_df.head())
print(churn_df.shape)

# Defining independent and output variables for the dataset
X = np.asarray(churn_df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip']])
print(X[0:5])

y = np.asarray(churn_df['churn'])
print(y[0:5])

# Normalizing the feature matrix
X = preprocessing.StandardScaler().fit(X).transform(X)
print(X[0:5])

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)
print('Train set:', X_train.shape, y_train.shape)
print('Test set:', X_test.shape, y_test.shape)

# Training a Logistic Regression Model using scikit-learn
LR = LogisticRegression(C=0.01, solver='liblinear').fit(X_train, y_train)
print(LR)

# Predicting
yhat = LR.predict(X_test)
print(yhat)

# Predicting with probabilities
# predict_proba()  returns estimates for all classes, ordered by the label of classes.
# So, the first column is the probability of class 0, P(Y=0|X), and second column is probability of class 1, P(Y=1|X)
yhat_prob = LR.predict_proba(X_test)
print(yhat_prob)

# Evaluation using Jaccard Score
print(jaccard_score(y_test, yhat, pos_label=0))


# Evaluation using Confusion Matrix
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, yhat, labels=[1, 0])
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['churn=1', 'churn=0'], normalize=False,  title='Confusion matrix')
plt.show()

print(classification_report(y_test, yhat))

# Evaluation using Log Loss
print("Log Loss first model: %.2f" % log_loss(y_test, yhat_prob))

# Building a new Logistic Regression Model using different __solver__ and __regularization__ values
LR2 = LogisticRegression(C=0.01, solver='sag').fit(X_train, y_train)
yhat_prob2 = LR2.predict_proba(X_test)
print("Log Loss second model: %.2f" % log_loss(y_test, yhat_prob2))
