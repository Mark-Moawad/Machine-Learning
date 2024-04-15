"""
This code converts a linear classifier into a multi-class classifier, including multinomial logistic regression or
softmax regression, One vs. All (One-vs-Rest) and One vs. One.

In Multi-class classification, data is classified into multiple class labels.
Unlike classification trees and k-nearest neighbor, the concept of multi-class classification for linear classifiers
is not as straightforward. Logistic regression can be converted to multi-class classification using multinomial
logistic regression or softmax regression; this is a generalization of logistic regression, that will not work
for support vector machines. One vs. All (One-vs-Rest) and One vs. One are two other multi-class classification
techniques that can convert any two-class classifier into a multi-class classifier.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pandas as pd

# Utility function that plots a different decision boundary

plot_colors = "ryb"
plot_step = 0.02


def decision_boundary(X, y, model, iris, two=None):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                         np.arange(y_min, y_max, plot_step))
    plt.tight_layout(h_pad=0.5, w_pad=0.5, pad=2.5)

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    cs = plt.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu)

    if two:
        cs = plt.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu)
        for i, color in zip(np.unique(y), plot_colors):
            idx = np.where(y == i)
            plt.scatter(X[idx, 0], X[idx, 1], label=y, cmap=plt.cm.RdYlBu, s=15)
        plt.show()

    else:
        set_ = {0, 1, 2}
        print(set_)
        for i, color in zip(range(3), plot_colors):
            idx = np.where(y == i)
            if np.any(idx):
                set_.remove(i)

                plt.scatter(X[idx, 0], X[idx, 1], label=y, cmap=plt.cm.RdYlBu, edgecolor='black', s=15)

        for i in set_:
            idx = np.where(iris.target == i)
            plt.scatter(X[idx, 0], X[idx, 1], marker='x', color='black')

        plt.show()


def plot_probability_array(X, probability_array):
    plot_array = np.zeros((X.shape[0], 30))
    col_start = 0
    ones = np.ones((X.shape[0], 30))
    for class_, col_end in enumerate([10, 20, 30]):
        plot_array[:, col_start:col_end] = np.repeat(probability_array[:, class_].reshape(-1, 1), 10, axis=1)
        col_start = col_end
    plt.imshow(plot_array)
    plt.xticks([])
    plt.ylabel("samples")
    plt.xlabel("probability of 3 classes")
    plt.colorbar()
    plt.show()


"""
The  iris dataset will be used in this model, it consists of three different types of irises’
(Setosa y=0, Versicolour y=1, and Virginica y=2), petal and sepal length, stored in a 150x4 numpy.ndarray.

The rows being the samples and the columns: Sepal Length, Sepal Width, Petal Length and Petal Width.

The following plot uses the two features Sepal Width and Petal Width"""

pair = [1, 3]
iris = datasets.load_iris()
X = iris.data[:, pair]
y = iris.target
print(np.unique(y))

plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu)
plt.xlabel("sepal width (cm)")
plt.ylabel("petal width")
plt.show()

# Training a Logistic Regression Model using scikit-learn
lr = LogisticRegression(random_state=0).fit(X, y)

probability = lr.predict_proba(X)
plot_probability_array(X, probability)

print(probability[0, :])

print(probability[0, :].sum())

# Applying argmax function
print(np.argmax(probability[0, :]))

# Apply the argmax function to each sample
softmax_prediction = np.argmax(probability, axis=1)
print(softmax_prediction)

# verify that sklearn does this under the hood by comparing it to the output of the method predict
yhat = lr.predict(X)
print(accuracy_score(yhat, softmax_prediction))

# Training an SVM to perform multi-class classification
model = SVC(kernel='linear', gamma=.5, probability=True)
model.fit(X, y)

# Evaluation
yhat = model.predict(X)
print(accuracy_score(y, yhat))

# Plot decision boundary
decision_boundary(X, y, model, iris)

# One vs. All (One-vs-Rest)
"""
For one-vs-all classification, if there are K classes, K two-class classifier models are used. The number of class
labels present in the dataset is equal to the number of generated classifiers. First, an artificial class is created,
it can be called "dummy" class. For each classifier, the data gets split into two classes. The class samples to be
classified are taken into consideration, the rest of the samples will be labelled as a dummy class. The process
is repeated for each class. To make a  classification, the classifier with the highest probability is used,
disregarding the dummy class.
"""
# Train Each Classifier
"""
Three classifiers are trained and placed in the list>my_models. For each class only the class samples to be classified
are taken into account, and the rest will be labelled as a dummy class. The process is repeated for each class.
For each classifier, the decision regions are plotted. The class in interest is in red, and the dummy class is in blue.
Similarly, the class samples are marked in blue, and the dummy samples are marked with a black x.
"""
# dummy class
dummy_class = y.max() + 1
# list used for classifiers
my_models = []
# iterate through each class
for class_ in np.unique(y):
    # select the index of our  class
    select = (y == class_)
    temp_y = np.zeros(y.shape)
    # class, we are trying to classify
    temp_y[y == class_] = class_
    # set other samples  to a dummy class
    temp_y[y != class_] = dummy_class
    # Train model and add to list
    model = SVC(kernel='linear', gamma=.5, probability=True)
    my_models.append(model.fit(X, temp_y))
    # plot decision boundary
    decision_boundary(X, temp_y, model, iris)

# For each sample, calculate the  probability of belonging to each class, not including the dummy class.
probability_array = np.zeros((X.shape[0], 3))
for j, model in enumerate(my_models):
    real_class = np.where(np.array(model.classes_) != 3)[0]

    probability_array[:, j] = model.predict_proba(X)[:, real_class][:, 0]

# probability of belonging to each class for the first sample
print(probability_array[0, :])

# As each is the probability of belonging to the actual class and not the dummy class, it does not sum to one
print(probability_array[0, :].sum())

# Plot the probability of belonging to the class. The row number is the sample number.
plot_probability_array(X, probability_array)

# Apply the argmax function to each sample to find the class
one_vs_all = np.argmax(probability_array, axis=1)
print(one_vs_all)

# Calculate the accuracy
print(accuracy_score(y, one_vs_all))

# The accuracy is less than the one obtained by sklearn, and this is because for SVM, sklearn uses one vs one.
# Comparing the outputs
print(accuracy_score(one_vs_all, yhat))

# One Vs One
"""
In One-vs-One classification, the data is split up into each class, and then a two-class classifier is trained
on each pair of classes. For example, if there are classes 0,1,2, one classifier would be trained on the samples that
are class 0 and class 1, a second classifier on samples that are of class 0 and class 2, and a final classifier on
samples of class 1 and class 2.

For K classes, K(K-1)/2 have to be trained classifiers.

To perform classification on a sample, a majority vote is performed and the class with the most predictions is selected.
"""
classes_ = set(np.unique(y))
print(classes_)

# Determining the number of classifiers
K = len(classes_)
print(K * (K - 1) / 2)

# Train a two-class classifier on each pair of classes
pairs = []
left_overs = classes_.copy()
# List used for classifiers
my_models = []
# Iterate through each class
for class_ in classes_:
    # Remove class we have seen before
    left_overs.remove(class_)
    # The second class in the pair
    for second_class in left_overs:
        pairs.append(str(class_) + ' and ' + str(second_class))
        print("class {} vs class {} ".format(class_, second_class))
        temp_y = np.zeros(y.shape)
        # Find classes in pair
        select = np.logical_or(y == class_, y == second_class)
        # Train model
        model = SVC(kernel='linear', gamma=.5, probability=True)
        model.fit(X[select, :], y[select])
        my_models.append(model)
        # Plot decision boundary for each pair and corresponding Training samples
        decision_boundary(X[select, :], y[select], model, iris, two=True)

print(pairs)

majority_vote_array = np.zeros((X.shape[0], 3))
majority_vote_dict = {}
for j, (model, pair) in enumerate(zip(my_models, pairs)):

    majority_vote_dict[pair] = model.predict(X)
    majority_vote_array[:, j] = model.predict(X)

# In the following table, each column is the output of a classifier for each pair of classes and the output
# is the prediction
print(pd.DataFrame(majority_vote_dict).head(10))

# To perform classification on a sample, a majority vote is performed, that is, the class with the most predictions
# is selected. Repeat the process for each sample
one_vs_one = np.array([np.bincount(sample.astype(int)).argmax() for sample in majority_vote_array])
print(one_vs_one)

# Calculate accuracy
print(accuracy_score(y, one_vs_one))

# Compare it to sklearn SVM accuracy
print(accuracy_score(yhat, one_vs_one))
