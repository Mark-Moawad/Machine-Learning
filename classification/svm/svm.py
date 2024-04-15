"""
This code uses SVMs (Support Vector Machines) to build and train a model using human cell records, to classify cells
to benign or malignant.

SVM works by mapping data to a high-dimensional feature space so that data points can be categorized,
even when the data are not linearly separable. A separator between the categories is found, then the data
is transformed in such a way that the separator could be drawn as a hyperplane. Following this, characteristics
of new data can be used to predict the group to which a new record should belong.
"""
import itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import jaccard_score
from sklearn.model_selection import train_test_split

# Importing Data
cell_df = pd.read_csv("cell_samples.csv")
print(cell_df.head())

ax = cell_df[cell_df['Class'] == 4][0:50].plot(kind='scatter', x='Clump', y='UnifSize', color='DarkBlue',
                                               label='malignant')
cell_df[cell_df['Class'] == 2][0:50].plot(kind='scatter', x='Clump', y='UnifSize', color='Yellow', label='benign',
                                          ax=ax)
plt.show()

print(cell_df.dtypes)

# Data pre-processing and selection
# BareNuc column includes some values that are not numerical and need to be dropped
cell_df = cell_df[pd.to_numeric(cell_df['BareNuc'], errors='coerce').notnull()]
cell_df['BareNuc'] = cell_df['BareNuc'].astype('int')
print(cell_df.dtypes)

feature_df = cell_df[['Clump', 'UnifSize', 'UnifShape', 'MargAdh', 'SingEpiSize', 'BareNuc', 'BlandChrom',
                      'NormNucl', 'Mit']]
X = np.asarray(feature_df)
print(X[0:5])

# The model is aimed to predict the value of Class (benign (=2) or malignant (=4))
y = np.asarray(cell_df['Class'])
print(y[0:5])

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)
print('Train set:', X_train.shape, y_train.shape)
print('Test set:', X_test.shape, y_test.shape)

# Training a Support Vector Machine Model using scikit-learn
clf = svm.SVC(kernel='rbf')
clf.fit(X_train, y_train)

# Predicting
yhat = clf.predict(X_test)
print(yhat[0:5])


# Evaluation
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
cnf_matrix = confusion_matrix(y_test, yhat, labels=[2, 4])
np.set_printoptions(precision=2)

print(classification_report(y_test, yhat))

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['Benign(2)', 'Malignant(4)'], normalize=False, title='Confusion matrix')

# Evaluation using f1 score
f1_score(y_test, yhat, average='weighted')

# Evaluation using jaccard index
jaccard_score(y_test, yhat, pos_label=2)

# Rebuilding the model with 'linear' kernel and comparing accuracy
clf2 = svm.SVC(kernel='linear')
clf2.fit(X_train, y_train)
yhat2 = clf2.predict(X_test)
print("Avg F1-score: %.4f" % f1_score(y_test, yhat2, average='weighted'))
print("Jaccard score: %.4f" % jaccard_score(y_test, yhat2, pos_label=2))

plt.show()
