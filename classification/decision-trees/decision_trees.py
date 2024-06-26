import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import sklearn.tree as tree
from sklearn import metrics
import matplotlib.pyplot as plt

# Importing the data
my_data = pd.read_csv("drug200.csv", delimiter=",")

# Inspecting the data
print(my_data[0:5])  # equivalent to print(my_data.head())
print(my_data.shape)

# Preprocessing the data
X = my_data[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']].values
print(X[0:5])

# Sklearn Decision Trees do not handle categorical variables, so categorical features
# need to be converted to numerical values using LabelEncoder
# Converting Gender to numerical values
le_sex = preprocessing.LabelEncoder()
le_sex.fit(['F', 'M'])
X[:, 1] = le_sex.transform(X[:, 1])

# Converting Blood Pressure to numerical values
le_BP = preprocessing.LabelEncoder()
le_BP.fit(['LOW', 'NORMAL', 'HIGH'])
X[:, 2] = le_BP.transform(X[:, 2])

# Converting Cholestrol level to numerical values
le_Chol = preprocessing.LabelEncoder()
le_Chol.fit(['NORMAL', 'HIGH'])
X[:, 3] = le_Chol.transform(X[:, 3])

print(X[0:5])

# Create target variable vector
y = my_data["Drug"]
print(y[0:5])

# Splitting data into training and testing sets
X_trainset, X_testset, y_trainset, y_testset = train_test_split(X, y, test_size=0.3, random_state=3)

print('Shape of X training set {}'.format(X_trainset.shape), '&', ' Size of Y training set {}'.format(y_trainset.shape))
print('Shape of X testing set {}'.format(X_testset.shape), '&', ' Size of Y testing set {}'.format(y_testset.shape))

# Modeling
drugTree = DecisionTreeClassifier(criterion="entropy", max_depth=4)
print(drugTree)  # show default model parameters

# Training
drugTree.fit(X_trainset, y_trainset)

# Prediction
predTree = drugTree.predict(X_testset)

print(predTree[0:5])
print(y_testset[0:5])

# Evaluation
print("DecisionTrees's Accuracy: ", metrics.accuracy_score(y_testset, predTree))

tree.plot_tree(drugTree)
plt.show()
