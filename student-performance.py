import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from matplotlib import style
import pickle

data = pd.read_csv("data/student-mat.csv", sep=";")  # Load data from file
data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]  # Choosing what features to train your model

# print(data.head)  # Display the first 5 rows of the data

predict = "G3"  # Name of the value that will be our prediction

X = np.array(data.drop([predict], 1))
y = np.array(data[predict])

print(data.columns)
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)
# Split data to train and test set. test set is 10% of the data.
best = 0
# while best < 0.95:
#     x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)
#
#     linear = linear_model.LinearRegression()
#
#     linear.fit(x_train, y_train)
#     accuracy = linear.score(x_test, y_test)
#     if accuracy > best:
#         best = accuracy
#         with open("studentmodel.pickle", "wb") as f:
#             pickle.dump(linear, f)
#     print(best)
# print(accuracy)
print(type(X))
print(X)
print(type(x_train))
print(x_train)

pickle_in = open("pickle/studentmodel.pickle", "rb")
linear = pickle.load(pickle_in)

print("Coefficient: ", linear.coef_)
print("Intercept: ", linear.intercept_)
print("Accuracy: ", linear.score(x_train, y_train))

predictions = linear.predict(x_test)

# for x in range(len(predictions)):
#     print(predictions[x], x_test[x], y_test[x])

style.use("ggplot")
p = "absences"
plt.scatter(data[p], data["G3"])
plt.xlabel(p)
plt.ylabel("G3")
plt.show()
