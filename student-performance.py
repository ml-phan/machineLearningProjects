import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from matplotlib import style
import pickle

# Load data from file
data = pd.read_csv("data/student-mat.csv", sep=";")  # Load data from file
data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]  # Choosing what features to train your model

predict = "G3"  # Name of the value that will be our prediction

X = np.array(data.drop([predict], 1))
y = np.array(data[predict])

print(data.columns)

# Split data to train and test set. test set is 10% of the data.
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)

# The next part is commented out because the model is already train and saved in the file "pickle/studentmodel.pickle"

# Train the model until n accuracy of 95% is reached and save the best model to the file "pickle/studentmodel.pickle"
# best = 0
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

# Load model from the pickle file
pickle_in = open("pickle/studentmodel.pickle", "rb")
linear = pickle.load(pickle_in)

# Print out the model parameters and gauge the model accuracy on a random split dataset
print("Coefficient: ", linear.coef_)
print("Intercept: ", linear.intercept_)
print("Accuracy: ", linear.score(x_train, y_train))

# Test the model on the test set
predictions = linear.predict(x_test)

# for x in range(len(predictions)):
#     print(predictions[x], x_test[x], y_test[x])

# Plot two of the data features, in this case an "absences/g3" plot
style.use("ggplot")
p = "absences"
plt.scatter(data[p], data["G3"])
plt.xlabel(p)
plt.ylabel("G3")
plt.show()
