import sklearn
from sklearn import linear_model, preprocessing
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np

# Read data from file in data/car.data
data = pd.read_csv("data/car.data")

# Print the first 5 rows of the data
print(data.head())

# Encode target labels with n classes with value between 0 and n-1
le = preprocessing.LabelEncoder()

# Fit_transform will encode and return the encoded labels
buying = le.fit_transform(list(data["buying"]))
maint = le.fit_transform(list(data["maint"]))
door = le.fit_transform(list(data["door"]))
persons = le.fit_transform(list(data["persons"]))
lug_boot = le.fit_transform(list(data["lug_boot"]))
safety = le.fit_transform(list(data["safety"]))
cls = le.fit_transform(list(data["class"]))

#
predict = "class"

# Make a lit from zip iterator of all the lists above
X = list(zip(buying, maint, door, persons, lug_boot, safety))
y = list(cls)

# Split dataset into training and test set. Test set is 10%
# Input for train_test_split can be lists, numpy arrays, scipy matrices or pandas dataframes
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)

# Initialize K-NN model with K =7
model = KNeighborsClassifier(7)

# Train the model with training data x_train and y_train
model.fit(x_train, y_train)

# Return accuracy of the model with variable "acc"
acc = model.score(x_test, y_test)

print(acc)

# Test the model using the test set: x_test
predicted = model.predict(x_test)
names = ["unacc", "acc", "good", "vgood"]

# Output test results and correct labels together
for x in range(len(predicted)):
    print("Predicted:", names[predicted[x]], "Data:", x_test[x], "Actual:", names[y_test[x]])
    n = model.kneighbors([x_test[x]], 9, True)
    print("N: ", n)
