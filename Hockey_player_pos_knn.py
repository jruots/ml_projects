# The point of this classification algorithm is to see whether a variety of
# classification models can correctly classify hockey players according
# to position in relation to the amount of points scored.
# The data comes from hockey reference and consists of players who have
# played a minimum of 50 games during the 2015-2016 season.

import sys
import scipy
import numpy
import matplotlib
import pandas
from pandas.tools.plotting import scatter_matrix
import sklearn

# import data from csv file and name the features. 

data = "C:/Users/Joni/Documents/Machine Learning data/Hockey data/player_points_and_games_per_position_1.csv"
names = ["GP", "PTS", "Pos"]
dataset = pandas.read_csv(data, sep=";", names=names)

# let see the shape of the data i.e. rows and columns. 
# lets have a look at the beginning of the data for sanity check
# lets look at some descriptive statistics for some color

print(dataset.shape)
print(dataset.head(20))
print(dataset.describe())

# lets visualize with box and whisker plot

dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
matplotlib.pyplot.show()

# lets visualize with scatter matrix

scatter_matrix(dataset)
matplotlib.pyplot.show()

# split the data into training and test data

from sklearn.model_selection import train_test_split
array = dataset.values
X = array[:,0:1]
Y = array[:,2]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=0)

# import k-nearest neighbors

from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(n_neighbors=2)

# fit the data

clf.fit(X_train, Y_train)

# print the predictions and accuracy score

print("Test set predicitions: {}".format(clf.predict(X_test)))

print("Test set accuracy: {:.2f}".format(clf.score(X_test, Y_test)))

