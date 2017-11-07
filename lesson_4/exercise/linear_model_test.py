from lesson_4.exercise.linear_model import LinearRegression
from pprint import pprint
import numpy as np

import matplotlib.pyplot as plt

data = np.genfromtxt('e:\\study\\ml\\ml\\lesson_4\\exercise\\train.csv', delimiter=',', skip_header=True)
data[:5]

X = data[:, 0:11]
y = data[:, 11:]
pprint(X[0])
pprint(y[0])

clf = LinearRegression()

clf.fit(X, y)

print(clf.cost(X, y))

plt.figure(figsize=(10, 8))
plt.scatter(X[:, 0], y, c='b')
# plt.scatter(X[:, 0], clf.predict(X[3]), c='r')
plt.show()
