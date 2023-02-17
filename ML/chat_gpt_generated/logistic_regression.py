import numpy as np

class LogisticRegression:
    def __init__(self, learning_rate=0.01, num_iterations=100000, verbose=False):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.verbose = verbose

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        # Initialize parameters with zeros
        self.theta = np.zeros((X.shape[1], 1))
        m = X.shape[0]

        # Gradient descent
        for i in range(self.num_iterations):
            z = np.dot(X, self.theta)
            h = self.sigmoid(z)
            gradient = np.dot(X.T, (h - y)) / m
            self.theta -= self.learning_rate * gradient

            if self.verbose and i % 10000 == 0:
                cost = self.compute_cost(X, y)
                print(f"Iteration {i}, cost: {cost}")

    def predict(self, X):
        z = np.dot(X, self.theta)
        h = self.sigmoid(z)
        return (h > 0.5).astype(int)

    def compute_cost(self, X, y):
        m = X.shape[0]
        z = np.dot(X, self.theta)
        h = self.sigmoid(z)
        J = (-1 / m) * np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))
        return J

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

iris = load_iris()
X = iris.data[:, :2]
y = (iris.target != 0) * 1

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = LogisticRegression(lr=0.1, num_iter=300000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print('Accuracy:', (y_pred == y_test).mean())
