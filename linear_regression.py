import numpy as np

class LinearRegression:
    
    def __init__(self):
        self.w = None
        self.b = None

    # Stochastic gradient descent
    def fit(self, X, Y, lr, epochs):
        self.w = np.zeros((X.shape[1],))
        self.b = 0
        losses = []
        for _ in range(epochs):
            # Shuffle between epochs
            rand_perm = np.random.permutation(X.shape[0])
            # Update weights a little for every training example
            for x, y in zip(X[rand_perm, :], Y[rand_perm]):
                # Predict
                pred = self.predict(x)
                # Calculate loss
                error = pred - y
                loss = error**2  # not actually used by the algorithm but can be printed to see if the model is improving
                losses.append(loss)
                # Calculate loss gradient
                w_grad = 2 * error * x  # x is the derivative of the inner function (chain rule)
                b_grad = 2 * error
                # Update weights
                self.w -= lr * w_grad
                self.b -= lr * b_grad
        return losses

    def predict(self, X):
        return np.dot(X, self.w) + self.b


