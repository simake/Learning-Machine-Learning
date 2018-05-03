import numpy as np

class KNN:
    def __init__(self, k=1):
        self.k = k
    
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        num_rows = X.shape[0]
        y = np.empty(num_rows, dtype=self.y_train.dtype)
        # Make a prediction for each row of X
        for i in range(num_rows):
            # Calculate the distances to all points in X_train
            distances = np.linalg.norm(X[i] - self.X_train, axis=1, keepdims=True)
            # Sort distances and get the indices of the k nearest training examples
            nearest_indices = np.argpartition(distances, axis=0, kth=range(self.k))[:self.k]
            # Retrieve corresponding labels
            labels = self.y_train[nearest_indices]
            # Get unique labels and the number of occurances for each label
            values, counts = np.unique(labels, return_counts=True)
            # Predict the label with the most occurances
            y[i] = values[np.argmax(counts)]
        return y
