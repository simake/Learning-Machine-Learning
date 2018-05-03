import numpy as np

class KMeans:
    def __init__(self, n_clusters, max_iter=300):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.centroids = None

    def fit(self, X):
        n_points = X.shape[0]
        # Initialize centroids on top of different random examples
        centroids = X[np.random.choice(n_points, size=self.n_clusters, replace=False)]
        # Iterate until convergence (capped by max_iter)
        for _ in range(self.max_iter):
            # Calculate the distances from each centroid to every data point
            distances = [np.linalg.norm(centroid - X, axis=1) for centroid in centroids]
            # Assign data points to the closest centroid
            assignments = np.argmin(distances, axis=0)
            # Save a copy of the old centroids for detecting convergence
            prev_centroids = centroids.copy()
            # Move centroids to the mean of their assigned data points
            centroids = np.array([np.mean(X[assignments == i], axis=0) for i in range(self.n_clusters)])
            # Break on convergence
            if np.array_equal(centroids, prev_centroids):
                break
        self.centroids = centroids

    def predict(self, X):
        # Return the index of the predicted cluster for every data point in X
        return np.argmin([np.linalg.norm(centroid - X, axis=1) for centroid in self.centroids], axis=0)