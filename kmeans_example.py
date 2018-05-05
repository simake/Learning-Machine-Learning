from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

from kmeans import KMeans

iris = load_iris()
X = iris['data'][:, :2]
y = iris['target']

kmeans = KMeans(3)
kmeans.fit(X)
clusters = kmeans.predict(X)

colors = ['orange', 'salmon', 'purple']
f, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(5, 8))
for i in range(X.shape[0]):
    ax1.plot(X[i, 0], X[i, 1], marker='o', color=colors[y[i]])
    ax2.plot(X[i, 0], X[i, 1], marker='o', color='grey')
    ax3.plot(X[i, 0], X[i, 1], marker='o', color=colors[clusters[i]])
ax1.set_title('Original data (labeled)')
ax1.set_axis_off()
ax2.set_title('What k-means sees (unlabeled)')
ax2.set_axis_off()
ax3.set_title('k-means clustering')
ax3.set_axis_off()
plt.show()
