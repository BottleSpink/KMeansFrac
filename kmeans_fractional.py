"""
K-means clustering with fractional distance metric on MNIST dataset

Fractional distance is a Minkowski distance with p < 1:
d(x, y) = (Σ|x_i - y_i|^p)^(1/p)

where 0 < p < 1 for fractional distances
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.metrics import confusion_matrix
from scipy.optimize import linear_sum_assignment
import time


def fractional_distance(X, centroids, p=0.5):
    """
    Calculate fractional distance between data points and centroids

    Args:
        X: (n_samples, n_features) data matrix
        centroids: (n_clusters, n_features) centroid matrix
        p: fractional power (0 < p < 1)

    Returns:
        distances: (n_samples, n_clusters) distance matrix
    """
    n_samples = X.shape[0]
    n_clusters = centroids.shape[0]
    distances = np.zeros((n_samples, n_clusters))

    for i in range(n_clusters):
        # Calculate |x - centroid|^p element-wise, then sum and take (1/p) power
        diff = np.abs(X - centroids[i])
        distances[:, i] = np.sum(diff ** p, axis=1) ** (1.0 / p)

    return distances


class FractionalKMeans:
    """
    K-means clustering using fractional distance metric
    """

    def __init__(self, n_clusters=10, p=0.5, max_iter=300, n_init=10, random_state=None, verbose=0):
        """
        Args:
            n_clusters: number of clusters
            p: fractional power for distance metric (0 < p < 1)
            max_iter: maximum number of iterations
            n_init: number of times to run with different initializations
            random_state: random seed
            verbose: verbosity level
        """
        self.n_clusters = n_clusters
        self.p = p
        self.max_iter = max_iter
        self.n_init = n_init
        self.random_state = random_state
        self.verbose = verbose
        self.cluster_centers_ = None
        self.labels_ = None
        self.inertia_ = None
        self.n_iter_ = 0

    def _initialize_centroids(self, X):
        """Initialize centroids using k-means++ algorithm"""
        n_samples, n_features = X.shape
        rng = np.random.RandomState(self.random_state)

        # Choose first centroid randomly
        centroids = np.zeros((self.n_clusters, n_features))
        centroids[0] = X[rng.randint(n_samples)]

        # Choose remaining centroids with probability proportional to distance
        for i in range(1, self.n_clusters):
            distances = fractional_distance(X, centroids[:i], self.p)
            min_distances = np.min(distances, axis=1)
            probabilities = min_distances / np.sum(min_distances)
            cumulative_probs = np.cumsum(probabilities)
            r = rng.rand()
            idx = np.searchsorted(cumulative_probs, r)
            centroids[i] = X[idx]

        return centroids

    def _assign_clusters(self, X, centroids):
        """Assign each point to nearest centroid using fractional distance"""
        distances = fractional_distance(X, centroids, self.p)
        labels = np.argmin(distances, axis=1)
        inertia = np.sum(np.min(distances, axis=1) ** self.p)
        return labels, inertia

    def _update_centroids(self, X, labels):
        """Update centroids as mean of assigned points"""
        centroids = np.zeros((self.n_clusters, X.shape[1]))

        for i in range(self.n_clusters):
            mask = labels == i
            if np.sum(mask) > 0:
                centroids[i] = np.mean(X[mask], axis=0)
            else:
                # If cluster is empty, reinitialize randomly
                centroids[i] = X[np.random.randint(X.shape[0])]

        return centroids

    def fit(self, X):
        """Fit K-means with fractional distance"""
        best_inertia = np.inf
        best_centroids = None
        best_labels = None
        best_n_iter = 0

        for init_idx in range(self.n_init):
            if self.verbose:
                print(f"Initialization {init_idx + 1}/{self.n_init}")

            # Initialize centroids
            if self.random_state is not None:
                seed = self.random_state + init_idx
            else:
                seed = None

            temp_kmeans = FractionalKMeans(
                n_clusters=self.n_clusters,
                p=self.p,
                max_iter=self.max_iter,
                n_init=1,
                random_state=seed,
                verbose=0
            )
            centroids = temp_kmeans._initialize_centroids(X)

            # Iterate until convergence
            for iteration in range(self.max_iter):
                # Assignment step
                labels, inertia = self._assign_clusters(X, centroids)

                # Update step
                new_centroids = self._update_centroids(X, labels)

                # Check convergence
                if np.allclose(centroids, new_centroids):
                    if self.verbose:
                        print(f"  Converged at iteration {iteration + 1}")
                    break

                centroids = new_centroids

            # Keep best result
            if inertia < best_inertia:
                best_inertia = inertia
                best_centroids = centroids
                best_labels = labels
                best_n_iter = iteration + 1

        self.cluster_centers_ = best_centroids
        self.labels_ = best_labels
        self.inertia_ = best_inertia
        self.n_iter_ = best_n_iter

        return self

    def predict(self, X):
        """Predict cluster labels for new data"""
        labels, _ = self._assign_clusters(X, self.cluster_centers_)
        return labels

    def fit_predict(self, X):
        """Fit and return cluster labels"""
        self.fit(X)
        return self.labels_


def load_mnist(n_samples=None):
    """Load MNIST dataset"""
    print("Loading MNIST dataset...")
    mnist = fetch_openml('mnist_784', version=1, parser='auto')
    X = np.array(mnist.data, dtype='float32')
    y = np.array(mnist.target, dtype='int')

    if n_samples:
        np.random.seed(42)
        indices = np.random.choice(len(X), n_samples, replace=False)
        X = X[indices]
        y = y[indices]

    return X, y


def cluster_accuracy(y_true, y_pred):
    """
    Calculate clustering accuracy by finding the best label mapping
    """
    # Create confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Use Hungarian algorithm to find optimal label assignment
    row_ind, col_ind = linear_sum_assignment(-cm)

    # Calculate accuracy with optimal mapping
    accuracy = cm[row_ind, col_ind].sum() / cm.sum()

    return accuracy, dict(zip(col_ind, row_ind))


def plot_cluster_centers(cluster_centers, n_clusters, p_value, filename='cluster_centers_fractional.png'):
    """Visualize cluster centers as images"""
    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    axes = axes.ravel()

    for i in range(min(n_clusters, 10)):
        axes[i].imshow(cluster_centers[i].reshape(28, 28), cmap='gray')
        axes[i].set_title(f'Cluster {i}')
        axes[i].axis('off')

    plt.suptitle(f'Cluster Centers (Fractional Distance p={p_value})', fontsize=14)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"Cluster centers saved to {filename}")
    plt.close()


def main():
    # Parameters
    n_samples = 10000
    n_clusters = 10
    p_value = 0.5  # Fractional distance parameter (0 < p < 1)

    # Load MNIST dataset
    X, y = load_mnist(n_samples=n_samples)

    print(f"Dataset shape: {X.shape}")
    print(f"Labels shape: {y.shape}")

    # Normalize the data
    print("\nNormalizing data...")
    X_normalized = X / 255.0

    # Apply K-means with fractional distance
    print(f"\nApplying K-means with fractional distance (p={p_value})...")
    print(f"Number of clusters: {n_clusters}")

    start_time = time.time()
    kmeans = FractionalKMeans(
        n_clusters=n_clusters,
        p=p_value,
        random_state=42,
        n_init=5,  # Reduced for faster computation
        max_iter=100,
        verbose=1
    )

    y_pred = kmeans.fit_predict(X_normalized)
    elapsed_time = time.time() - start_time

    print(f"\nClustering completed in {elapsed_time:.2f} seconds")
    print(f"Inertia: {kmeans.inertia_:.2f}")
    print(f"Iterations: {kmeans.n_iter_}")

    # Calculate clustering accuracy
    print("\nCalculating clustering accuracy...")
    accuracy, label_mapping = cluster_accuracy(y, y_pred)
    print(f"Clustering Accuracy: {accuracy:.4f}")
    print(f"Label mapping (cluster -> digit): {label_mapping}")

    # Visualize cluster centers
    print("\nVisualizing cluster centers...")
    plot_cluster_centers(kmeans.cluster_centers_, n_clusters, p_value)

    # Print cluster distribution
    print("\nCluster distribution:")
    unique, counts = np.unique(y_pred, return_counts=True)
    for cluster, count in zip(unique, counts):
        digit = label_mapping.get(cluster, '?')
        print(f"  Cluster {cluster} (digit {digit}): {count} samples")

    # Print confusion matrix
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y, y_pred)
    print(cm)

    return kmeans, y_pred, accuracy


if __name__ == "__main__":
    kmeans, y_pred, accuracy = main()
    print(f"\nFinal Clustering Accuracy: {accuracy:.4f}")
