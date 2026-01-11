"""
K-means clustering on MNIST dataset
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import time


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
    from scipy.optimize import linear_sum_assignment

    # Create confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Use Hungarian algorithm to find optimal label assignment
    row_ind, col_ind = linear_sum_assignment(-cm)

    # Calculate accuracy with optimal mapping
    accuracy = cm[row_ind, col_ind].sum() / cm.sum()

    return accuracy, dict(zip(col_ind, row_ind))


def plot_cluster_centers(kmeans, n_clusters=10):
    """Visualize cluster centers as images"""
    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    axes = axes.ravel()

    for i in range(n_clusters):
        axes[i].imshow(kmeans.cluster_centers_[i].reshape(28, 28), cmap='gray')
        axes[i].set_title(f'Cluster {i}')
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig('cluster_centers.png', dpi=150, bbox_inches='tight')
    print("Cluster centers saved to cluster_centers.png")
    plt.close()


def main():
    # Load MNIST dataset (using subset for faster computation)
    X, y = load_mnist(n_samples=10000)

    print(f"Dataset shape: {X.shape}")
    print(f"Labels shape: {y.shape}")

    # Normalize the data
    print("\nNormalizing data...")
    X_normalized = X / 255.0

    # Apply K-means clustering
    n_clusters = 10
    print(f"\nApplying K-means with {n_clusters} clusters...")

    start_time = time.time()
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=10,
        max_iter=300,
        verbose=1
    )

    y_pred = kmeans.fit_predict(X_normalized)
    elapsed_time = time.time() - start_time

    print(f"\nClustering completed in {elapsed_time:.2f} seconds")
    print(f"Inertia: {kmeans.inertia_:.2f}")

    # Calculate clustering accuracy
    print("\nCalculating clustering accuracy...")
    accuracy, label_mapping = cluster_accuracy(y, y_pred)
    print(f"Clustering Accuracy: {accuracy:.4f}")
    print(f"Label mapping (cluster -> digit): {label_mapping}")

    # Visualize cluster centers
    print("\nVisualizing cluster centers...")
    plot_cluster_centers(kmeans, n_clusters)

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
