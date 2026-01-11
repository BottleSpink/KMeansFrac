"""
Comparison of K-means with Euclidean distance vs Fractional distance on MNIST
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
from scipy.optimize import linear_sum_assignment
import time
from kmeans_fractional import FractionalKMeans


def load_mnist(n_samples=None):
    """Load MNIST dataset"""
    print("Loading MNIST dataset...")
    mnist = fetch_openml('mnist_784', version=1, parser='auto')
    X = mnist.data.astype('float32')
    y = mnist.target.astype('int')

    if n_samples:
        np.random.seed(42)
        indices = np.random.choice(len(X), n_samples, replace=False)
        X = X[indices]
        y = y[indices]

    return X, y


def cluster_accuracy(y_true, y_pred):
    """Calculate clustering accuracy with optimal label mapping"""
    cm = confusion_matrix(y_true, y_pred)
    row_ind, col_ind = linear_sum_assignment(-cm)
    accuracy = cm[row_ind, col_ind].sum() / cm.sum()
    return accuracy, dict(zip(col_ind, row_ind))


def compare_kmeans(X, y, p_values=[0.3, 0.5, 0.7]):
    """
    Compare K-means with Euclidean distance vs multiple fractional distances

    Args:
        X: data matrix
        y: true labels
        p_values: list of p values for fractional distance
    """
    n_clusters = 10
    results = []

    # Normalize data
    X_normalized = X / 255.0

    print("=" * 70)
    print("EUCLIDEAN DISTANCE (p=2, scikit-learn implementation)")
    print("=" * 70)

    # Run standard K-means with Euclidean distance
    start_time = time.time()
    kmeans_euclidean = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=5,
        max_iter=100,
        verbose=0
    )
    y_pred_euclidean = kmeans_euclidean.fit_predict(X_normalized)
    time_euclidean = time.time() - start_time

    accuracy_euclidean, mapping_euclidean = cluster_accuracy(y, y_pred_euclidean)

    print(f"Time: {time_euclidean:.2f} seconds")
    print(f"Inertia: {kmeans_euclidean.inertia_:.2f}")
    print(f"Accuracy: {accuracy_euclidean:.4f}")
    print(f"Iterations: {kmeans_euclidean.n_iter_}")

    results.append({
        'method': 'Euclidean (p=2)',
        'p': 2,
        'time': time_euclidean,
        'accuracy': accuracy_euclidean,
        'inertia': kmeans_euclidean.inertia_,
        'n_iter': kmeans_euclidean.n_iter_
    })

    # Run K-means with different fractional distances
    for p in p_values:
        print("\n" + "=" * 70)
        print(f"FRACTIONAL DISTANCE (p={p})")
        print("=" * 70)

        start_time = time.time()
        kmeans_fractional = FractionalKMeans(
            n_clusters=n_clusters,
            p=p,
            random_state=42,
            n_init=5,
            max_iter=100,
            verbose=0
        )
        y_pred_fractional = kmeans_fractional.fit_predict(X_normalized)
        time_fractional = time.time() - start_time

        accuracy_fractional, mapping_fractional = cluster_accuracy(y, y_pred_fractional)

        print(f"Time: {time_fractional:.2f} seconds")
        print(f"Inertia: {kmeans_fractional.inertia_:.2f}")
        print(f"Accuracy: {accuracy_fractional:.4f}")
        print(f"Iterations: {kmeans_fractional.n_iter_}")

        results.append({
            'method': f'Fractional (p={p})',
            'p': p,
            'time': time_fractional,
            'accuracy': accuracy_fractional,
            'inertia': kmeans_fractional.inertia_,
            'n_iter': kmeans_fractional.n_iter_
        })

    return results


def plot_comparison(results):
    """Plot comparison results"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    methods = [r['method'] for r in results]
    p_values = [r['p'] for r in results]
    accuracies = [r['accuracy'] for r in results]
    times = [r['time'] for r in results]
    inertias = [r['inertia'] for r in results]
    iterations = [r['n_iter'] for r in results]

    # Accuracy comparison
    axes[0, 0].bar(methods, accuracies, color=['blue'] + ['orange'] * (len(methods) - 1))
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].set_title('Clustering Accuracy Comparison')
    axes[0, 0].set_ylim([0, 1])
    axes[0, 0].tick_params(axis='x', rotation=45)
    for i, v in enumerate(accuracies):
        axes[0, 0].text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom')

    # Time comparison
    axes[0, 1].bar(methods, times, color=['blue'] + ['orange'] * (len(methods) - 1))
    axes[0, 1].set_ylabel('Time (seconds)')
    axes[0, 1].set_title('Computation Time Comparison')
    axes[0, 1].tick_params(axis='x', rotation=45)
    for i, v in enumerate(times):
        axes[0, 1].text(i, v + max(times) * 0.02, f'{v:.1f}s', ha='center', va='bottom')

    # Inertia comparison
    axes[1, 0].bar(methods, inertias, color=['blue'] + ['orange'] * (len(methods) - 1))
    axes[1, 0].set_ylabel('Inertia')
    axes[1, 0].set_title('Inertia Comparison')
    axes[1, 0].tick_params(axis='x', rotation=45)

    # Iterations comparison
    axes[1, 1].bar(methods, iterations, color=['blue'] + ['orange'] * (len(methods) - 1))
    axes[1, 1].set_ylabel('Iterations')
    axes[1, 1].set_title('Convergence Speed (Iterations)')
    axes[1, 1].tick_params(axis='x', rotation=45)
    for i, v in enumerate(iterations):
        axes[1, 1].text(i, v + 1, f'{v}', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig('distance_comparison.png', dpi=150, bbox_inches='tight')
    print("\nComparison plot saved to distance_comparison.png")
    plt.close()


def print_summary_table(results):
    """Print a summary table of results"""
    print("\n" + "=" * 90)
    print("SUMMARY TABLE")
    print("=" * 90)
    print(f"{'Method':<20} {'p':<8} {'Accuracy':<12} {'Time (s)':<12} {'Inertia':<12} {'Iterations':<12}")
    print("-" * 90)

    for r in results:
        print(f"{r['method']:<20} {r['p']:<8.1f} {r['accuracy']:<12.4f} {r['time']:<12.2f} {r['inertia']:<12.2f} {r['n_iter']:<12}")

    print("=" * 90)


def main():
    # Load MNIST dataset
    n_samples = 5000  # Smaller for faster comparison
    X, y = load_mnist(n_samples=n_samples)

    print(f"Dataset shape: {X.shape}")
    print(f"Labels shape: {y.shape}\n")

    # Compare different distance metrics
    p_values = [0.3, 0.5, 0.7]
    results = compare_kmeans(X, y, p_values=p_values)

    # Print summary
    print_summary_table(results)

    # Plot comparison
    plot_comparison(results)

    # Analysis
    print("\nKEY OBSERVATIONS:")
    print("-" * 90)
    print("1. Fractional distances (p < 1) create non-convex distance metrics")
    print("2. Lower p values emphasize differences more strongly")
    print("3. Fractional distances may converge slower but can find different cluster structures")
    print("4. Euclidean distance (p=2) is fastest due to optimized scikit-learn implementation")
    print("5. Clustering accuracy can vary significantly depending on the distance metric")


if __name__ == "__main__":
    main()
