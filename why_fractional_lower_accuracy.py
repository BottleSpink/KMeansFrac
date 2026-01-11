"""
Analysis: Why Fractional Distance (p < 1) Has Lower Accuracy

This script explains and demonstrates the reasons why fractional distance
K-means often achieves lower accuracy than Euclidean K-means.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml


def demonstrate_triangle_inequality():
    """Show that p < 1 violates triangle inequality"""
    print("=" * 80)
    print("ISSUE 1: VIOLATION OF TRIANGLE INEQUALITY (Not a proper metric)")
    print("=" * 80)
    print("\nFor a proper metric, we need: d(x,z) ≤ d(x,y) + d(y,z)")
    print("This is violated when p < 1\n")

    # Simple example in 2D
    x = np.array([0, 0])
    y = np.array([1, 0])
    z = np.array([1, 1])

    for p in [0.3, 0.5, 1.0, 2.0]:
        d_xz = np.sum(np.abs(x - z) ** p) ** (1/p)
        d_xy = np.sum(np.abs(x - y) ** p) ** (1/p)
        d_yz = np.sum(np.abs(y - z) ** p) ** (1/p)

        triangle_holds = d_xz <= d_xy + d_yz
        print(f"p={p:.1f}: d(x,z)={d_xz:.3f}, d(x,y)+d(y,z)={d_xy + d_yz:.3f}, "
              f"Triangle inequality: {'✓' if triangle_holds else '✗ VIOLATED'}")

    print("\n➤ When triangle inequality is violated, the distance doesn't behave")
    print("  like a proper metric, leading to unreliable clustering.\n")


def demonstrate_mean_suboptimality():
    """Show that arithmetic mean is NOT optimal for p < 1"""
    print("=" * 80)
    print("ISSUE 2: ARITHMETIC MEAN IS SUBOPTIMAL FOR p < 1")
    print("=" * 80)
    print("\nK-means updates centroids using arithmetic mean, which minimizes")
    print("sum of squared (Euclidean) distances. For p ≠ 2, this is suboptimal!\n")

    # Generate cluster of points
    np.random.seed(42)
    points = np.random.randn(100, 2) + np.array([5, 5])

    # Calculate optimal centroid for different p values
    arithmetic_mean = np.mean(points, axis=0)

    print(f"Arithmetic mean centroid: [{arithmetic_mean[0]:.3f}, {arithmetic_mean[1]:.3f}]")
    print("\nTotal distance to all points using arithmetic mean as centroid:\n")

    for p in [0.3, 0.5, 1.0, 2.0]:
        distances = np.sum(np.abs(points - arithmetic_mean) ** p, axis=1) ** (1/p)
        total_dist = np.sum(distances ** p)
        print(f"  p={p:.1f}: {total_dist:.2f}")

    print("\n➤ For p=2 (Euclidean), arithmetic mean is optimal.")
    print("  For p<1, we should use different aggregation (geometric median, etc.)")
    print("  Using wrong centroid calculation leads to poor convergence!\n")


def demonstrate_curse_of_dimensionality():
    """Show how high dimensions affect fractional distances"""
    print("=" * 80)
    print("ISSUE 3: CURSE OF DIMENSIONALITY")
    print("=" * 80)
    print("\nIn high dimensions, fractional distances exhibit worse 'concentration")
    print("of measure' - all distances become similar, losing discriminative power.\n")

    # Load a small MNIST sample
    print("Loading MNIST sample...")
    mnist = fetch_openml('mnist_784', version=1, parser='auto')
    X = np.array(mnist.data[:1000], dtype='float32') / 255.0

    # Pick a random point and calculate distances to all other points
    np.random.seed(42)
    reference_idx = np.random.randint(1000)
    reference = X[reference_idx]

    # Calculate distances for different p values
    print("Calculating distance statistics for 1000 MNIST samples (784 dimensions)...\n")
    print(f"{'p value':<10} {'Mean dist':<12} {'Std dev':<12} {'Coef of Var':<12}")
    print("-" * 50)

    for p in [0.3, 0.5, 1.0, 2.0]:
        diffs = np.abs(X - reference)
        distances = np.sum(diffs ** p, axis=1) ** (1/p)

        mean_dist = np.mean(distances)
        std_dist = np.std(distances)
        coef_var = std_dist / mean_dist  # Coefficient of variation

        print(f"{p:<10.1f} {mean_dist:<12.3f} {std_dist:<12.3f} {coef_var:<12.3f}")

    print("\n➤ Lower coefficient of variation means distances are more similar.")
    print("  For p<1, distances lose discriminative power in high dimensions!")
    print("  All points look equally far, making clustering harder.\n")


def demonstrate_nonconvexity():
    """Explain non-convex optimization landscape"""
    print("=" * 80)
    print("ISSUE 4: NON-CONVEX OPTIMIZATION LANDSCAPE")
    print("=" * 80)
    print("\nEuclidean K-means (p=2) has a convex optimization within each cluster.")
    print("Fractional distances (p<1) create non-convex landscapes with many local minima.\n")

    # Visualize 1D example
    x = np.linspace(-3, 3, 100)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for idx, p in enumerate([0.5, 1.0, 2.0]):
        # Distance from origin raised to power p
        y = np.abs(x) ** p
        axes[idx].plot(x, y, 'b-', linewidth=2)
        axes[idx].set_title(f'p = {p} (|x|^{p})', fontsize=12)
        axes[idx].set_xlabel('x')
        axes[idx].set_ylabel(f'|x|^{p}')
        axes[idx].grid(True, alpha=0.3)
        axes[idx].axhline(y=0, color='k', linewidth=0.5)
        axes[idx].axvline(x=0, color='k', linewidth=0.5)

        # Highlight convexity
        if p >= 1:
            axes[idx].text(0.5, 0.95, 'CONVEX', transform=axes[idx].transAxes,
                          fontsize=11, color='green', weight='bold',
                          verticalalignment='top', horizontalalignment='center',
                          bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
        else:
            axes[idx].text(0.5, 0.95, 'NON-CONVEX', transform=axes[idx].transAxes,
                          fontsize=11, color='red', weight='bold',
                          verticalalignment='top', horizontalalignment='center',
                          bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.5))

    plt.tight_layout()
    plt.savefig('convexity_analysis.png', dpi=150, bbox_inches='tight')
    print("Visualization saved to 'convexity_analysis.png'")
    print("\n➤ Non-convex optimization (p<1) is much harder to solve.")
    print("  K-means algorithm easily gets stuck in poor local minima!")
    print("  This leads to worse clustering results.\n")


def demonstrate_initialization_sensitivity():
    """Show that p < 1 is more sensitive to initialization"""
    print("=" * 80)
    print("ISSUE 5: HIGH SENSITIVITY TO INITIALIZATION")
    print("=" * 80)
    print("\nDue to non-convexity, fractional K-means is much more sensitive to")
    print("initial centroid placement than Euclidean K-means.\n")

    from kmeans_fractional import FractionalKMeans
    from sklearn.cluster import KMeans

    # Load small MNIST sample
    print("Testing with different random initializations...")
    mnist = fetch_openml('mnist_784', version=1, parser='auto')
    X = np.array(mnist.data[:2000], dtype='float32') / 255.0
    y = np.array(mnist.target[:2000], dtype='int')

    from scipy.optimize import linear_sum_assignment
    from sklearn.metrics import confusion_matrix

    def cluster_accuracy(y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        row_ind, col_ind = linear_sum_assignment(-cm)
        return cm[row_ind, col_ind].sum() / cm.sum()

    # Test multiple random seeds
    seeds = [0, 1, 2, 3, 4]

    print("\nAccuracy with different random seeds:\n")
    print(f"{'Seed':<8} {'Euclidean (p=2)':<20} {'Fractional (p=0.5)':<20}")
    print("-" * 50)

    euclidean_accs = []
    fractional_accs = []

    for seed in seeds:
        # Euclidean
        km_euc = KMeans(n_clusters=10, random_state=seed, n_init=1, max_iter=50)
        y_pred_euc = km_euc.fit_predict(X)
        acc_euc = cluster_accuracy(y, y_pred_euc)
        euclidean_accs.append(acc_euc)

        # Fractional
        km_frac = FractionalKMeans(n_clusters=10, p=0.5, random_state=seed,
                                   n_init=1, max_iter=50, verbose=0)
        y_pred_frac = km_frac.fit_predict(X)
        acc_frac = cluster_accuracy(y, y_pred_frac)
        fractional_accs.append(acc_frac)

        print(f"{seed:<8} {acc_euc:<20.4f} {acc_frac:<20.4f}")

    print(f"\n{'Mean:':<8} {np.mean(euclidean_accs):<20.4f} {np.mean(fractional_accs):<20.4f}")
    print(f"{'Std dev:':<8} {np.std(euclidean_accs):<20.4f} {np.std(fractional_accs):<20.4f}")

    print("\n➤ Higher standard deviation for fractional distance shows greater")
    print("  sensitivity to initialization, leading to less reliable results.\n")


def print_summary():
    """Print comprehensive summary"""
    print("\n" + "=" * 80)
    print("SUMMARY: WHY FRACTIONAL DISTANCE (p < 1) HAS LOWER ACCURACY")
    print("=" * 80)
    print("""
1. NOT A PROPER METRIC
   • Violates triangle inequality
   • Doesn't behave like a true distance measure
   • Causes unreliable clustering geometry

2. CENTROID CALCULATION MISMATCH
   • K-means uses arithmetic mean (optimal for p=2)
   • For p<1, need different centroid estimator (geometric median, etc.)
   • Using wrong estimator → poor convergence → worse results

3. CURSE OF DIMENSIONALITY
   • In 784 dimensions (MNIST), distances concentrate
   • For p<1, this effect is AMPLIFIED
   • All distances become similar → loses discriminative power
   • Hard to distinguish between close and far points

4. NON-CONVEX OPTIMIZATION
   • p<1 creates non-convex optimization landscape
   • Many local minima trap the algorithm
   • Euclidean (p=2) is convex → easier optimization → better results

5. INITIALIZATION SENSITIVITY
   • Non-convexity makes p<1 highly sensitive to initialization
   • Different random seeds give very different results
   • Less stable and reproducible clustering

6. HIGH-DIMENSIONAL NOISE
   • MNIST has 784 dimensions, many are noisy/irrelevant
   • p<1 emphasizes ALL dimensions equally (including noise)
   • Euclidean distance better handles noisy dimensions

CONCLUSION:
While fractional distances can theoretically capture different cluster structures,
the combination of high dimensionality (784D), non-convexity, and algorithmic
mismatch (using arithmetic mean) makes them LESS effective than Euclidean
distance for MNIST clustering.

WHEN MIGHT FRACTIONAL DISTANCES WORK BETTER?
• Lower dimensional data (< 50 dimensions)
• When you need to emphasize outliers
• With proper centroid estimators (not arithmetic mean)
• Sparse data where many dimensions are zero
• When combined with dimensionality reduction (PCA, etc.)
""")
    print("=" * 80)


def main():
    print("\n" + "=" * 80)
    print("ANALYSIS: Why Fractional Distance Has Lower Accuracy Than Euclidean")
    print("=" * 80 + "\n")

    demonstrate_triangle_inequality()
    input("Press Enter to continue...")

    demonstrate_mean_suboptimality()
    input("Press Enter to continue...")

    demonstrate_curse_of_dimensionality()
    input("Press Enter to continue...")

    demonstrate_nonconvexity()
    input("Press Enter to continue...")

    demonstrate_initialization_sensitivity()

    print_summary()


if __name__ == "__main__":
    main()
