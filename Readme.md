# K-Means Clustering on MNIST

This project implements K-means clustering on the MNIST handwritten digit dataset with both **Euclidean distance** and **Fractional distance** metrics.

## Overview

The implementation provides two distance metrics for K-means clustering:

1. **Euclidean Distance (p=2)**: Standard K-means using scikit-learn
2. **Fractional Distance (0 < p < 1)**: Custom implementation using Minkowski distance with fractional exponents

The fractional distance is defined as: `d(x, y) = (Σ|x_i - y_i|^p)^(1/p)` where 0 < p < 1.

Fractional distances create non-convex optimization problems and can discover different cluster structures compared to Euclidean distance.

## Features

- Load and preprocess MNIST dataset
- **Standard K-means** with Euclidean distance (scikit-learn)
- **Fractional K-means** with customizable p parameter (custom implementation)
- **Comparison tool** to evaluate different distance metrics
- Calculate clustering accuracy with optimal label mapping using Hungarian algorithm
- Visualize cluster centers as digit images
- Display confusion matrix and cluster distribution
- Performance comparison across multiple distance metrics

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### 1. Standard K-means with Euclidean Distance

```bash
python kmeans_mnist.py
```

Runs standard K-means (p=2) using scikit-learn's optimized implementation.

### 2. K-means with Fractional Distance

```bash
python kmeans_fractional.py
```

Runs custom K-means with fractional distance (default p=0.5). You can modify the `p_value` variable in the script to experiment with different fractional distances.

### 3. Compare Distance Metrics

```bash
python compare_distances.py
```

Compares Euclidean distance vs multiple fractional distances (p=0.3, 0.5, 0.7) and generates:
- Performance comparison chart
- Accuracy, time, and convergence analysis
- Visual comparison saved to `distance_comparison.png`

## What Each Script Does

**kmeans_mnist.py**:
1. Load 10,000 MNIST samples
2. Apply standard K-means (Euclidean distance)
3. Calculate and display clustering accuracy
4. Save cluster centers to `cluster_centers.png`

**kmeans_fractional.py**:
1. Load 10,000 MNIST samples
2. Apply K-means with fractional distance (default p=0.5)
3. Calculate and display clustering accuracy
4. Save cluster centers to `cluster_centers_fractional.png`

**compare_distances.py**:
1. Load 5,000 MNIST samples
2. Run K-means with Euclidean and multiple fractional distances
3. Compare accuracy, speed, and convergence
4. Generate comparison visualization

## Results

### Euclidean Distance (p=2)
- Clustering accuracy: ~50-60%
- Fastest computation (optimized implementation)
- Standard convex optimization

### Fractional Distance (0 < p < 1)
- Clustering accuracy: Varies by p value
- Slower than Euclidean (custom Python implementation)
- Non-convex optimization, may find different cluster structures
- Lower p values emphasize differences more strongly

## Requirements

- numpy >= 1.21.0
- scikit-learn >= 1.0.0
- matplotlib >= 3.4.0
- scipy >= 1.7.0
