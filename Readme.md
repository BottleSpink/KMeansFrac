# K-Means Clustering on MNIST

This project implements K-means clustering on the MNIST handwritten digit dataset.

## Overview

The implementation uses scikit-learn's K-means algorithm to cluster MNIST digits into 10 clusters (corresponding to digits 0-9). The clustering accuracy is calculated using the Hungarian algorithm to find the optimal mapping between clusters and true labels.

## Features

- Load and preprocess MNIST dataset
- Apply K-means clustering with k=10
- Calculate clustering accuracy with optimal label mapping
- Visualize cluster centers as digit images
- Display confusion matrix and cluster distribution

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python kmeans_mnist.py
```

The script will:
1. Load 10,000 MNIST samples
2. Apply K-means clustering
3. Calculate and display clustering accuracy
4. Save cluster centers visualization to `cluster_centers.png`
5. Display confusion matrix and cluster statistics

## Results

The implementation achieves clustering accuracy of approximately 50-60% on the MNIST dataset, which is expected for unsupervised K-means clustering without labels.

## Requirements

- numpy >= 1.21.0
- scikit-learn >= 1.0.0
- matplotlib >= 3.4.0
- scipy >= 1.7.0
