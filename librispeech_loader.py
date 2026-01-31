"""
LibriSpeech dataset loader for K-means clustering

Loads LibriSpeech dev-clean subset and extracts MFCC features for clustering.
"""

import numpy as np
from datasets import load_dataset
import librosa


def extract_mfcc_features(audio_array, sample_rate=16000, n_mfcc=13, max_len=100):
    """
    Extract MFCC features from audio array.

    Args:
        audio_array: Raw audio samples
        sample_rate: Audio sample rate (LibriSpeech uses 16kHz)
        n_mfcc: Number of MFCC coefficients to extract
        max_len: Maximum number of time frames (for fixed-size output)

    Returns:
        Flattened MFCC feature vector
    """
    # Extract MFCCs
    mfccs = librosa.feature.mfcc(
        y=audio_array.astype(np.float32),
        sr=sample_rate,
        n_mfcc=n_mfcc
    )

    # Add delta and delta-delta features
    delta_mfccs = librosa.feature.delta(mfccs)
    delta2_mfccs = librosa.feature.delta(mfccs, order=2)

    # Stack all features
    features = np.vstack([mfccs, delta_mfccs, delta2_mfccs])

    # Pad or truncate to fixed length
    if features.shape[1] < max_len:
        # Pad with zeros
        padding = np.zeros((features.shape[0], max_len - features.shape[1]))
        features = np.hstack([features, padding])
    else:
        # Truncate
        features = features[:, :max_len]

    # Flatten to 1D vector
    return features.flatten()


def load_librispeech(n_samples=None, subset="dev-clean", n_mfcc=13, max_len=100):
    """
    Load LibriSpeech dataset and extract MFCC features.

    Args:
        n_samples: Number of samples to load (None for all)
        subset: LibriSpeech subset to use (default: "dev-clean")
        n_mfcc: Number of MFCC coefficients
        max_len: Maximum time frames for feature extraction

    Returns:
        X: Feature matrix (n_samples, n_features)
        y: Speaker IDs as labels
        metadata: Dictionary with additional info (text, audio_ids)
    """
    print(f"Loading LibriSpeech {subset} dataset...")

    # Load dataset from Hugging Face
    # LibriSpeech subsets: train.clean.100, train.clean.360, train.other.500,
    #                     validation.clean, validation.other, test.clean, test.other
    hf_subset_map = {
        "dev-clean": "clean",
        "dev-other": "other",
        "test-clean": "clean",
        "test-other": "other",
    }

    # Determine split name
    if "dev" in subset:
        split = "validation." + hf_subset_map.get(subset, "clean")
    elif "test" in subset:
        split = "test." + hf_subset_map.get(subset, "clean")
    else:
        split = subset

    dataset = load_dataset(
        "librispeech_asr",
        hf_subset_map.get(subset, "clean"),
        split=split.replace(".", ""),
        trust_remote_code=True
    )

    # Limit samples if specified
    if n_samples and n_samples < len(dataset):
        np.random.seed(42)
        indices = np.random.choice(len(dataset), n_samples, replace=False)
        dataset = dataset.select(indices)

    print(f"Processing {len(dataset)} audio samples...")

    # Extract features
    features_list = []
    speaker_ids = []
    texts = []
    audio_ids = []

    for i, sample in enumerate(dataset):
        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(dataset)} samples...")

        # Get audio array and sample rate
        audio_array = np.array(sample["audio"]["array"])
        sample_rate = sample["audio"]["sampling_rate"]

        # Extract MFCC features
        features = extract_mfcc_features(
            audio_array,
            sample_rate=sample_rate,
            n_mfcc=n_mfcc,
            max_len=max_len
        )
        features_list.append(features)

        # Store metadata
        speaker_ids.append(sample["speaker_id"])
        texts.append(sample["text"])
        audio_ids.append(sample["id"])

    X = np.array(features_list, dtype='float32')
    y = np.array(speaker_ids, dtype='int')

    metadata = {
        "texts": texts,
        "audio_ids": audio_ids,
        "n_mfcc": n_mfcc,
        "max_len": max_len,
        "feature_dim": X.shape[1],
        "subset": subset
    }

    print(f"Feature matrix shape: {X.shape}")
    print(f"Number of unique speakers: {len(np.unique(y))}")

    return X, y, metadata


def load_librispeech_simple(n_samples=None):
    """
    Simplified loader that matches the load_mnist interface.

    Args:
        n_samples: Number of samples to load

    Returns:
        X: Feature matrix
        y: Speaker IDs
    """
    X, y, _ = load_librispeech(n_samples=n_samples, subset="dev-clean")
    return X, y


if __name__ == "__main__":
    # Test the loader
    print("Testing LibriSpeech loader...")

    # Load a small subset for testing
    X, y, metadata = load_librispeech(n_samples=100, subset="dev-clean")

    print(f"\nDataset loaded successfully!")
    print(f"  Feature matrix shape: {X.shape}")
    print(f"  Labels shape: {y.shape}")
    print(f"  Number of unique speakers: {len(np.unique(y))}")
    print(f"  Feature dimension: {metadata['feature_dim']}")
    print(f"\nSample texts:")
    for i in range(min(3, len(metadata['texts']))):
        print(f"  [{i}] {metadata['texts'][i][:80]}...")
