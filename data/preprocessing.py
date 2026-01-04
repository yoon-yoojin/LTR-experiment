"""Data preprocessing utilities."""
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from typing import Dict, Tuple, Optional
import pickle


class LTRPreprocessor:
    """Preprocessor for LTR features."""

    def __init__(self, method: str = 'standard'):
        """Initialize preprocessor.

        Args:
            method: Scaling method ('standard', 'minmax', or 'none')
        """
        self.method = method
        self.scaler = None

        if method == 'standard':
            self.scaler = StandardScaler()
        elif method == 'minmax':
            self.scaler = MinMaxScaler()
        elif method != 'none':
            raise ValueError(f"Unknown scaling method: {method}")

    def fit(self, features_dict: Dict[str, np.ndarray]) -> 'LTRPreprocessor':
        """Fit preprocessor on features.

        Args:
            features_dict: Dictionary of {qid: features}

        Returns:
            Self
        """
        if self.scaler is None:
            return self

        # Concatenate all features
        all_features = np.vstack([features for features in features_dict.values()])

        # Fit scaler
        self.scaler.fit(all_features)

        return self

    def transform(self, features_dict: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Transform features.

        Args:
            features_dict: Dictionary of {qid: features}

        Returns:
            Transformed features dictionary
        """
        if self.scaler is None:
            return features_dict

        transformed = {}
        for qid, features in features_dict.items():
            transformed[qid] = self.scaler.transform(features)

        return transformed

    def fit_transform(self, features_dict: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Fit and transform features.

        Args:
            features_dict: Dictionary of {qid: features}

        Returns:
            Transformed features dictionary
        """
        self.fit(features_dict)
        return self.transform(features_dict)

    def save(self, path: str) -> None:
        """Save preprocessor to file.

        Args:
            path: Path to save preprocessor
        """
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str) -> 'LTRPreprocessor':
        """Load preprocessor from file.

        Args:
            path: Path to saved preprocessor

        Returns:
            Loaded preprocessor
        """
        with open(path, 'rb') as f:
            return pickle.load(f)


def remove_missing_features(features: np.ndarray, threshold: float = 0.9) -> Tuple[np.ndarray, np.ndarray]:
    """Remove features with too many missing values.

    Args:
        features: Feature array [num_samples, num_features]
        threshold: Maximum ratio of missing values allowed

    Returns:
        Tuple of (cleaned_features, valid_feature_indices)
    """
    missing_ratio = np.sum(features == 0, axis=0) / features.shape[0]
    valid_features = missing_ratio < threshold
    valid_indices = np.where(valid_features)[0]

    return features[:, valid_features], valid_indices


def create_train_val_split(
    queries: Dict,
    features: Dict,
    labels: Dict,
    val_ratio: float = 0.2,
    seed: int = 42
) -> Tuple[Dict, Dict, Dict, Dict, Dict, Dict]:
    """Split data into train and validation sets by query.

    Args:
        queries: Query dictionary
        features: Features dictionary
        labels: Labels dictionary
        val_ratio: Ratio of validation data
        seed: Random seed

    Returns:
        Tuple of (train_queries, train_features, train_labels,
                  val_queries, val_features, val_labels)
    """
    np.random.seed(seed)
    qids = list(queries.keys())
    np.random.shuffle(qids)

    split_idx = int(len(qids) * (1 - val_ratio))
    train_qids = qids[:split_idx]
    val_qids = qids[split_idx:]

    train_queries = {qid: queries[qid] for qid in train_qids}
    train_features = {qid: features[qid] for qid in train_qids}
    train_labels = {qid: labels[qid] for qid in train_qids}

    val_queries = {qid: queries[qid] for qid in val_qids}
    val_features = {qid: features[qid] for qid in val_qids}
    val_labels = {qid: labels[qid] for qid in val_qids}

    return train_queries, train_features, train_labels, val_queries, val_features, val_labels
