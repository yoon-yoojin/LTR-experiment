"""Dataset classes for Learning to Rank."""
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import List, Tuple, Dict
import random


class LTRDataset:
    """Base dataset class for LTR data in LETOR format."""

    def __init__(self, file_path: str):
        """Initialize LTR dataset.

        Args:
            file_path: Path to data file in LETOR format
        """
        self.queries, self.features, self.labels = self._load_data(file_path)
        self.qids = list(self.queries.keys())
        self.num_features = self.features[self.qids[0]].shape[1] if self.qids else 0

    def _load_data(self, file_path: str) -> Tuple[Dict, Dict, Dict]:
        """Load data from LETOR format file.

        Format: <label> qid:<qid> <feature>:<value> ... <feature>:<value>

        Args:
            file_path: Path to data file

        Returns:
            Tuple of (queries, features, labels) dictionaries
        """
        queries = {}
        features = {}
        labels = {}

        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue

                parts = line.split()
                label = float(parts[0])
                qid = parts[1].split(':')[1]

                # Parse features
                feat_dict = {}
                for part in parts[2:]:
                    if ':' in part:
                        feat_id, feat_val = part.split(':')
                        feat_dict[int(feat_id)] = float(feat_val)

                # Convert to array (assuming features are 1-indexed)
                if feat_dict:
                    max_feat_id = max(feat_dict.keys())
                    feat_array = np.zeros(max_feat_id)
                    for feat_id, feat_val in feat_dict.items():
                        feat_array[feat_id - 1] = feat_val

                    if qid not in queries:
                        queries[qid] = []
                        features[qid] = []
                        labels[qid] = []

                    queries[qid].append(qid)
                    features[qid].append(feat_array)
                    labels[qid].append(label)

        # Convert lists to numpy arrays
        for qid in queries:
            features[qid] = np.array(features[qid], dtype=np.float32)
            labels[qid] = np.array(labels[qid], dtype=np.float32)

        return queries, features, labels

    def __len__(self) -> int:
        """Return number of queries."""
        return len(self.qids)

    def get_query_data(self, qid: str) -> Tuple[np.ndarray, np.ndarray]:
        """Get features and labels for a query.

        Args:
            qid: Query ID

        Returns:
            Tuple of (features, labels)
        """
        return self.features[qid], self.labels[qid]


class PairwiseDataset(Dataset):
    """Dataset for pairwise LTR approach."""

    def __init__(self, base_dataset: LTRDataset, num_pairs_per_query: int = 10):
        """Initialize pairwise dataset.

        Args:
            base_dataset: Base LTR dataset
            num_pairs_per_query: Number of pairs to generate per query
        """
        self.base_dataset = base_dataset
        self.num_pairs_per_query = num_pairs_per_query
        self.pairs = self._generate_pairs()

    def _generate_pairs(self) -> List[Tuple[str, int, int]]:
        """Generate pairs of documents for each query.

        Returns:
            List of (qid, doc_i_idx, doc_j_idx) tuples where doc_i > doc_j
        """
        pairs = []

        for qid in self.base_dataset.qids:
            features, labels = self.base_dataset.get_query_data(qid)

            # Find all valid pairs where label_i > label_j
            valid_pairs = []
            for i in range(len(labels)):
                for j in range(len(labels)):
                    if labels[i] > labels[j]:
                        valid_pairs.append((qid, i, j))

            # Sample pairs if too many
            if len(valid_pairs) > self.num_pairs_per_query:
                valid_pairs = random.sample(valid_pairs, self.num_pairs_per_query)

            pairs.extend(valid_pairs)

        return pairs

    def __len__(self) -> int:
        """Return number of pairs."""
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get a pair of documents.

        Args:
            idx: Index of pair

        Returns:
            Tuple of (doc_i_features, doc_j_features, label)
            label = 1 if doc_i > doc_j, 0 otherwise
        """
        qid, i, j = self.pairs[idx]
        features, labels = self.base_dataset.get_query_data(qid)

        doc_i = torch.tensor(features[i], dtype=torch.float32)
        doc_j = torch.tensor(features[j], dtype=torch.float32)
        label = torch.tensor(1.0 if labels[i] > labels[j] else 0.0, dtype=torch.float32)

        return doc_i, doc_j, label


class ListwiseDataset(Dataset):
    """Dataset for listwise LTR approach."""

    def __init__(self, base_dataset: LTRDataset, list_size: int = 10):
        """Initialize listwise dataset.

        Args:
            base_dataset: Base LTR dataset
            list_size: Maximum number of documents per list
        """
        self.base_dataset = base_dataset
        self.list_size = list_size
        self.qids = base_dataset.qids

    def __len__(self) -> int:
        """Return number of queries."""
        return len(self.qids)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get a list of documents for a query.

        Args:
            idx: Index of query

        Returns:
            Tuple of (features, labels, mask)
            - features: [list_size, num_features]
            - labels: [list_size]
            - mask: [list_size] (1 for valid docs, 0 for padding)
        """
        qid = self.qids[idx]
        features, labels = self.base_dataset.get_query_data(qid)

        num_docs = len(labels)
        actual_size = min(num_docs, self.list_size)

        # Create padded tensors
        feature_tensor = torch.zeros(self.list_size, features.shape[1], dtype=torch.float32)
        label_tensor = torch.zeros(self.list_size, dtype=torch.float32)
        mask = torch.zeros(self.list_size, dtype=torch.float32)

        # Fill with actual data
        feature_tensor[:actual_size] = torch.tensor(features[:actual_size], dtype=torch.float32)
        label_tensor[:actual_size] = torch.tensor(labels[:actual_size], dtype=torch.float32)
        mask[:actual_size] = 1.0

        return feature_tensor, label_tensor, mask


def collate_fn_listwise(batch):
    """Collate function for listwise dataloader.

    Args:
        batch: List of (features, labels, mask) tuples

    Returns:
        Batched tensors
    """
    features = torch.stack([item[0] for item in batch])
    labels = torch.stack([item[1] for item in batch])
    masks = torch.stack([item[2] for item in batch])

    return features, labels, masks
