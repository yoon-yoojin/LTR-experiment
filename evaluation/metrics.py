"""Evaluation metrics for Learning to Rank."""
import numpy as np
import torch
from typing import List, Dict, Union


def dcg_at_k(relevance: np.ndarray, k: int) -> float:
    """Compute Discounted Cumulative Gain at k.

    Args:
        relevance: Relevance scores in ranked order
        k: Cutoff position

    Returns:
        DCG@k value
    """
    relevance = np.asarray(relevance)[:k]
    if relevance.size == 0:
        return 0.0

    # DCG = sum(rel_i / log2(i + 1)) for i in 1..k
    discounts = np.log2(np.arange(2, relevance.size + 2))
    return np.sum(relevance / discounts)


def ndcg_at_k(relevance: np.ndarray, k: int) -> float:
    """Compute Normalized Discounted Cumulative Gain at k.

    Args:
        relevance: Relevance scores in ranked order
        k: Cutoff position

    Returns:
        NDCG@k value
    """
    dcg = dcg_at_k(relevance, k)
    ideal_relevance = np.sort(relevance)[::-1]
    idcg = dcg_at_k(ideal_relevance, k)

    if idcg == 0:
        return 0.0

    return dcg / idcg


def precision_at_k(relevance: np.ndarray, k: int, threshold: float = 0.0) -> float:
    """Compute Precision at k.

    Args:
        relevance: Relevance scores in ranked order
        k: Cutoff position
        threshold: Relevance threshold for considering as relevant

    Returns:
        Precision@k value
    """
    relevance = np.asarray(relevance)[:k]
    if relevance.size == 0:
        return 0.0

    return np.sum(relevance > threshold) / k


def recall_at_k(relevance: np.ndarray, k: int, threshold: float = 0.0) -> float:
    """Compute Recall at k.

    Args:
        relevance: Relevance scores in ranked order
        k: Cutoff position
        threshold: Relevance threshold for considering as relevant

    Returns:
        Recall@k value
    """
    relevance = np.asarray(relevance)
    total_relevant = np.sum(relevance > threshold)

    if total_relevant == 0:
        return 0.0

    relevant_at_k = np.sum(relevance[:k] > threshold)
    return relevant_at_k / total_relevant


def average_precision(relevance: np.ndarray, threshold: float = 0.0) -> float:
    """Compute Average Precision.

    Args:
        relevance: Relevance scores in ranked order
        threshold: Relevance threshold for considering as relevant

    Returns:
        Average Precision value
    """
    relevance = np.asarray(relevance)
    relevant_mask = relevance > threshold

    if np.sum(relevant_mask) == 0:
        return 0.0

    precisions = []
    for k in range(1, len(relevance) + 1):
        if relevant_mask[k - 1]:
            precisions.append(precision_at_k(relevance, k, threshold))

    return np.mean(precisions) if precisions else 0.0


def mean_average_precision(relevances: List[np.ndarray], threshold: float = 0.0) -> float:
    """Compute Mean Average Precision across multiple queries.

    Args:
        relevances: List of relevance scores for each query
        threshold: Relevance threshold

    Returns:
        MAP value
    """
    aps = [average_precision(rel, threshold) for rel in relevances]
    return np.mean(aps) if aps else 0.0


def reciprocal_rank(relevance: np.ndarray, threshold: float = 0.0) -> float:
    """Compute Reciprocal Rank.

    Args:
        relevance: Relevance scores in ranked order
        threshold: Relevance threshold for considering as relevant

    Returns:
        Reciprocal Rank value
    """
    relevance = np.asarray(relevance)
    relevant_positions = np.where(relevance > threshold)[0]

    if len(relevant_positions) == 0:
        return 0.0

    return 1.0 / (relevant_positions[0] + 1)


def mean_reciprocal_rank(relevances: List[np.ndarray], threshold: float = 0.0) -> float:
    """Compute Mean Reciprocal Rank across multiple queries.

    Args:
        relevances: List of relevance scores for each query
        threshold: Relevance threshold

    Returns:
        MRR value
    """
    rrs = [reciprocal_rank(rel, threshold) for rel in relevances]
    return np.mean(rrs) if rrs else 0.0


class RankingMetrics:
    """Compute multiple ranking metrics at once."""

    def __init__(self, k_values: List[int] = [1, 3, 5, 10]):
        """Initialize ranking metrics.

        Args:
            k_values: List of k values for metrics@k
        """
        self.k_values = k_values

    def compute_metrics(
        self,
        scores: Union[np.ndarray, torch.Tensor],
        labels: Union[np.ndarray, torch.Tensor],
        threshold: float = 0.0
    ) -> Dict[str, float]:
        """Compute all metrics for a single query.

        Args:
            scores: Predicted scores [list_size]
            labels: Ground truth relevance labels [list_size]
            threshold: Relevance threshold

        Returns:
            Dictionary of metric values
        """
        # Convert to numpy
        if isinstance(scores, torch.Tensor):
            scores = scores.cpu().numpy()
        if isinstance(labels, torch.Tensor):
            labels = labels.cpu().numpy()

        # Rank documents by scores (descending)
        ranking = np.argsort(scores)[::-1]
        ranked_labels = labels[ranking]

        metrics = {}

        # NDCG@k
        for k in self.k_values:
            metrics[f'ndcg@{k}'] = ndcg_at_k(ranked_labels, k)

        # Precision@k and Recall@k
        for k in self.k_values:
            metrics[f'precision@{k}'] = precision_at_k(ranked_labels, k, threshold)
            metrics[f'recall@{k}'] = recall_at_k(ranked_labels, k, threshold)

        # MAP and MRR
        metrics['map'] = average_precision(ranked_labels, threshold)
        metrics['mrr'] = reciprocal_rank(ranked_labels, threshold)

        return metrics

    def compute_metrics_batch(
        self,
        scores_list: List[Union[np.ndarray, torch.Tensor]],
        labels_list: List[Union[np.ndarray, torch.Tensor]],
        threshold: float = 0.0
    ) -> Dict[str, float]:
        """Compute average metrics across multiple queries.

        Args:
            scores_list: List of predicted scores for each query
            labels_list: List of ground truth labels for each query
            threshold: Relevance threshold

        Returns:
            Dictionary of averaged metric values
        """
        all_metrics = []

        for scores, labels in zip(scores_list, labels_list):
            metrics = self.compute_metrics(scores, labels, threshold)
            all_metrics.append(metrics)

        # Average metrics
        avg_metrics = {}
        for key in all_metrics[0].keys():
            avg_metrics[key] = np.mean([m[key] for m in all_metrics])

        return avg_metrics


def evaluate_model(
    model,
    dataset,
    device: str = 'cpu',
    k_values: List[int] = [1, 3, 5, 10]
) -> Dict[str, float]:
    """Evaluate model on dataset.

    Args:
        model: Trained model
        dataset: LTRDataset
        device: Device to use
        k_values: List of k values for metrics@k

    Returns:
        Dictionary of metric values
    """
    model.eval()
    model = model.to(device)

    metrics_calculator = RankingMetrics(k_values)
    all_scores = []
    all_labels = []

    with torch.no_grad():
        for qid in dataset.qids:
            features, labels = dataset.get_query_data(qid)
            features_tensor = torch.tensor(features, dtype=torch.float32).to(device)

            # Get predictions
            scores = model.predict(features_tensor)

            all_scores.append(scores.cpu().numpy())
            all_labels.append(labels)

    # Compute metrics
    metrics = metrics_calculator.compute_metrics_batch(all_scores, all_labels)

    return metrics
