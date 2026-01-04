"""Pairwise LTR models: RankNet and LambdaRank."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple
from .base import BaseRankingModel


class RankNet(BaseRankingModel):
    """RankNet: Learning to Rank using Gradient Descent.

    Paper: https://icml.cc/2015/wp-content/uploads/2015/06/icml_ranking.pdf
    """

    def __init__(
        self,
        num_features: int,
        hidden_dims: List[int] = [256, 128, 64],
        dropout: float = 0.2,
        activation: str = 'relu',
        sigma: float = 1.0
    ):
        """Initialize RankNet model.

        Args:
            num_features: Number of input features
            hidden_dims: List of hidden layer dimensions
            dropout: Dropout probability
            activation: Activation function
            sigma: Temperature parameter for sigmoid
        """
        super().__init__(num_features, hidden_dims, dropout, activation)
        self.sigma = sigma

    def compute_loss(
        self,
        doc_i: torch.Tensor,
        doc_j: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """Compute RankNet loss.

        Args:
            doc_i: Features for document i [batch_size, num_features]
            doc_j: Features for document j [batch_size, num_features]
            target: Target labels (1 if doc_i > doc_j, 0 otherwise) [batch_size]

        Returns:
            Loss value
        """
        # Get scores
        score_i = self.forward(doc_i)
        score_j = self.forward(doc_j)

        # Compute probability that doc_i > doc_j
        # P_ij = 1 / (1 + exp(-sigma * (s_i - s_j)))
        score_diff = self.sigma * (score_i - score_j)
        prob_ij = torch.sigmoid(score_diff)

        # Cross-entropy loss
        # L = -target * log(P_ij) - (1 - target) * log(1 - P_ij)
        loss = F.binary_cross_entropy(prob_ij, target)

        return loss


class LambdaRank(BaseRankingModel):
    """LambdaRank: RankNet with NDCG-based gradients.

    Paper: From RankNet to LambdaRank to LambdaMART: An Overview
    """

    def __init__(
        self,
        num_features: int,
        hidden_dims: List[int] = [256, 128, 64],
        dropout: float = 0.2,
        activation: str = 'relu',
        sigma: float = 1.0
    ):
        """Initialize LambdaRank model.

        Args:
            num_features: Number of input features
            hidden_dims: List of hidden layer dimensions
            dropout: Dropout probability
            activation: Activation function
            sigma: Temperature parameter for sigmoid
        """
        super().__init__(num_features, hidden_dims, dropout, activation)
        self.sigma = sigma

    def compute_lambda_weights(
        self,
        labels: torch.Tensor,
        scores: torch.Tensor
    ) -> torch.Tensor:
        """Compute lambda weights based on NDCG change.

        Args:
            labels: Relevance labels [list_size]
            scores: Predicted scores [list_size]

        Returns:
            Lambda weights [list_size, list_size]
        """
        list_size = labels.size(0)

        # Compute ideal DCG (IDCG)
        sorted_labels, _ = torch.sort(labels, descending=True)
        positions = torch.arange(1, list_size + 1, dtype=torch.float32, device=labels.device)
        idcg = torch.sum((2 ** sorted_labels - 1) / torch.log2(positions + 1))

        if idcg == 0:
            return torch.zeros(list_size, list_size, device=labels.device)

        # Compute delta NDCG for swapping positions
        labels_2d = labels.unsqueeze(1).expand(list_size, list_size)
        labels_diff = torch.abs(labels_2d - labels_2d.t())

        # Get ranking positions
        _, ranking = torch.sort(scores, descending=True)
        positions = torch.zeros_like(ranking, dtype=torch.float32)
        positions[ranking] = torch.arange(1, list_size + 1, dtype=torch.float32, device=labels.device)

        # Compute delta NDCG
        pos_i = positions.unsqueeze(1).expand(list_size, list_size)
        pos_j = positions.unsqueeze(0).expand(list_size, list_size)

        delta_ndcg = torch.abs(
            (2 ** labels_2d - 1) / torch.log2(pos_i + 1) -
            (2 ** labels_2d - 1) / torch.log2(pos_j + 1)
        ) / idcg

        # Lambda weights
        lambda_weights = delta_ndcg * labels_diff

        return lambda_weights

    def compute_loss(
        self,
        doc_i: torch.Tensor,
        doc_j: torch.Tensor,
        target: torch.Tensor,
        labels_i: torch.Tensor = None,
        labels_j: torch.Tensor = None
    ) -> torch.Tensor:
        """Compute LambdaRank loss.

        Args:
            doc_i: Features for document i [batch_size, num_features]
            doc_j: Features for document j [batch_size, num_features]
            target: Target labels (1 if doc_i > doc_j, 0 otherwise) [batch_size]
            labels_i: Relevance labels for document i (optional)
            labels_j: Relevance labels for document j (optional)

        Returns:
            Loss value
        """
        # Get scores
        score_i = self.forward(doc_i)
        score_j = self.forward(doc_j)

        # Compute probability that doc_i > doc_j
        score_diff = self.sigma * (score_i - score_j)
        prob_ij = torch.sigmoid(score_diff)

        # Cross-entropy loss
        loss = F.binary_cross_entropy(prob_ij, target)

        # If relevance labels provided, weight by lambda (simplified version)
        if labels_i is not None and labels_j is not None:
            label_diff = torch.abs(labels_i - labels_j)
            weights = label_diff / (label_diff.max() + 1e-8)
            loss = (loss * weights).mean()

        return loss
