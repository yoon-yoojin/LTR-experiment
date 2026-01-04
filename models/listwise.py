"""Listwise LTR models: ListNet and ListMLE."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
from .base import BaseRankingModel


class ListNet(BaseRankingModel):
    """ListNet: Learning to Rank using Listwise Approach.

    Paper: Learning to Rank: From Pairwise Approach to Listwise Approach (Cao et al., 2007)
    """

    def __init__(
        self,
        num_features: int,
        hidden_dims: List[int] = [256, 128, 64],
        dropout: float = 0.2,
        activation: str = 'relu',
        temperature: float = 1.0
    ):
        """Initialize ListNet model.

        Args:
            num_features: Number of input features
            hidden_dims: List of hidden layer dimensions
            dropout: Dropout probability
            activation: Activation function
            temperature: Temperature parameter for softmax
        """
        super().__init__(num_features, hidden_dims, dropout, activation)
        self.temperature = temperature

    def top_k_probability(
        self,
        scores: torch.Tensor,
        k: int = 1
    ) -> torch.Tensor:
        """Compute top-k probability distribution.

        Args:
            scores: Predicted scores [batch_size, list_size]
            k: Number of top positions to consider

        Returns:
            Probability distribution [batch_size, list_size]
        """
        # Apply temperature
        scores = scores / self.temperature

        # For top-1, this is just softmax
        if k == 1:
            return F.softmax(scores, dim=-1)

        # For top-k, compute permutation probability (simplified as softmax)
        return F.softmax(scores, dim=-1)

    def compute_loss(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """Compute ListNet loss using cross-entropy.

        Args:
            features: Document features [batch_size, list_size, num_features]
            labels: Relevance labels [batch_size, list_size]
            mask: Valid document mask [batch_size, list_size]

        Returns:
            Loss value
        """
        # Get predicted scores
        scores = self.forward(features)  # [batch_size, list_size]

        # Apply mask to scores and labels
        scores = scores.masked_fill(mask == 0, float('-inf'))
        labels = labels.masked_fill(mask == 0, 0)

        # Compute probability distributions
        pred_probs = self.top_k_probability(scores, k=1)
        target_probs = self.top_k_probability(labels, k=1)

        # Cross-entropy loss
        # KL divergence: sum(P(y) * log(P(y) / P(y_hat)))
        loss = -torch.sum(target_probs * torch.log(pred_probs + 1e-10), dim=-1)
        loss = loss.mean()

        return loss


class ListMLE(BaseRankingModel):
    """ListMLE: Listwise approach using Maximum Likelihood Estimation.

    Paper: Listwise Approach to Learning to Rank - Theory and Algorithm (Xia et al., 2008)
    """

    def __init__(
        self,
        num_features: int,
        hidden_dims: List[int] = [256, 128, 64],
        dropout: float = 0.2,
        activation: str = 'relu'
    ):
        """Initialize ListMLE model.

        Args:
            num_features: Number of input features
            hidden_dims: List of hidden layer dimensions
            dropout: Dropout probability
            activation: Activation function
        """
        super().__init__(num_features, hidden_dims, dropout, activation)

    def compute_loss(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """Compute ListMLE loss.

        Args:
            features: Document features [batch_size, list_size, num_features]
            labels: Relevance labels [batch_size, list_size]
            mask: Valid document mask [batch_size, list_size]

        Returns:
            Loss value
        """
        # Get predicted scores
        scores = self.forward(features)  # [batch_size, list_size]

        # Apply mask
        scores = scores.masked_fill(mask == 0, float('-inf'))

        batch_size, list_size = scores.size()

        # Get ranking based on labels (descending order)
        _, ranking = torch.sort(labels, dim=-1, descending=True)

        # Compute ListMLE loss
        loss = 0.0
        for i in range(list_size):
            # Get scores for remaining documents at position i
            if i == 0:
                remaining_scores = scores
            else:
                # Mask out already selected documents
                selected_mask = torch.zeros_like(mask)
                for j in range(i):
                    selected_idx = ranking[:, j]
                    selected_mask[torch.arange(batch_size), selected_idx] = 1
                remaining_scores = scores.masked_fill(selected_mask == 1, float('-inf'))

            # Get the score of the document at position i in the ground truth ranking
            target_idx = ranking[:, i]
            target_scores = remaining_scores[torch.arange(batch_size), target_idx]

            # Compute log probability
            log_prob = target_scores - torch.logsumexp(remaining_scores, dim=-1)

            # Filter out invalid positions (where mask is 0)
            valid = mask[torch.arange(batch_size), target_idx]
            log_prob = log_prob * valid

            loss = loss - log_prob

        # Average over batch and list
        loss = loss.sum() / mask.sum()

        return loss


class ApproxNDCG(BaseRankingModel):
    """Approximate NDCG loss for direct optimization.

    Paper: Learning to Rank with Nonsmooth Cost Functions (Taylor et al., 2008)
    """

    def __init__(
        self,
        num_features: int,
        hidden_dims: List[int] = [256, 128, 64],
        dropout: float = 0.2,
        activation: str = 'relu',
        temperature: float = 1.0
    ):
        """Initialize ApproxNDCG model.

        Args:
            num_features: Number of input features
            hidden_dims: List of hidden layer dimensions
            dropout: Dropout probability
            activation: Activation function
            temperature: Temperature for softmax approximation
        """
        super().__init__(num_features, hidden_dims, dropout, activation)
        self.temperature = temperature

    def compute_dcg(
        self,
        scores: torch.Tensor,
        labels: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """Compute approximate DCG using soft sorting.

        Args:
            scores: Predicted scores [batch_size, list_size]
            labels: Relevance labels [batch_size, list_size]
            mask: Valid document mask [batch_size, list_size]

        Returns:
            DCG value [batch_size]
        """
        batch_size, list_size = scores.size()

        # Compute soft ranks using temperature-scaled softmax
        # Higher scores get lower ranks (positions)
        positions = torch.arange(1, list_size + 1, dtype=torch.float32, device=scores.device)
        positions = positions.unsqueeze(0).expand(batch_size, list_size)

        # Soft ranking: use attention-like mechanism
        score_diff = scores.unsqueeze(2) - scores.unsqueeze(1)  # [batch_size, list_size, list_size]
        soft_ranks = torch.sigmoid(score_diff / self.temperature).sum(dim=-1)  # [batch_size, list_size]

        # Compute gains
        gains = (2 ** labels - 1) * mask

        # Compute discounts
        discounts = 1.0 / torch.log2(soft_ranks + 2.0)  # +2 to match 1-indexed positions

        # DCG
        dcg = (gains * discounts).sum(dim=-1)

        return dcg

    def compute_loss(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """Compute approximate NDCG loss.

        Args:
            features: Document features [batch_size, list_size, num_features]
            labels: Relevance labels [batch_size, list_size]
            mask: Valid document mask [batch_size, list_size]

        Returns:
            Loss value (negative NDCG)
        """
        # Get predicted scores
        scores = self.forward(features)

        # Compute DCG
        dcg = self.compute_dcg(scores, labels, mask)

        # Compute IDCG (ideal DCG)
        sorted_labels, _ = torch.sort(labels, dim=-1, descending=True)
        ideal_scores, _ = torch.sort(scores, dim=-1, descending=True)
        idcg = self.compute_dcg(ideal_scores, sorted_labels, mask)

        # NDCG
        ndcg = dcg / (idcg + 1e-10)

        # Loss is negative NDCG
        loss = -ndcg.mean()

        return loss
