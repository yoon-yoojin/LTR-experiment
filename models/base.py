"""Base model class for LTR models."""
import torch
import torch.nn as nn
from typing import List


class BaseRankingModel(nn.Module):
    """Base class for ranking models."""

    def __init__(
        self,
        num_features: int,
        hidden_dims: List[int] = [256, 128, 64],
        dropout: float = 0.2,
        activation: str = 'relu'
    ):
        """Initialize base ranking model.

        Args:
            num_features: Number of input features
            hidden_dims: List of hidden layer dimensions
            dropout: Dropout probability
            activation: Activation function ('relu', 'tanh', 'elu')
        """
        super().__init__()
        self.num_features = num_features
        self.hidden_dims = hidden_dims
        self.dropout = dropout

        # Get activation function
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'elu':
            self.activation = nn.ELU()
        else:
            raise ValueError(f"Unknown activation: {activation}")

        # Build network layers
        self.layers = self._build_layers()

    def _build_layers(self) -> nn.ModuleList:
        """Build network layers.

        Returns:
            ModuleList of layers
        """
        layers = nn.ModuleList()
        input_dim = self.num_features

        for hidden_dim in self.hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(self.activation)
            layers.append(nn.Dropout(self.dropout))
            input_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(input_dim, 1))

        return layers

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor [batch_size, num_features] or [batch_size, list_size, num_features]

        Returns:
            Output scores
        """
        original_shape = x.shape
        if len(original_shape) == 3:
            # Reshape for batch processing [batch_size * list_size, num_features]
            batch_size, list_size, num_features = original_shape
            x = x.view(-1, num_features)

        # Pass through layers
        for layer in self.layers:
            x = layer(x)

        if len(original_shape) == 3:
            # Reshape back to [batch_size, list_size]
            x = x.view(batch_size, list_size)
        else:
            # [batch_size, 1] -> [batch_size]
            x = x.squeeze(-1)

        return x

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Predict scores for input.

        Args:
            x: Input tensor

        Returns:
            Predicted scores
        """
        self.eval()
        with torch.no_grad():
            scores = self.forward(x)
        return scores
