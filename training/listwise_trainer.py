"""Trainer for listwise LTR models."""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
import json
from typing import Dict, Optional, Any

from evaluation.metrics import evaluate_model
from utils.logger import setup_logger, MetricLogger
from .trainer import EarlyStopping


class ListwiseTrainer:
    """Trainer for listwise LTR models."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_dataset: Any,
        config: Dict,
        device: str = 'cpu'
    ):
        """Initialize listwise trainer.

        Args:
            model: LTR model
            train_loader: Training data loader
            val_dataset: Validation dataset (LTRDataset)
            config: Configuration dictionary
            device: Device to use
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_dataset = val_dataset
        self.config = config
        self.device = device

        # Setup optimizer
        self.optimizer = self._setup_optimizer()
        self.scheduler = self._setup_scheduler()

        # Setup logging
        self.logger = setup_logger('ListwiseTrainer', config.get('logging', 'log_dir'))
        self.metric_logger = MetricLogger(
            config.get('logging', 'log_dir'),
            config.get('logging', 'tensorboard', True)
        )

        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=config.get('training', 'early_stopping_patience', 10),
            mode='max'
        )

        # Checkpointing
        self.checkpoint_dir = Path(config.get('logging', 'checkpoint_dir'))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.best_metric = 0.0
        self.global_step = 0

    def _setup_optimizer(self) -> torch.optim.Optimizer:
        """Setup optimizer."""
        lr = self.config.get('training', 'learning_rate', 0.001)
        weight_decay = self.config.get('training', 'weight_decay', 0.0001)

        optimizer_name = self.config.get('training', 'optimizer', 'adam').lower()

        if optimizer_name == 'adam':
            return torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name == 'sgd':
            return torch.optim.SGD(self.model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")

    def _setup_scheduler(self) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """Setup learning rate scheduler."""
        scheduler_name = self.config.get('training', 'scheduler', 'cosine').lower()
        num_epochs = self.config.get('training', 'num_epochs', 50)

        if scheduler_name == 'cosine':
            return torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=num_epochs)
        elif scheduler_name == 'step':
            return torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.5)
        elif scheduler_name == 'none':
            return None
        else:
            raise ValueError(f"Unknown scheduler: {scheduler_name}")

    def train_epoch(self) -> float:
        """Train for one epoch.

        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch in tqdm(self.train_loader, desc='Training'):
            features, labels, mask = batch
            features = features.to(self.device)
            labels = labels.to(self.device)
            mask = mask.to(self.device)

            # Forward pass
            loss = self.model.compute_loss(features, labels, mask)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            grad_clip = self.config.get('training', 'gradient_clip', 5.0)
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)

            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1

            # Log metrics
            if self.global_step % 100 == 0:
                self.metric_logger.log_scalar('train/loss', loss.item(), self.global_step)

        return total_loss / num_batches if num_batches > 0 else 0.0

    def validate(self) -> Dict[str, float]:
        """Validate model.

        Returns:
            Dictionary of validation metrics
        """
        k_values = self.config.get('evaluation', 'k_values', [1, 3, 5, 10])
        metrics = evaluate_model(self.model, self.val_dataset, self.device, k_values)
        return metrics

    def save_checkpoint(self, epoch: int, metrics: Dict[str, float], is_best: bool = False) -> None:
        """Save model checkpoint.

        Args:
            epoch: Current epoch
            metrics: Validation metrics
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'config': self.config.config if hasattr(self.config, 'config') else self.config
        }

        # Save regular checkpoint
        checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pt'
        torch.save(checkpoint, checkpoint_path)

        # Save best model
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)
            self.logger.info(f'Saved best model with NDCG@10: {metrics.get("ndcg@10", 0.0):.4f}')

    def train(self) -> Dict[str, Any]:
        """Train model.

        Returns:
            Training history
        """
        num_epochs = self.config.get('training', 'num_epochs', 50)
        save_frequency = self.config.get('logging', 'save_frequency', 5)

        history = {
            'train_loss': [],
            'val_metrics': []
        }

        self.logger.info('Starting training...')
        self.logger.info(f'Device: {self.device}')
        self.logger.info(f'Number of epochs: {num_epochs}')

        for epoch in range(1, num_epochs + 1):
            self.logger.info(f'\nEpoch {epoch}/{num_epochs}')

            # Train
            train_loss = self.train_epoch()
            history['train_loss'].append(train_loss)
            self.logger.info(f'Training loss: {train_loss:.4f}')

            # Validate
            val_metrics = self.validate()
            history['val_metrics'].append(val_metrics)

            # Log metrics
            self.logger.info('Validation metrics:')
            for metric_name, metric_value in val_metrics.items():
                self.logger.info(f'  {metric_name}: {metric_value:.4f}')
                self.metric_logger.log_scalar(f'val/{metric_name}', metric_value, epoch)

            # Check for improvement
            current_metric = val_metrics.get('ndcg@10', 0.0)
            is_best = current_metric > self.best_metric
            if is_best:
                self.best_metric = current_metric

            # Save checkpoint
            if epoch % save_frequency == 0 or is_best:
                self.save_checkpoint(epoch, val_metrics, is_best)

            # Learning rate scheduling
            if self.scheduler is not None:
                self.scheduler.step()
                current_lr = self.optimizer.param_groups[0]['lr']
                self.metric_logger.log_scalar('train/lr', current_lr, epoch)

            # Early stopping
            if self.early_stopping(current_metric):
                self.logger.info(f'Early stopping triggered at epoch {epoch}')
                break

        # Save final results
        results_dir = Path(self.config.get('logging', 'result_dir'))
        results_dir.mkdir(parents=True, exist_ok=True)

        with open(results_dir / 'training_history.json', 'w') as f:
            # Convert numpy types to Python types for JSON serialization
            history_serializable = {
                'train_loss': [float(x) for x in history['train_loss']],
                'val_metrics': [
                    {k: float(v) for k, v in metrics.items()}
                    for metrics in history['val_metrics']
                ]
            }
            json.dump(history_serializable, f, indent=2)

        self.metric_logger.close()
        self.logger.info('Training completed!')
        self.logger.info(f'Best NDCG@10: {self.best_metric:.4f}')

        return history
