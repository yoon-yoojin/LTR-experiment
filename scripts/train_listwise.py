"""Training script for listwise LTR models."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from torch.utils.data import DataLoader
import argparse

from utils.config import Config
from utils.logger import setup_logger
from data.dataset import LTRDataset, ListwiseDataset, collate_fn_listwise
from models.listwise import ListNet, ListMLE, ApproxNDCG
from training.listwise_trainer import ListwiseTrainer


def main():
    """Main training function for listwise models."""
    parser = argparse.ArgumentParser(description='Train listwise LTR model')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    parser.add_argument('--model', type=str, default='listnet',
                        choices=['listnet', 'listmle', 'approxndcg'],
                        help='Model type')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use')
    parser.add_argument('--train_file', type=str, help='Path to training data (overrides config)')
    parser.add_argument('--val_file', type=str, help='Path to validation data (overrides config)')
    args = parser.parse_args()

    # Load config
    config = Config(args.config)
    logger = setup_logger('train_listwise', config.get('logging', 'log_dir'))

    logger.info('=' * 80)
    logger.info('Training Listwise LTR Model')
    logger.info('=' * 80)
    logger.info(f'Model: {args.model}')
    logger.info(f'Device: {args.device}')

    # Load data
    train_file = args.train_file or str(Path(config.get('data', 'raw_data_path')) / config.get('data', 'train_file'))
    val_file = args.val_file or str(Path(config.get('data', 'raw_data_path')) / config.get('data', 'val_file'))

    logger.info(f'Loading training data from {train_file}')
    train_base_dataset = LTRDataset(train_file)
    logger.info(f'Loaded {len(train_base_dataset)} training queries')

    logger.info(f'Loading validation data from {val_file}')
    val_base_dataset = LTRDataset(val_file)
    logger.info(f'Loaded {len(val_base_dataset)} validation queries')

    # Create listwise datasets
    list_size = config.get('training', 'list_size', 10)
    train_dataset = ListwiseDataset(train_base_dataset, list_size=list_size)

    # Create data loaders
    batch_size = config.get('training', 'batch_size', 32)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn_listwise,
        num_workers=4
    )

    # Create model
    num_features = train_base_dataset.num_features
    model_config = config.get('model', 'listwise')

    if args.model == 'listnet':
        model = ListNet(
            num_features=num_features,
            hidden_dims=model_config.get('hidden_dims', [256, 128, 64]),
            dropout=model_config.get('dropout', 0.2),
            activation=model_config.get('activation', 'relu')
        )
    elif args.model == 'listmle':
        model = ListMLE(
            num_features=num_features,
            hidden_dims=model_config.get('hidden_dims', [256, 128, 64]),
            dropout=model_config.get('dropout', 0.2),
            activation=model_config.get('activation', 'relu')
        )
    elif args.model == 'approxndcg':
        model = ApproxNDCG(
            num_features=num_features,
            hidden_dims=model_config.get('hidden_dims', [256, 128, 64]),
            dropout=model_config.get('dropout', 0.2),
            activation=model_config.get('activation', 'relu')
        )
    else:
        raise ValueError(f"Unknown model: {args.model}")

    logger.info(f'Created {args.model} model with {sum(p.numel() for p in model.parameters())} parameters')

    # Create trainer
    trainer = ListwiseTrainer(
        model=model,
        train_loader=train_loader,
        val_dataset=val_base_dataset,
        config=config,
        device=args.device
    )

    # Train
    history = trainer.train()

    logger.info('Training finished!')
    logger.info(f'Best NDCG@10: {trainer.best_metric:.4f}')


if __name__ == '__main__':
    main()
