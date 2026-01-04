"""Inference pipeline for LTR models."""
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
import json

from data.dataset import LTRDataset
from data.preprocessing import LTRPreprocessor
from utils.logger import setup_logger


class LTRPredictor:
    """Predictor for LTR models."""

    def __init__(
        self,
        model_path: str,
        device: str = 'cpu',
        preprocessor_path: Optional[str] = None
    ):
        """Initialize LTR predictor.

        Args:
            model_path: Path to saved model checkpoint
            device: Device to use for inference
            preprocessor_path: Path to saved preprocessor (optional)
        """
        self.device = device
        self.logger = setup_logger('LTRPredictor')

        # Load model
        self.model, self.config = self._load_model(model_path)
        self.model = self.model.to(device)
        self.model.eval()

        # Load preprocessor if provided
        self.preprocessor = None
        if preprocessor_path and Path(preprocessor_path).exists():
            self.preprocessor = LTRPreprocessor.load(preprocessor_path)
            self.logger.info(f'Loaded preprocessor from {preprocessor_path}')

        self.logger.info(f'Model loaded from {model_path}')
        self.logger.info(f'Device: {device}')

    def _load_model(self, model_path: str) -> Tuple[torch.nn.Module, Dict]:
        """Load model from checkpoint.

        Args:
            model_path: Path to model checkpoint

        Returns:
            Tuple of (model, config)
        """
        checkpoint = torch.load(model_path, map_location=self.device)
        config = checkpoint['config']

        # Determine model type from config
        if 'pairwise' in config.get('model', {}):
            model_config = config['model']['pairwise']
            model_type = model_config.get('name', 'ranknet')

            if model_type == 'ranknet':
                from models.pairwise import RankNet
                model = RankNet(
                    num_features=config['data']['num_features'],
                    hidden_dims=model_config.get('hidden_dims', [256, 128, 64]),
                    dropout=model_config.get('dropout', 0.2),
                    activation=model_config.get('activation', 'relu')
                )
            elif model_type == 'lambdarank':
                from models.pairwise import LambdaRank
                model = LambdaRank(
                    num_features=config['data']['num_features'],
                    hidden_dims=model_config.get('hidden_dims', [256, 128, 64]),
                    dropout=model_config.get('dropout', 0.2),
                    activation=model_config.get('activation', 'relu')
                )
            else:
                raise ValueError(f"Unknown pairwise model: {model_type}")

        elif 'listwise' in config.get('model', {}):
            model_config = config['model']['listwise']
            model_type = model_config.get('name', 'listnet')

            if model_type == 'listnet':
                from models.listwise import ListNet
                model = ListNet(
                    num_features=config['data']['num_features'],
                    hidden_dims=model_config.get('hidden_dims', [256, 128, 64]),
                    dropout=model_config.get('dropout', 0.2),
                    activation=model_config.get('activation', 'relu')
                )
            elif model_type == 'listmle':
                from models.listwise import ListMLE
                model = ListMLE(
                    num_features=config['data']['num_features'],
                    hidden_dims=model_config.get('hidden_dims', [256, 128, 64]),
                    dropout=model_config.get('dropout', 0.2),
                    activation=model_config.get('activation', 'relu')
                )
            elif model_type == 'approxndcg':
                from models.listwise import ApproxNDCG
                model = ApproxNDCG(
                    num_features=config['data']['num_features'],
                    hidden_dims=model_config.get('hidden_dims', [256, 128, 64]),
                    dropout=model_config.get('dropout', 0.2),
                    activation=model_config.get('activation', 'relu')
                )
            else:
                raise ValueError(f"Unknown listwise model: {model_type}")
        else:
            raise ValueError("Model type not specified in config")

        model.load_state_dict(checkpoint['model_state_dict'])

        return model, config

    def preprocess_features(self, features: np.ndarray) -> np.ndarray:
        """Preprocess features using loaded preprocessor.

        Args:
            features: Raw features

        Returns:
            Preprocessed features
        """
        if self.preprocessor is None:
            return features

        # Create temporary dict for preprocessing
        temp_dict = {'temp': features}
        processed_dict = self.preprocessor.transform(temp_dict)
        return processed_dict['temp']

    def predict_scores(self, features: np.ndarray) -> np.ndarray:
        """Predict relevance scores for documents.

        Args:
            features: Document features [num_docs, num_features]

        Returns:
            Predicted scores [num_docs]
        """
        # Preprocess
        features = self.preprocess_features(features)

        # Convert to tensor
        features_tensor = torch.tensor(features, dtype=torch.float32).to(self.device)

        # Predict
        with torch.no_grad():
            scores = self.model.predict(features_tensor)

        return scores.cpu().numpy()

    def rank_documents(
        self,
        features: np.ndarray,
        doc_ids: Optional[List[str]] = None,
        top_k: Optional[int] = None
    ) -> List[Tuple[Union[int, str], float]]:
        """Rank documents by predicted scores.

        Args:
            features: Document features [num_docs, num_features]
            doc_ids: Optional document IDs
            top_k: Return only top k documents (None for all)

        Returns:
            List of (doc_id, score) tuples sorted by score (descending)
        """
        scores = self.predict_scores(features)

        # Create doc IDs if not provided
        if doc_ids is None:
            doc_ids = list(range(len(scores)))

        # Rank documents
        ranking = sorted(
            zip(doc_ids, scores),
            key=lambda x: x[1],
            reverse=True
        )

        # Return top k
        if top_k is not None:
            ranking = ranking[:top_k]

        return ranking

    def predict_dataset(
        self,
        dataset: LTRDataset,
        output_dir: str,
        top_k: Optional[int] = None
    ) -> None:
        """Predict and save rankings for entire dataset.

        Args:
            dataset: LTRDataset to predict on
            output_dir: Directory to save predictions
            top_k: Return only top k documents per query
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info(f'Predicting on {len(dataset)} queries...')

        all_predictions = {}

        for qid in dataset.qids:
            features, labels = dataset.get_query_data(qid)

            # Predict scores
            scores = self.predict_scores(features)

            # Rank documents
            ranking = sorted(
                enumerate(scores),
                key=lambda x: x[1],
                reverse=True
            )

            if top_k is not None:
                ranking = ranking[:top_k]

            # Store predictions
            all_predictions[qid] = {
                'ranking': [(int(idx), float(score)) for idx, score in ranking],
                'ground_truth': labels.tolist()
            }

        # Save predictions
        output_path = output_dir / 'predictions.json'
        with open(output_path, 'w') as f:
            json.dump(all_predictions, f, indent=2)

        self.logger.info(f'Predictions saved to {output_path}')

    def predict_batch(
        self,
        features_list: List[np.ndarray],
        batch_size: int = 64
    ) -> List[np.ndarray]:
        """Predict scores for multiple queries in batches.

        Args:
            features_list: List of feature arrays for each query
            batch_size: Batch size for processing

        Returns:
            List of score arrays for each query
        """
        all_scores = []

        for i in range(0, len(features_list), batch_size):
            batch_features = features_list[i:i + batch_size]

            # Process each query in batch
            batch_scores = []
            for features in batch_features:
                scores = self.predict_scores(features)
                batch_scores.append(scores)

            all_scores.extend(batch_scores)

        return all_scores


def load_and_predict(
    model_path: str,
    data_path: str,
    output_dir: str,
    preprocessor_path: Optional[str] = None,
    device: str = 'cpu',
    top_k: Optional[int] = None
) -> None:
    """Convenience function to load model and predict on dataset.

    Args:
        model_path: Path to model checkpoint
        data_path: Path to data file (LETOR format)
        output_dir: Directory to save predictions
        preprocessor_path: Path to preprocessor (optional)
        device: Device to use
        top_k: Return only top k documents per query
    """
    # Load dataset
    dataset = LTRDataset(data_path)

    # Create predictor
    predictor = LTRPredictor(model_path, device, preprocessor_path)

    # Predict
    predictor.predict_dataset(dataset, output_dir, top_k)
