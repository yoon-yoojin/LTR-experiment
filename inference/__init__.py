"""Inference modules for LTR models."""
from .predictor import LTRPredictor, load_and_predict

__all__ = [
    'LTRPredictor',
    'load_and_predict',
]
