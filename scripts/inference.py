"""Inference script for LTR models."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
from inference.predictor import load_and_predict


def main():
    """Main inference function."""
    parser = argparse.ArgumentParser(description='Run inference with trained LTR model')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model checkpoint')
    parser.add_argument('--data_path', type=str, required=True, help='Path to test data (LETOR format)')
    parser.add_argument('--output_dir', type=str, default='results/predictions', help='Output directory')
    parser.add_argument('--preprocessor_path', type=str, help='Path to preprocessor')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use')
    parser.add_argument('--top_k', type=int, help='Return only top k documents per query')
    args = parser.parse_args()

    print('=' * 80)
    print('Running LTR Model Inference')
    print('=' * 80)
    print(f'Model: {args.model_path}')
    print(f'Data: {args.data_path}')
    print(f'Output: {args.output_dir}')
    print(f'Device: {args.device}')

    # Run inference
    load_and_predict(
        model_path=args.model_path,
        data_path=args.data_path,
        output_dir=args.output_dir,
        preprocessor_path=args.preprocessor_path,
        device=args.device,
        top_k=args.top_k
    )

    print('Inference completed!')


if __name__ == '__main__':
    main()
