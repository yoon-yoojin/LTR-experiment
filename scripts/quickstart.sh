#!/bin/bash
# Quick Start Script for Learning to Rank Project

set -e  # Exit on error

echo "========================================="
echo "Learning to Rank - Quick Start"
echo "========================================="
echo ""

# Create necessary directories
echo "Creating directories..."
mkdir -p data/raw data/processed checkpoints logs results

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install -r requirements.txt

# Generate sample data
echo ""
echo "Generating sample training data..."
python scripts/generate_sample_data.py \
    --num_queries 1000 \
    --num_docs 50 \
    --num_features 136 \
    --output data/raw/train.txt

echo ""
echo "Generating sample validation data..."
python scripts/generate_sample_data.py \
    --num_queries 200 \
    --num_docs 50 \
    --num_features 136 \
    --output data/raw/val.txt

echo ""
echo "Generating sample test data..."
python scripts/generate_sample_data.py \
    --num_queries 200 \
    --num_docs 50 \
    --num_features 136 \
    --output data/raw/test.txt

echo ""
echo "========================================="
echo "Setup Complete!"
echo "========================================="
echo ""
echo "Next steps:"
echo ""
echo "1. Train a pairwise model (RankNet):"
echo "   python scripts/train_pairwise.py --model ranknet --device cpu"
echo ""
echo "2. Train a listwise model (ListNet):"
echo "   python scripts/train_listwise.py --model listnet --device cpu"
echo ""
echo "3. Run inference:"
echo "   python scripts/inference.py \\"
echo "       --model_path checkpoints/best_model.pt \\"
echo "       --data_path data/raw/test.txt \\"
echo "       --output_dir results/predictions"
echo ""
echo "For GPU training, replace '--device cpu' with '--device cuda'"
echo ""
