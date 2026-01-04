"""Generate sample LETOR format data for testing."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import argparse
from pathlib import Path


def generate_letor_data(
    num_queries: int = 100,
    num_docs_per_query: int = 50,
    num_features: int = 136,
    output_path: str = 'data/raw/sample.txt'
):
    """Generate synthetic LETOR format data.

    Args:
        num_queries: Number of queries to generate
        num_docs_per_query: Average number of documents per query
        num_features: Number of features
        output_path: Output file path
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f'Generating {num_queries} queries with ~{num_docs_per_query} docs each...')

    with open(output_path, 'w') as f:
        for qid in range(1, num_queries + 1):
            # Vary number of documents per query
            n_docs = np.random.randint(
                max(10, num_docs_per_query - 20),
                num_docs_per_query + 20
            )

            for _ in range(n_docs):
                # Generate relevance label (0-4 for MSLR-like data)
                label = np.random.choice([0, 1, 2, 3, 4], p=[0.4, 0.3, 0.15, 0.1, 0.05])

                # Generate features
                # Make features somewhat correlated with label
                features = np.random.randn(num_features) + label * 0.5

                # Write in LETOR format
                line = f"{label} qid:{qid}"
                for feat_id in range(1, num_features + 1):
                    line += f" {feat_id}:{features[feat_id-1]:.6f}"
                line += "\n"

                f.write(line)

    print(f'Data saved to {output_path}')
    print(f'Total lines: {num_queries * num_docs_per_query}')


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Generate sample LETOR data')
    parser.add_argument('--num_queries', type=int, default=100, help='Number of queries')
    parser.add_argument('--num_docs', type=int, default=50, help='Docs per query (avg)')
    parser.add_argument('--num_features', type=int, default=136, help='Number of features')
    parser.add_argument('--output', type=str, default='data/raw/sample.txt', help='Output path')
    args = parser.parse_args()

    generate_letor_data(
        num_queries=args.num_queries,
        num_docs_per_query=args.num_docs,
        num_features=args.num_features,
        output_path=args.output
    )


if __name__ == '__main__':
    main()
