import argparse
from pathlib import Path

from tomato.utils.interpretability import plot_layer_confidence_from_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot layer confidence heatmap from CSV")
    parser.add_argument("csv", type=Path, help="Path to layer_confidence.csv")
    parser.add_argument("--output", type=Path, default=None, help="Optional output PNG path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    csv_path = args.csv
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file {csv_path} not found")
    output_path = plot_layer_confidence_from_csv(csv_path, args.output)
    print(f"Heatmap saved to {output_path}")


if __name__ == "__main__":
    main()
