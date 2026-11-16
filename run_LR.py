"""HW07 CSV loading and experiment driver. Author: course staff."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from LogisticRegression import FloatArray, IntArray
import LogisticRegression as lr


def parse_csv(filename: str | Path) -> tuple[FloatArray, IntArray]:
    """Read features and labels from a headerless CSV; the last column is the label."""
    features: list[list[float]] = []
    labels: list[int] = []
    with open(filename) as csv_file:
        for line in csv_file:
            values = line.strip().split(",")
            features.append([float(value) for value in values[:-1]])
            labels.append(int(values[-1]))
    return np.array(features, dtype=float), np.array(labels, dtype=np.int64)

def print_confusion_matrix(matrix: IntArray) -> None:
    """Provided presentation helper; computation belongs in confusion_matrix()."""
    print("Actual rows; predicted columns")
    print("     " + "".join(f"{k:6d}" for k in range(len(matrix))))
    for k, row in enumerate(matrix):
        print(f"{k:3d} |" + "".join(f"{value:6d}" for value in row))


def confusion_matrix(y: IntArray, predicted: IntArray) -> IntArray:
    """Return a 2x2 count matrix: rows=true labels, columns=predicted labels."""
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    for i in range(len(y)):
        if y[i] == 1:
            if predicted[i] == 1:
                TP += 1
            else:
                FN += 1
        if y[i] == 0:
            if predicted[i] == 1:
                FP += 1
            else:
                TN += 1
    m = np.array([[TN, FP], [FN, TP]])
    print(m)
    print_confusion_matrix(m)


def parse_args() -> argparse.Namespace:
    """Parse training/test CSV paths and experiment controls."""
    parser = argparse.ArgumentParser(description="HW07 binary logistic regression")
    parser.add_argument("-r", "--train_filename", required=True)
    parser.add_argument("-e", "--test_filename", required=True)
    parser.add_argument("-a", "--alpha", type=float, default=0.02)
    parser.add_argument("--eps", type=float, default=1e-4)
    parser.add_argument("--tmax", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--compare", action="store_true", help="Run your sklearn comparison (to be implemented)")
    parser.add_argument("--plot", type=Path, help="Save a one-feature Challenger plot as PDF")
    args = parser.parse_args()
    if not np.isfinite(args.alpha) or args.alpha <= 0:
        parser.error("alpha must be finite and greater than zero")
    if not np.isfinite(args.eps) or args.eps < 0:
        parser.error("eps must be finite and nonnegative")
    if args.tmax < 1:
        parser.error("tmax must be at least 1")
    return args


def main() -> None:
    args = parse_args()
    X, y = parse_csv(args.train_filename)
    test_X, test_y = parse_csv(args.test_filename)
    if X.shape[1] != test_X.shape[1]:
        raise ValueError("Train/test feature counts differ")
    coef = lr.fit_SGD(X, y, args.alpha, args.eps, args.tmax, args.seed)
    probabilities = lr.prediction(test_X, coef)
    labels = lr.threshold(probabilities)
    print("Coefficients (bias first):", coef)
    print("Test probabilities:", probabilities)
    print(f"Accuracy: {np.mean(labels == test_y):.6f}")
    print("Confusion matrix (rows=true, columns=predicted):")
    print(confusion_matrix(test_y, labels))
    if args.compare:
        # TODO: train and evaluate sklearn's LogisticRegression on both datasets.
        # Report accuracy, confusion matrix, and positive-class probabilities.
        raise NotImplementedError("Add the sklearn comparison")
    if args.plot:
        plot_challenger(X, y, test_X, test_y, coef, args.plot)


def plot_challenger(X: FloatArray, y: IntArray, test_X: FloatArray,
                    test_y: IntArray, coef: FloatArray, output: Path) -> None:
    """Save labeled training/test observations and fitted sigmoid; require one feature.

    Use a sorted grid spanning both splits, probability on the vertical axis,
    temperature (degrees F) on the horizontal axis, title, legend and distinct markers.
    Save the figure to output and close it; do not call show().
    """
    # TODO: implement the contract above.
    raise NotImplementedError("Implement plot_challenger")


if __name__ == "__main__":
    main()
