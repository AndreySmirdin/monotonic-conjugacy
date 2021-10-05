import argparse
import pandas as pd
import numpy as np
from typing import Tuple, List
from scipy.stats import rankdata


def read_input_file(filename: str) -> Tuple[List[int], List[int]]:
    data = pd.read_csv(filename, delimiter=' ', header=None)
    return data[0].values, data[1].values


def write_output(filename: str, delta: float, sigma: float, conjugacy: float):
    with open(filename, 'w') as f:
        f.write("{:.2f}".format(delta))
        f.write(' ')
        f.write("{:.2f}".format(sigma))
        f.write(' ')
        f.write("{:.2f}".format(conjugacy))


def check_conjugacy(y: List[str]) -> Tuple[float, float, float]:
    N = len(y)
    p = int(N / 3)
    ranks = (N + 1) - rankdata(y)
    R1 = ranks[:p].sum()
    R2 = ranks[-p:].sum()
    sigma = (N + 0.5) * np.sqrt(p / 6)
    conjugacy = (R1 - R2) / (p * (N - p))
    return R1 - R2, sigma, conjugacy


def main():
    parser = argparse.ArgumentParser(description='Calculate monotonic conjugancy.')
    parser.add_argument('--input', required=True, help='input file')
    parser.add_argument('--output', required=True, help='output file')
    args = parser.parse_args()

    try:
        x, y = read_input_file(args.input)
    except Exception as e:
        print(f"Could not read the input file: {e}")
        return

    if len(x) < 3:
        print("Amount of data is too small, impossible to apply the method.")
        return

    sorted_indices = np.argsort(x)
    y = y[sorted_indices]

    delta, sigma, conjugacy = check_conjugacy(y)
    try:
        write_output(args.output, delta, sigma, conjugacy)
    except Exception as e:
        print(f"Could not save the result: {e}")
        return


if __name__ == '__main__':
    main()
