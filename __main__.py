import matplotlib

from core.utils import parser
from train import train

matplotlib.use('Agg')


def main():
    args = parser.get_parser().parse_args()
    args = vars(args)  # Converting argparse Namespace to a dict.

    for _ in train(args):  # `train` is a generator in order to be used with hyperfind.
        pass


if __name__ == "__main__":
    main()
    # import torch
    # print(torch.__version__)
