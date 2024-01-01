import matplotlib

from core.utils import parser
from train import train

matplotlib.use('Agg')


def main():
    args = parser.get_parser().parse_args()
    args = vars(args)  # Converting argparse Namespace to a dict.
    if args["seed_range"] is not None:
        print("haha")
        args["seed"] = list(range(args["seed_range"][0], args["seed_range"][1] + 1))
        print("Seed range", args["seed"])

    for _ in train(args):  # `train` is a generator in order to be used with hyperfind.
        pass


if __name__ == "__main__":
    main()
    # import torch
    # print(torch.__version__)
