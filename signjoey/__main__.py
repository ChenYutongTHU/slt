import argparse
import os

import sys
from signjoey.training import train
from signjoey.prediction import test

sys.path.append("/vol/research/extol/personal/cihan/code/SignJoey")


def main():
    ap = argparse.ArgumentParser("Joey NMT")

    ap.add_argument("mode", choices=["train", "test"], help="train a model or test")

    ap.add_argument("config_path", type=str, help="path to YAML config file")

    ap.add_argument("--ckpt", type=str, help="checkpoint for prediction")

    ap.add_argument(
        "--output_path", type=str, help="path for saving translation output"
    )
    args = ap.parse_args()


    if args.mode == "train":
        train(cfg_file=args.config_path)
    elif args.mode == "test":
        from signjoey.helpers import make_logger, make_model_dir
        make_model_dir(args.output_path, overwrite=True)
        test_logger = make_logger(model_dir=args.output_path)
        test(cfg_file=args.config_path, ckpt=args.ckpt, output_path=args.output_path, logger=test_logger)
    else:
        raise ValueError("Unknown mode")


if __name__ == "__main__":
    main()
