import argparse

from .runner import run


def main():
    ap = argparse.ArgumentParser(description="Culture eval: score ground-truth answers")
    ap.add_argument("--config", required=True, help="Path to culture eval config JSON")
    args = ap.parse_args()
    run(args.config, mode="ground_truth")


if __name__ == "__main__":
    main()
