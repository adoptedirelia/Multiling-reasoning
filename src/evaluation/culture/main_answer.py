import argparse

from .runner import run


def main():
    ap = argparse.ArgumentParser(description="Culture eval: score mt2 answers")
    ap.add_argument("--config", required=True, help="Path to culture eval config JSON")
    args = ap.parse_args()
    run(args.config, mode="answer")


if __name__ == "__main__":
    main()
