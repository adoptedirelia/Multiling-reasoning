import argparse
import logging

from ..cascade.runner import run_cascade_predictions
from ..config import load_config
from ..corruptions.builder import build_corruptions
from ..metrics.runner import run_metrics

LOGGER = logging.getLogger(__name__)


def run_pipeline(config_path: str, steps_csv: str = "build,predict,eval"):
    logging.basicConfig(level=logging.INFO)
    cfg = load_config(config_path)
    steps = [s.strip().lower() for s in steps_csv.split(",") if s.strip()]
    if not steps:
        steps = ["build", "predict", "eval"]
    LOGGER.info("pipeline start: run_name=%s dataset=%s steps=%s", cfg.run_name, cfg.dataset.dataset_type, steps)

    pred_path = cfg.outputs.predictions_jsonl
    if "build" in steps:
        LOGGER.info("pipeline step=build start")
        build_corruptions(config_path)
        LOGGER.info("pipeline step=build done")
    if "predict" in steps:
        LOGGER.info("pipeline step=predict start")
        pred_path = run_cascade_predictions(
            config_path=config_path,
            corruption_jsonl=cfg.corruption.input_jsonl or cfg.corruption.output_jsonl or cfg.outputs.corruption_jsonl,
            out_jsonl=cfg.outputs.predictions_jsonl,
        )
        LOGGER.info("pipeline step=predict done pred_path=%s", pred_path)
    if "eval" in steps:
        LOGGER.info("pipeline step=eval start")
        run_metrics(
            config_path=config_path,
            predictions_jsonl=pred_path,
            out_json=cfg.outputs.metrics_json,
        )
        LOGGER.info("pipeline step=eval done out=%s", cfg.outputs.metrics_json)
    LOGGER.info("pipeline done")


def main():
    ap = argparse.ArgumentParser(description="V2 pipeline wrapper")
    ap.add_argument("--config", required=True)
    ap.add_argument("--steps", default="build,predict,eval")
    args = ap.parse_args()
    run_pipeline(args.config, args.steps)


if __name__ == "__main__":
    main()
