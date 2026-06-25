import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

from scipy.stats import binomtest, wilcoxon


DATASET_ORDER: List[Tuple[str, str]] = [
    ("aya", "Aya"),
    ("blend", "BLEnD"),
    ("global_piqa", "Global-PIQA-OE"),
    ("mkqa", "MKQA"),
    ("mmlu", "Global MMLU"),
    ("belebele", "Belebele"),
    ("global-piqa-mc", "PIQA"),
    ("mcsqa", "MCSQA"),
    ("mgsm", "MGSM"),
    ("mmath", "MMath"),
]


@dataclass
class DatasetSigResult:
    dataset_key: str
    dataset_label: str
    metric: str
    n_languages: int
    wins: int
    losses: int
    ties: int
    non_tied: int
    win_rate: float
    mean_delta: float
    median_delta: float
    binomial_p_two_sided: float
    wilcoxon_statistic: float
    wilcoxon_p_two_sided: float
    binomial_q_bh: float = math.nan
    wilcoxon_q_bh: float = math.nan


def _metric_name(by_language: Dict[str, Dict]) -> str:
    sample = next(iter(by_language.values()))
    for preferred in ("chrf", "accuracy", "f1"):
        if preferred in sample:
            return preferred
    for key in sample.keys():
        if key not in {"count", "resource_level"}:
            return key
    raise ValueError("Could not determine metric field")


def _load_language_scores(path: Path) -> Tuple[str, Dict[str, float], Dict[str, float]]:
    obj = json.loads(path.read_text())
    std = obj["slices"]["baseline/standard"]["by_language"]
    ctx = obj["slices"]["baseline/context"]["by_language"]
    metric = _metric_name(std)
    std_scores = {lang: float(std[lang][metric]) for lang in std}
    ctx_scores = {lang: float(ctx[lang][metric]) for lang in ctx}
    return metric, std_scores, ctx_scores


def _median(values: List[float]) -> float:
    if not values:
        return math.nan
    vals = sorted(values)
    n = len(vals)
    mid = n // 2
    if n % 2 == 1:
        return vals[mid]
    return (vals[mid - 1] + vals[mid]) / 2.0


def _bh_adjust(pvals: List[float]) -> List[float]:
    n = len(pvals)
    indexed = sorted(enumerate(pvals), key=lambda x: (math.inf if math.isnan(x[1]) else x[1]))
    out = [math.nan] * n
    prev = 1.0
    rank = 0
    finite = [(i, p) for i, p in indexed if not math.isnan(p)]
    m = len(finite)
    for rev_rank, (idx, p) in enumerate(reversed(finite), start=1):
        k = m - rev_rank + 1
        q = min(prev, p * m / k)
        out[idx] = q
        prev = q
    return out


def default_metrics_path(repo_root: Path, model: str, dataset_key: str) -> Path:
    return repo_root / "results" / "final" / model / dataset_key / "metrics" / "metrics.json"


def compute_ctx_std_tests_for_model(
    repo_root: Path,
    model: str,
    metrics_path_resolver: Optional[Callable[[Path, str, str], Path]] = None,
) -> List[DatasetSigResult]:
    if metrics_path_resolver is None:
        metrics_path_resolver = default_metrics_path

    results: List[DatasetSigResult] = []
    for dataset_key, dataset_label in DATASET_ORDER:
        metrics_path = metrics_path_resolver(repo_root, model, dataset_key)
        if not metrics_path.exists():
            continue
        metric, std_scores, ctx_scores = _load_language_scores(metrics_path)
        langs = sorted(set(std_scores) & set(ctx_scores))
        deltas = [ctx_scores[lang] - std_scores[lang] for lang in langs]
        wins = sum(1 for d in deltas if d > 0)
        losses = sum(1 for d in deltas if d < 0)
        ties = sum(1 for d in deltas if d == 0)
        non_tied = wins + losses
        nonzero_deltas = [d for d in deltas if d != 0]

        if non_tied > 0:
            binom_p = float(binomtest(wins, non_tied, p=0.5, alternative="two-sided").pvalue)
        else:
            binom_p = math.nan

        if nonzero_deltas:
            wil = wilcoxon(nonzero_deltas, zero_method="wilcox", alternative="two-sided", method="auto")
            wil_stat = float(wil.statistic)
            wil_p = float(wil.pvalue)
        else:
            wil_stat = math.nan
            wil_p = math.nan

        results.append(
            DatasetSigResult(
                dataset_key=dataset_key,
                dataset_label=dataset_label,
                metric=metric,
                n_languages=len(langs),
                wins=wins,
                losses=losses,
                ties=ties,
                non_tied=non_tied,
                win_rate=(wins / non_tied) if non_tied else math.nan,
                mean_delta=sum(deltas) / len(deltas) if deltas else math.nan,
                median_delta=_median(deltas),
                binomial_p_two_sided=binom_p,
                wilcoxon_statistic=wil_stat,
                wilcoxon_p_two_sided=wil_p,
            )
        )

    binom_q = _bh_adjust([r.binomial_p_two_sided for r in results])
    wil_q = _bh_adjust([r.wilcoxon_p_two_sided for r in results])
    for result, bq, wq in zip(results, binom_q, wil_q):
        result.binomial_q_bh = bq
        result.wilcoxon_q_bh = wq
    return results


def _format_float(value: float) -> str:
    if value is None or math.isnan(value):
        return "NaN"
    return f"{value:.6g}"


def write_results_json(path: Path, rows: Iterable[DatasetSigResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [asdict(row) for row in rows]
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_results_csv(path: Path, rows: Iterable[DatasetSigResult]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(asdict(rows[0]).keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def write_results_markdown(path: Path, rows: Iterable[DatasetSigResult]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = [
        "Dataset",
        "Metric",
        "n",
        "Wins",
        "Losses",
        "Ties",
        "Win rate",
        "Mean Δ",
        "Median Δ",
        "Binom p",
        "Binom q",
        "Wilcoxon W",
        "Wilcoxon p",
        "Wilcoxon q",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    row.dataset_label,
                    row.metric,
                    str(row.n_languages),
                    str(row.wins),
                    str(row.losses),
                    str(row.ties),
                    _format_float(row.win_rate),
                    _format_float(row.mean_delta),
                    _format_float(row.median_delta),
                    _format_float(row.binomial_p_two_sided),
                    _format_float(row.binomial_q_bh),
                    _format_float(row.wilcoxon_statistic),
                    _format_float(row.wilcoxon_p_two_sided),
                    _format_float(row.wilcoxon_q_bh),
                ]
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
