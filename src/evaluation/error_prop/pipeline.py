import logging
from typing import Dict, List, Optional

from .constraints import batch_constraint_accuracy, check_constraint


def _attach_constraint(text: str, constraint: str) -> str:
    if not constraint.strip():
        return text
    return f"{text}\n\n{constraint}"


def _run_mt2(
    mt2_component,
    *,
    q_l: List[str],
    q_en: List[str],
    y_en: List[str],
    reason_en: List[str],
    batch_size: int,
    max_new_tokens: int,
) -> List[str]:
    # Translator mode: reuse LLMTranslator directly.
    if hasattr(mt2_component, "batched_translate"):
        return mt2_component.batched_translate(
            y_en,
            batch_size=batch_size,
            max_new_tokens=max_new_tokens,
        )
    # Corrector mode: use richer context.
    return mt2_component.produce_batch(
        q_l=q_l,
        q_en=q_en,
        y_en=y_en,
        reason_en=reason_en,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
    )


def run_pipeline_constraints(
    *,
    dataset_name: str,
    examples: List[Dict],
    translator_l2en,
    translator_en2l,
    reasoner,
    mt2_component,
    lang_l: str,
    constraints: List[Dict],
    max_examples: Optional[int] = None,
    translation_batch_size: int = 16,
    translation_max_new_tokens: int = 128,
    reasoner_batch_size: int = 8,
    reasoner_max_new_tokens: int = 256,
    reasoner_temperature: float = 0.0,
    reasoner_top_p: float = 1.0,
    mt2_batch_size: int = 8,
    mt2_max_new_tokens: int = 256,
) -> Dict:
    if max_examples is not None:
        examples = examples[:max_examples]

    logging.info("Dataset=%s | lang=%s | examples=%d", dataset_name, lang_l, len(examples))
    q_l = [ex["q_L"] for ex in examples]
    q_en_gold = [ex["q_en"] for ex in examples]

    results: Dict = {
        "dataset": dataset_name,
        "lang": lang_l,
        "num_examples": len(examples),
    }
    if not constraints:
        logging.warning("No constraints provided; skipping.")
        return results

    for c in constraints:
        c_id = c["id"]
        c_en = c["text"]
        logging.info("Constraint=%s", c_id)

        c_l_list = translator_en2l.batched_translate(
            [c_en] * len(examples),
            batch_size=translation_batch_size,
            max_new_tokens=translation_max_new_tokens,
        )
        q_l_plus_c = [_attach_constraint(q, c_l) for q, c_l in zip(q_l, c_l_list)]
        q_en_plus_c = [_attach_constraint(q, c_en) for q in q_en_gold]

        q_en_mt = translator_l2en.batched_translate(
            q_l_plus_c,
            batch_size=translation_batch_size,
            max_new_tokens=translation_max_new_tokens,
        )
        err_reasoner = reasoner.reason_answer_batch(
            q_en_mt,
            batch_size=reasoner_batch_size,
            max_new_tokens=reasoner_max_new_tokens,
            temperature=reasoner_temperature,
            top_p=reasoner_top_p,
        )
        y_en_err = [x.answer for x in err_reasoner]
        reason_en_err = [x.reasoning for x in err_reasoner]
        y_l_err = _run_mt2(
            mt2_component,
            q_l=q_l_plus_c,
            q_en=q_en_mt,
            y_en=y_en_err,
            reason_en=reason_en_err,
            batch_size=mt2_batch_size,
            max_new_tokens=mt2_max_new_tokens,
        )

        base_reasoner = reasoner.reason_answer_batch(
            q_en_plus_c,
            batch_size=reasoner_batch_size,
            max_new_tokens=reasoner_max_new_tokens,
            temperature=reasoner_temperature,
            top_p=reasoner_top_p,
        )
        y_en_base = [x.answer for x in base_reasoner]
        reason_en_base = [x.reasoning for x in base_reasoner]
        y_l_base = _run_mt2(
            mt2_component,
            q_l=q_l_plus_c,
            q_en=q_en_plus_c,
            y_en=y_en_base,
            reason_en=reason_en_base,
            batch_size=mt2_batch_size,
            max_new_tokens=mt2_max_new_tokens,
        )

        acc_base = batch_constraint_accuracy(y_l_base, c_id)
        acc_err = batch_constraint_accuracy(y_l_err, c_id)
        base_ok = [check_constraint(pred, c_id) for pred in y_l_base]
        err_ok = [check_constraint(pred, c_id) for pred in y_l_err]
        base_ok_count = sum(base_ok)
        base_ok_to_err_bad_count = sum(1 for b, e in zip(base_ok, err_ok) if b and (not e))
        base_ok_to_err_bad_rate = (
            (base_ok_to_err_bad_count / base_ok_count) if base_ok_count > 0 else 0.0
        )

        results[f"constraint_{c_id}_acc_base"] = acc_base
        results[f"constraint_{c_id}_acc_err"] = acc_err
        results[f"constraint_{c_id}_acc_gap"] = acc_base - acc_err
        results[f"constraint_{c_id}_base_ok_to_err_bad_rate"] = base_ok_to_err_bad_rate
        results[f"_sample_{c_id}_base_0"] = y_l_base[0] if y_l_base else ""
        results[f"_sample_{c_id}_err_0"] = y_l_err[0] if y_l_err else ""
        logging.info(
            "Constraint=%s | base=%.4f err=%.4f gap=%.4f base_ok_to_err_bad_rate=%.4f",
            c_id,
            acc_base,
            acc_err,
            acc_base - acc_err,
            base_ok_to_err_bad_rate,
        )
    return results
