#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import sys
from pathlib import Path
from statistics import mean


def main():
    repo_root = Path(__file__).resolve().parents[2]
    redirector_dir = repo_root / "redirector"
    results_dir = redirector_dir / "cowrie_analysis_results"
    metrics_dir = results_dir / "metrics"
    reports_dir = results_dir / "reports"

    # Permitir imports relativos como em main.py
    sys.path.insert(0, str(redirector_dir))

    # Imports locais (mesmo que o pipeline)
    from config.seed_sets import SEED_SETS  # type: ignore
    from config.command_relations import COMMAND_RELATIONS  # type: ignore
    from expanders.semi_supervised_expander import SemiSupervisedExpander  # type: ignore
    from main import create_stratified_holdout_split  # type: ignore

    # Carrega resultados completos (precisamos de initial/refined scores)
    complete_path = results_dir / "complete_analysis_results.json"
    if not complete_path.exists():
        print(f"ERRO: Arquivo não encontrado: {complete_path}")
        sys.exit(1)

    with complete_path.open("r", encoding="utf-8") as f:
        complete = json.load(f)

    initial_scores = complete.get("initial_scores", {})
    refined_scores = complete.get("refined_scores", {})

    # Reconstrói expanded sets e split hold-out exatamente como no pipeline (seed=42)
    expander = SemiSupervisedExpander(SEED_SETS, COMMAND_RELATIONS)
    expanded_sets = expander.expand_seeds()
    _, holdout_sets = create_stratified_holdout_split(expanded_sets, test_size=0.2, random_state=42)

    traits = [
        "HonestyHumility",
        "Emotionality",
        "Extraversion",
        "Agreeableness",
        "Conscientiousness",
        "OpennessToExperience",
    ]

    deltas_all = []
    deltas_by_trait = {}

    for trait in traits:
        pos_key = f"{trait}_Positive"
        neg_key = f"{trait}_Negative"
        deltas_trait = []

        for cmd in holdout_sets.get(pos_key, set()):
            if cmd in initial_scores and cmd in refined_scores and trait in initial_scores[cmd] and trait in refined_scores[cmd]:
                d = refined_scores[cmd][trait]["positive"] - initial_scores[cmd][trait]["positive"]
                deltas_trait.append(d)
        for cmd in holdout_sets.get(neg_key, set()):
            if cmd in initial_scores and cmd in refined_scores and trait in initial_scores[cmd] and trait in refined_scores[cmd]:
                d = refined_scores[cmd][trait]["positive"] - initial_scores[cmd][trait]["positive"]
                deltas_trait.append(d)

        if deltas_trait:
            deltas_by_trait[trait] = {
                "samples": len(deltas_trait),
                "delta_avg": float(mean(deltas_trait)),
            }
            deltas_all.extend(deltas_trait)
        else:
            deltas_by_trait[trait] = {
                "samples": 0,
                "delta_avg": None,
            }

    global_avg = float(mean(deltas_all)) if deltas_all else None
    global_n = len(deltas_all)

    # Atualiza holdout_metrics.json anexando delta_avg
    holdout_metrics_path = metrics_dir / "holdout_metrics.json"
    if holdout_metrics_path.exists():
        with holdout_metrics_path.open("r", encoding="utf-8") as f:
            holdout_metrics = json.load(f)
    else:
        holdout_metrics = {}

    for trait, vals in deltas_by_trait.items():
        if trait not in holdout_metrics:
            holdout_metrics[trait] = {}
        holdout_metrics[trait]["delta_avg"] = vals["delta_avg"]
        holdout_metrics[trait]["delta_samples"] = vals["samples"]

    holdout_metrics["global_delta_avg"] = global_avg
    holdout_metrics["global_delta_samples"] = global_n

    metrics_dir.mkdir(parents=True, exist_ok=True)
    with holdout_metrics_path.open("w", encoding="utf-8") as f:
        json.dump(holdout_metrics, f, indent=2, ensure_ascii=False)

    # Acrescenta uma linha no relatório resumido
    report_path = reports_dir / "analysis_summary.md"
    line = f"\n- Δ médio (hold-out 20%): {global_avg:.3f} (N={global_n})\n" if global_avg is not None else "\n- Δ médio (hold-out 20%): N/A\n"
    try:
        reports_dir.mkdir(parents=True, exist_ok=True)
        if report_path.exists():
            with report_path.open("a", encoding="utf-8") as f:
                f.write(line)
        else:
            with report_path.open("w", encoding="utf-8") as f:
                f.write("# Resumo de Análise\n")
                f.write(line)
    except Exception as e:
        print(f"Aviso: não foi possível atualizar o relatório ({e})")

    print(f"delta_holdout_global={global_avg:.6f} N={global_n}")
    for trait in traits:
        t = deltas_by_trait[trait]
        print(f"{trait}: delta_avg={t['delta_avg'] if t['delta_avg'] is not None else 'N/A'} N={t['samples']}")


if __name__ == "__main__":
    main()






