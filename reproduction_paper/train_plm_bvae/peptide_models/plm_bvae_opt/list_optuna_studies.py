#!/usr/bin/env python3
"""
List Optuna studies and optionally delete a named study.

Examples:
  - List studies as text
    python list_optuna_studies.py --storage-url sqlite:///optuna_plm_bvae.db

  - List studies as JSON
    python list_optuna_studies.py --storage-url sqlite:///optuna_plm_bvae.db --json

  - Delete a study
    python list_optuna_studies.py --storage-url sqlite:///optuna_plm_bvae.db --delete-study plm_bvae_opt_old
"""

import argparse
import optuna
import json
from typing import Any, Dict


def summarize_study(study: optuna.Study) -> Dict[str, Any]:
    """Return the study metadata and best completed trial as a dictionary."""
    info: Dict[str, Any] = {
        "study_name": study.study_name,
        "direction": str(study.direction),
        "n_trials": len(study.trials),
    }
    try:
        if study.best_trial is not None and study.best_trial.value is not None:
            info["best_value"] = float(study.best_value)
            info["best_trial_number"] = int(study.best_trial.number)
            info["best_params"] = study.best_trial.params
        else:
            info["best_value"] = None
            info["best_trial_number"] = None
            info["best_params"] = None
    except Exception:
        info["best_value"] = None
        info["best_trial_number"] = None
        info["best_params"] = None
    return info


def main():
    parser = argparse.ArgumentParser(
        description="List studies in Optuna storage and optionally delete a study."
    )
    parser.add_argument(
        "--storage-url",
        required=True,
        type=str,
        help="Optuna storage URL (e.g. sqlite:///optuna_plm_bvae.db)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output detailed study summaries as JSON.",
    )
    parser.add_argument(
        "--delete-study",
        type=str,
        default=None,
        help="Name of the study to delete after displaying the summaries.",
    )
    args = parser.parse_args()

    storage_url = args.storage_url


    try:
        study_names = optuna.get_all_study_names(storage=storage_url)
    except Exception as e:
        print(f"[ERROR] Failed to list studies: {e}")
        print("Hint: --storage-url requires a SQLAlchemy URL, e.g. sqlite:///optuna_plm_bvae.db")
        return

    if not study_names:
        print(f"[INFO] No studies found in storage '{storage_url}'.")
    else:
        if args.json:
            # Emit study summaries as a JSON array.
            summaries = []
            for name in study_names:
                try:
                    st = optuna.load_study(study_name=name, storage=storage_url)
                    summaries.append(summarize_study(st))
                except Exception as e:
                    summaries.append(
                        {
                            "study_name": name,
                            "error": f"Failed to load: {e}",
                        }
                    )
            print(json.dumps(summaries, ensure_ascii=False, indent=2))
        else:
            # Print readable summaries for interactive inspection.
            print(f"[INFO] Storage: {storage_url}")
            print(f"[INFO] Study count: {len(study_names)}\n")
            for name in study_names:
                print(f"Study: {name}")
                try:
                    st = optuna.load_study(study_name=name, storage=storage_url)
                    summary = summarize_study(st)
                    print(f"  direction          : {summary['direction']}")
                    print(f"  n_trials           : {summary['n_trials']}")
                    if summary["best_value"] is not None:
                        print(f"  best_value         : {summary['best_value']:.6f}")
                        print(f"  best_trial_number  : {summary['best_trial_number']}")
                        params_str = ", ".join(
                            f"{k}={v}" for k, v in (summary["best_params"] or {}).items()
                        )
                        print(f"  best_params        : {params_str}")
                    else:
                        print("  (no completed trials)")
                except Exception as e:
                    print(f"  Failed to load study details: {e}")
                print("-" * 60)

    # Delete the requested study after displaying the summaries.
    target = args.deleteStudy if hasattr(args, "deleteStudy") else args.delete_study
    if target:
        try:
            names = optuna.get_all_study_names(storage=storage_url)
            if target in names:
                optuna.delete_study(study_name=target, storage=storage_url)
                print(f"Deleted: {target}")
            else:
                print(f"Study '{target}' not found. Existing: {names}")
        except Exception as e:
            print(f"[ERROR] Failed to delete study: {e}")


if __name__ == "__main__":
    main()
