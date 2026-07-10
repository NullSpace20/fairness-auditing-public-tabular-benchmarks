"""Assemble the Journal of Big Data supplementary package (Additional file 1).

Copies existing Q1-upgrade outputs into the supplementary package folder,
syncs additional robustness outputs (R2A--R2C), writes pinned
requirements and the reproducibility fact sheet, and optionally rebuilds
Supplementary_Information.pdf and the distributable ZIP.

No experiments are run and no result values are edited.
"""
from __future__ import annotations

import shutil
import subprocess
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
Q1 = REPO / "q1_upgrade"
PP = Q1 / "results" / "phase5_postprocessing"
SUPP = PP / "supplementary"
TABLES = PP / "tables"
P4B = Q1 / "results" / "phase4b_optionB_30seed"
READY = P4B / "phase5_ready"
CODE = Q1 / "code_branch"
MS = REPO / "Paper_Springer_JBigData_Q1Upgrade"
MS_FIGS = MS / "Q1_Figures"

PKG = MS / "Supplementary_Materials_and_Reproducibility_Package"
SUP_ZIP = MS / "Supplementary_Materials_and_Reproducibility_Package.zip"

MANUSCRIPT_TITLE = (
    "Fairness Auditing in Tabular Machine Learning: "
    "A Multi-Dataset Benchmark of Bias, Mitigation, and Robustness"
)

# Never removed or overwritten by sync (curated in the manuscript folder).
PRESERVE_NAMES = frozenset(
    {
        "Supplementary_Information.tex",
        "README.txt",
        "si_tables",
    }
)

SYNC_SUBDIRS = ("data_tables", "figures", "code", "manifests", "revision_robustness")

ENV = {
    "python": "3.8.5",
    "aif360": "0.6.1",
    "fairlearn": "0.12.0",
    "scikit-learn": "1.3.2",
    "xgboost": "2.1.4",
    "folktables": "0.0.12",
    "pandas": "2.0.3",
    "numpy": "1.24.4",
    "scipy": "1.10.1",
    "statsmodels": "0.14.1",
    "matplotlib": "3.3.2",
}

COPY = [
    (SUPP / "full_aggregate_30seed.csv", "data_tables/full_aggregate_30seed.csv"),
    (P4B / "phase4b_results.csv", "data_tables/per_seed_full_results_3690.csv"),
    (SUPP / "supp_paired_statistical_tests.csv", "data_tables/paired_statistical_tests.csv"),
    (SUPP / "supp_acfs_rankings.csv", "data_tables/acfs_rankings.csv"),
    (SUPP / "supp_pareto_optimal.csv", "data_tables/pareto_optimal_configurations.csv"),
    (SUPP / "supp_bounded_di_robustness.csv", "data_tables/bounded_di_robustness.csv"),
    (TABLES / "main_table_runtime_summary.csv", "data_tables/runtime_summary.csv"),
    (READY / "runtime_fact_sheet.md", "manifests/runtime_fact_sheet.md"),
    (TABLES / "main_table_dataset_provenance.csv", "data_tables/dataset_provenance_and_fingerprints.csv"),
    (TABLES / "main_table_model_mitigation_scope.csv", "data_tables/method_scope.csv"),
    (P4B / "phase4b_config_manifest.csv", "manifests/configuration_manifest.csv"),
    (PP / "phase5a_manifest.json", "manifests/phase5a_postprocessing_manifest.json"),
    (PP / "validation_report.json", "manifests/validation_report.json"),
    (CODE / "phase5a_core.py", "code/phase5a_core.py"),
    (CODE / "run_phase5a.py", "code/run_phase5a.py"),
    (CODE / "build_manuscript_assets.py", "code/build_manuscript_assets.py"),
    (CODE / "pipeline_core.py", "code/pipeline_core.py"),
    (CODE / "sam_fair_select.py", "code/sam_fair_select.py"),
    (CODE / "run_intersectional_adult_baseline.py", "code/run_intersectional_adult_baseline.py"),
    (CODE / "run_xgboost_eg_probe.py", "code/run_xgboost_eg_probe.py"),
    (CODE / "build_xgboost_eg_probe_table.py", "code/build_xgboost_eg_probe_table.py"),
]

FIG_NAMES = [
    "fig_pipeline_workflow",
    "fig_dataset_comparison_heatmap",
    "fig_mitigation_tradeoff_slope",
    "fig_fairness_variability_box",
    "fig_pareto_accuracy_fairness",
    "fig_eg_vs_aif360",
    "fig_runtime_comparison",
]

R2A = Q1 / "results" / "revision_R2_age_robustness"
R2B = Q1 / "results" / "revision_R2B_eo_calibration"
R2C = Q1 / "results" / "revision_R2C_cfs_sensitivity"
R2_INTER = Q1 / "results" / "intersectional_adult_baseline"
R2_XGB = Q1 / "results" / "xgboost_eg_probe"

REVISION_COPY: list[tuple[Path, str, list[str]]] = [
    (
        R2A,
        "revision_robustness/age_binning",
        [
            "age_group_counts.csv",
            "age_robustness_summary.csv",
            "age_robustness_aggregate.csv",
            "age_robustness_stat_tests.csv",
            "age_robustness_classification.csv",
        ],
    ),
    (
        R2B,
        "revision_robustness/eo_calibration",
        [
            "eo_calibration_group_counts_summary.csv",
            "eo_calibration_comparison_summary.csv",
            "eo_calibration_stat_tests.csv",
            "eo_calibration_robustness_classification.csv",
        ],
    ),
    (
        R2C,
        "revision_robustness/cfs_sensitivity",
        [
            "cfs_sensitivity_aggregate.csv",
            "cfs_ranking_stability.csv",
            "cfs_top_configurations.csv",
            "cfs_sensitivity_classification.csv",
            "cfs_claim_check.csv",
        ],
    ),
    (
        R2_INTER,
        "revision_robustness/intersectional_adult_baseline",
        [
            "intersectional_per_seed.csv",
            "intersectional_summary.csv",
            "run_metadata.json",
        ],
    ),
    (
        R2_XGB,
        "revision_robustness/xgboost_eg_probe",
        [
            "xgboost_eg_probe_feasibility_per_seed.csv",
            "xgboost_eg_probe_feasibility_summary.csv",
            "xgboost_eg_probe_feasibility_metadata.json",
            "xgboost_eg_probe_full_per_seed.csv",
            "xgboost_eg_probe_full_summary.csv",
            "xgboost_eg_probe_full_metadata.json",
        ],
    ),
]

ZIP_SKIP_SUFFIXES = {".aux", ".log", ".out", ".toc", ".synctex.gz", ".fdb_latexmk", ".fls"}

RAW_DATA_PATTERNS = (
    "bank-additional-full.csv",
    "adult.data",
    "adult.test",
    "acs_income",
)


def sync_subdirs() -> None:
    PKG.mkdir(parents=True, exist_ok=True)
    for name in SYNC_SUBDIRS:
        target = PKG / name
        if target.exists():
            shutil.rmtree(target)
        target.mkdir(parents=True, exist_ok=True)


def copy_file(src: Path, dst_rel: str, missing: list[str]) -> None:
    dst = PKG / dst_rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.exists():
        shutil.copy2(src, dst)
    else:
        missing.append(str(src))


def copy_all(missing: list[str]) -> None:
    for src, dst in COPY:
        copy_file(src, dst, missing)
    for name in FIG_NAMES:
        for ext in ("pdf", "png"):
            copy_file(MS_FIGS / f"{name}.{ext}", f"figures/{name}.{ext}", missing)
    for src_root, dst_prefix, files in REVISION_COPY:
        for fname in files:
            copy_file(src_root / fname, f"{dst_prefix}/{fname}", missing)
        fig_src = src_root / "figures"
        fig_dst = PKG / dst_prefix / "figures"
        if fig_src.is_dir():
            fig_dst.mkdir(parents=True, exist_ok=True)
            for item in fig_src.iterdir():
                if item.is_file():
                    shutil.copy2(item, fig_dst / item.name)
        else:
            missing.append(str(fig_src))


def write_requirements() -> None:
    lines = [
        "# Pinned Python environment used for Q1-upgrade experiments and",
        f"# post-processing. Python {ENV['python']} (CPython).",
        "",
    ]
    for pkg, ver in ENV.items():
        if pkg == "python":
            continue
        lines.append(f"{pkg}=={ver}")
    (PKG / "requirements.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_repro_fact_sheet() -> None:
    txt = f"""# Reproducibility Fact Sheet

{MANUSCRIPT_TITLE}

## Software environment
- Python {ENV['python']} (CPython).
- Pinned package versions are in `requirements.txt`:
  aif360 {ENV['aif360']}, fairlearn {ENV['fairlearn']}, scikit-learn {ENV['scikit-learn']},
  xgboost {ENV['xgboost']}, folktables {ENV['folktables']}, pandas {ENV['pandas']},
  numpy {ENV['numpy']}, scipy {ENV['scipy']}, statsmodels {ENV['statsmodels']},
  matplotlib {ENV['matplotlib']}.

## Datasets (not redistributed here)
- Adult Income and Bank Marketing: UCI Machine Learning Repository.
- ACSIncome: Folktables interface to U.S. Census ACS microdata (2018, California, 1-Year).
- Row counts, targets, positive rates, protected settings, and the UCI Bank SHA-256
  fingerprint are in `data_tables/dataset_provenance_and_fingerprints.csv`.
- UCI Bank reference file: `bank-additional-full.csv` (41,188 rows).
- ACSIncome: California pool subsampled once to 49,999 rows with fixed sample seed 42;
  experiment seeds 42--71 affect train/test splits only.

## Split protocol
- 80/20 target-stratified train/test split per seed.
- 25% of the training partition held out for post-processing calibration under Equalized Odds.
- Seeds: 42--71 (30 consecutive seeds).

## Models and mitigations
- Models: Logistic Regression, Random Forest, Gradient Boosting, XGBoost, MLP.
- Mitigations on all five families: Baseline, Reweighing, Equalized Odds (AIF360).
- Exponentiated Gradient: Logistic Regression on all datasets; Random Forest on Adult and
  UCI Bank only. Not applied to Gradient Boosting, XGBoost, MLP, or ACSIncome Random Forest.
- Coverage tables: `data_tables/method_scope.csv`, `manifests/configuration_manifest.csv`.

## Run tally (verified)
- 3,690 completed main-grid runs; 0 failed, 0 skipped, 0 duplicate rows.
- Additional robustness outputs (age binning, EO calibration, CFS sensitivity, Adult
  intersectional baseline, 30-seed XGBoost--EG probe) are in `revision_robustness/`;
  they supplement but do not replace the main result files.
- The XGBoost--EG probe (180 runs on three settings; `revision_robustness/xgboost_eg_probe/`)
  is separate from the original 3,690-run main grid and is not merged into main-grid averages.

## How to reproduce analysis outputs
1. Create a Python {ENV['python']} environment and `pip install -r requirements.txt`.
2. Obtain raw datasets from the public sources above (not included in this package).
3. Per-seed results: `data_tables/per_seed_full_results_3690.csv`.
4. Post-processing scripts are in `code/` (`phase5a_core.py`, `run_phase5a.py`,
   `build_manuscript_assets.py`, `sam_fair_select.py`, `run_xgboost_eg_probe.py`;
   `pipeline_core.py` is a shared dependency).

## Public archive

- GitHub: https://github.com/NullSpace20/fairness-auditing-public-tabular-benchmarks
- Zenodo (this release): https://doi.org/10.5281/zenodo.21284708
"""
    (PKG / "REPRODUCIBILITY_FACT_SHEET.md").write_text(txt, encoding="utf-8")


def write_readme_if_missing() -> None:
    """Write README.txt only when absent so curated package README is preserved."""
    readme = PKG / "README.txt"
    if readme.exists():
        return
    txt = f"""Supplementary Materials and Reproducibility Package
===================================================

Manuscript: "{MANUSCRIPT_TITLE}" (Journal of Big Data submission).

This archive is Additional file 1. See REPRODUCIBILITY_FACT_SHEET.md and the
folder listing in Supplementary_Information.pdf for contents. Raw benchmark
datasets are NOT included.
"""
    readme.write_text(txt, encoding="utf-8")


def rebuild_supplementary_information_pdf() -> bool:
    tex = PKG / "Supplementary_Information.tex"
    if not tex.exists():
        print("SKIP SI PDF: Supplementary_Information.tex not found")
        return False
    for _ in range(2):
        result = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "Supplementary_Information.tex"],
            cwd=PKG,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print("pdflatex warning/error (see log); continuing")
    pdf = PKG / "Supplementary_Information.pdf"
    ok = pdf.exists()
    print("SI PDF rebuilt:" if ok else "SI PDF missing:", pdf)
    return ok


def build_zip() -> tuple[Path, int]:
    tmp_zip = SUP_ZIP.with_name(SUP_ZIP.stem + "_build.zip")
    if tmp_zip.exists():
        tmp_zip.unlink()
    count = 0
    with zipfile.ZipFile(tmp_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(PKG.rglob("*")):
            if not path.is_file():
                continue
            rel = path.relative_to(PKG).as_posix()
            if path.suffix.lower() in ZIP_SKIP_SUFFIXES:
                continue
            if any(raw in rel.lower() for raw in RAW_DATA_PATTERNS):
                continue
            zf.write(path, rel)
            count += 1
    test = zipfile.ZipFile(tmp_zip).testzip()
    if test is not None:
        raise RuntimeError(f"Corrupt ZIP entry: {test}")
    for attempt in range(5):
        try:
            if SUP_ZIP.exists():
                SUP_ZIP.unlink()
            tmp_zip.replace(SUP_ZIP)
            break
        except PermissionError:
            if attempt == 4:
                fallback = SUP_ZIP.with_name(SUP_ZIP.stem + "_R2E.zip")
                shutil.copyfile(tmp_zip, fallback)
                tmp_zip.unlink(missing_ok=True)
                print(
                    "WARNING: could not overwrite locked ZIP;",
                    SUP_ZIP,
                    "\nFresh build written to:",
                    fallback,
                )
                return fallback, count
            import time
            time.sleep(1.5)
    else:
        tmp_zip.unlink(missing_ok=True)
    return SUP_ZIP, count


def main() -> None:
    sync_subdirs()
    missing: list[str] = []
    copy_all(missing)
    write_requirements()
    write_repro_fact_sheet()
    write_readme_if_missing()
    rebuild_supplementary_information_pdf()
    zip_path, zip_entries = build_zip()
    files = sorted(p.relative_to(PKG).as_posix() for p in PKG.rglob("*") if p.is_file())
    print("PACKAGE:", PKG)
    print("FILE COUNT:", len(files))
    print("ZIP:", zip_path, "entries:", zip_entries)
    for f in files:
        print("  ", f)
    if missing:
        print("MISSING SOURCES:")
        for m in missing:
            print("  ", m)
    else:
        print("MISSING SOURCES: none")


if __name__ == "__main__":
    main()
