"""Build the public repository / Zenodo release package from existing outputs.

No experiments are run. Raw benchmark datasets are not copied.
"""
from __future__ import annotations

import shutil
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
MS = REPO / "Paper_Springer_JBigData_Q1Upgrade"
SUPP_PKG = MS / "Supplementary_Materials_and_Reproducibility_Package"
CODE = REPO / "q1_upgrade" / "code_branch"
OUT = MS / "Public_Repository_Release_Package"
OUT_ZIP = MS / "Public_Repository_Release_Package.zip"

COPY_DIRS_FROM_SUPP = (
    "data_tables",
    "figures",
    "manifests",
    "revision_robustness",
)

RELEASE_DOCS = REPO / "q1_upgrade" / "release_docs"

COPY_FILES_FROM_SUPP = (
    "requirements.txt",
    "REPRODUCIBILITY_FACT_SHEET.md",
)

CODE_FILES = (
    "phase5a_core.py",
    "run_phase5a.py",
    "build_manuscript_assets.py",
    "pipeline_core.py",
    "fairness_utils.py",
    "loaders.py",
    "mitigations_aif360.py",
    "run_revision_R2A_age_robustness.py",
    "run_revision_R2B_eo_calibration.py",
    "run_revision_R2C_cfs_sensitivity.py",
    "assemble_supplementary_zip.py",
    "assemble_public_repository_release.py",
    "run_phase4b.py",
)

RAW_DATA_PATTERNS = (
    "bank-additional-full.csv",
    "adult.data",
    "adult.test",
)


def reset_out() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    (OUT / "code").mkdir()


def copy_supplementary_tree() -> None:
    for name in COPY_DIRS_FROM_SUPP:
        src = SUPP_PKG / name
        if src.is_dir():
            shutil.copytree(src, OUT / name)
    for name in COPY_FILES_FROM_SUPP:
        src = SUPP_PKG / name
        if src.is_file():
            shutil.copy2(src, OUT / name)


def copy_code() -> None:
    for name in CODE_FILES:
        src = CODE / name
        if src.is_file():
            shutil.copy2(src, OUT / "code" / name)


def copy_curated_docs() -> None:
    for name in (
        "README.md",
        "FILE_MANIFEST.md",
        "DOI_RELEASE_NOTE.md",
        "LICENSE_NOTE.txt",
        "CITATION.cff",
    ):
        src = RELEASE_DOCS / name
        if not src.is_file():
            raise FileNotFoundError(f"Create curated doc before build: {src}")
        shutil.copy2(src, OUT / name)


def build_zip() -> tuple[int, int]:
    if OUT_ZIP.exists():
        OUT_ZIP.unlink()
    count = 0
    total = 0
    with zipfile.ZipFile(OUT_ZIP, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(OUT.rglob("*")):
            if not path.is_file():
                continue
            rel = path.relative_to(OUT).as_posix()
            if any(raw in rel.lower() for raw in RAW_DATA_PATTERNS):
                continue
            zf.write(path, rel)
            count += 1
            total += path.stat().st_size
    test = zipfile.ZipFile(OUT_ZIP).testzip()
    if test is not None:
        raise RuntimeError(f"Corrupt ZIP entry: {test}")
    return count, total


def main() -> None:
    reset_out()
    copy_supplementary_tree()
    copy_code()
    copy_curated_docs()
    n, size = build_zip()
    print("RELEASE PACKAGE:", OUT)
    print("ZIP:", OUT_ZIP)
    print("FILES:", n, "SIZE_BYTES:", size)


if __name__ == "__main__":
    main()
