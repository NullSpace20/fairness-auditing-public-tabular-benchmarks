"""Build optional auditor checklist figure (W1) from Section 6.5.

Grayscale, protocol-scoped; no compliance or deployment claims.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

REPO = Path(__file__).resolve().parents[2]
OUT_DIR = REPO / "Paper_Springer_JBigData_Q1Upgrade" / "Q1_Figures"

ITEMS = [
    (
        "Audit more than one dataset",
        "Dominant protected settings differed across Adult, UCI Bank, and ACSIncome.",
    ),
    (
        "Report component metrics, not only composites",
        "DI can dominate CFS while |SPD| stays modest; read cells alongside marginals.",
    ),
    (
        "Match mitigation to objective",
        "Equalized Odds and EG-DP cost more accuracy/runtime; Reweighing is lighter.",
    ),
    (
        "Treat EG as scoped unless fully evaluated",
        "Main-grid Pareto uses baseline, Reweighing, and EO; EG overlays are illustrative.",
    ),
    (
        "Check protocol choices",
        "Report calibration, age binning, and other preprocessing decisions.",
    ),
    (
        "State runtime and accuracy budget",
        "Mitigation scope and SAM-Fair selections depend on stated budgets.",
    ),
    (
        "Make selection assumptions explicit",
        "Document ACFS weights, delta, and optional uncertainty/runtime penalties.",
    ),
]


def build_figure() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    fig_w, fig_h = 7.0, 9.5
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Title
    ax.text(
        0.5,
        0.97,
        "Practical fairness-audit checklist under this protocol",
        ha="center",
        va="top",
        fontsize=13,
        fontweight="bold",
        color="black",
    )
    ax.text(
        0.5,
        0.935,
        "Benchmark audit habits supported by the reported evidence (not a compliance checklist)",
        ha="center",
        va="top",
        fontsize=8.5,
        color="#444444",
        style="italic",
    )

    n = len(ITEMS)
    top = 0.88
    bottom = 0.06
    gap = 0.012
    box_h = (top - bottom - gap * (n - 1)) / n
    left = 0.06
    width = 0.88

    for i, (title, detail) in enumerate(ITEMS):
        y = top - i * (box_h + gap) - box_h
        box = FancyBboxPatch(
            (left, y),
            width,
            box_h,
            boxstyle="round,pad=0.008,rounding_size=0.015",
            linewidth=0.8,
            edgecolor="#555555",
            facecolor="#f2f2f2" if i % 2 == 0 else "#ffffff",
            transform=ax.transAxes,
            clip_on=False,
        )
        ax.add_patch(box)

        num_x = left + 0.03
        text_x = left + 0.08
        cy = y + box_h / 2

        ax.text(
            num_x,
            cy + 0.012,
            f"{i + 1}.",
            ha="left",
            va="center",
            fontsize=10,
            fontweight="bold",
            color="black",
            transform=ax.transAxes,
        )
        ax.text(
            text_x,
            cy + 0.018,
            title,
            ha="left",
            va="center",
            fontsize=9.5,
            fontweight="bold",
            color="black",
            transform=ax.transAxes,
        )
        ax.text(
            text_x,
            cy - 0.022,
            detail,
            ha="left",
            va="center",
            fontsize=7.8,
            color="#333333",
            wrap=True,
            transform=ax.transAxes,
        )

    pdf_path = OUT_DIR / "fig_auditor_checklist.pdf"
    eps_path = OUT_DIR / "fig_auditor_checklist.eps"
    png_path = OUT_DIR / "fig_auditor_checklist.png"

    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    fig.savefig(eps_path, bbox_inches="tight", facecolor="white", format="eps")
    fig.savefig(png_path, bbox_inches="tight", facecolor="white", dpi=200)
    plt.close(fig)

    print("Wrote", pdf_path)
    print("Wrote", eps_path)
    print("Wrote", png_path)


if __name__ == "__main__":
    build_figure()
