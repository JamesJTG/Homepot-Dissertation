from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# Paths
# ============================================================

ROOT = Path("research/diagnostic_reliability")

TABLES = ROOT / "dissertation_outputs" / "tables"

FIGURES = ROOT / "dissertation_outputs" / "figures"
MAIN = FIGURES / "main"
APPENDIX = FIGURES / "appendix"

MAIN.mkdir(parents=True, exist_ok=True)
APPENDIX.mkdir(parents=True, exist_ok=True)


# ============================================================
# Academic figure defaults
# ============================================================

plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 12,
})


def save_figure(fig, stem, directory=MAIN):
    """
    Save dissertation figure in both high-resolution PNG
    and vector PDF formats.
    """
    png = directory / f"{stem}.png"
    pdf = directory / f"{stem}.pdf"

    fig.savefig(
        png,
        dpi=300,
        bbox_inches="tight",
        facecolor="white"
    )

    fig.savefig(
        pdf,
        bbox_inches="tight",
        facecolor="white"
    )

    print(f"[FIGURE] {png}")
    print(f"[FIGURE] {pdf}")

    plt.close(fig)


# ============================================================
# Load verified dissertation evidence tables
# ============================================================

summary = pd.read_csv(
    TABLES / "table_01_experimental_dataset_and_primary_outcome_summary.csv"
)

dev_performance = pd.read_csv(
    TABLES / "table_10_development_model_performance.csv"
)

holdout_performance = pd.read_csv(
    TABLES / "table_16_final_holdout_estimator_performance.csv"
)

dev_paired = pd.read_csv(
    TABLES / "table_13_development_paired_model_differences.csv"
)

holdout_paired = pd.read_csv(
    TABLES / "table_18_final_holdout_paired_differences.csv"
)

evidence_condition = pd.read_csv(
    TABLES / "table_05_primary_outcome_by_evidence_condition.csv"
)

fault_class = pd.read_csv(
    TABLES / "table_06_primary_outcome_by_fault_class.csv"
)

error_categories = pd.read_csv(
    TABLES / "table_04_error_category_distribution.csv"
)

holdout_policy = pd.read_csv(
    TABLES / "table_20_holdout_pidrl_policy_state_evaluation.csv"
)


# ============================================================
# Figure 1
# Experimental sample and outcome flow
# ============================================================

fig, ax = plt.subplots(figsize=(11, 5.5))
ax.axis("off")

boxes = [
    (0.08, 0.70, "150\nIndependent\nscenarios"),
    (0.28, 0.70, "600\nEvidence\ncases"),
    (0.48, 0.70, "4,200\nLLM\ngenerations"),
    (0.68, 0.70, "600\nModal case\ndiagnoses"),
]

for x, y, text in boxes:
    ax.text(
        x,
        y,
        text,
        ha="center",
        va="center",
        transform=ax.transAxes,
        bbox=dict(
            boxstyle="round,pad=0.5",
            facecolor="white",
            edgecolor="black",
            linewidth=1.2,
        ),
    )

for x1, x2 in [(0.13, 0.23), (0.33, 0.43), (0.53, 0.63)]:
    ax.annotate(
        "",
        xy=(x2, 0.70),
        xytext=(x1, 0.70),
        xycoords=ax.transAxes,
        arrowprops=dict(arrowstyle="->", linewidth=1.2),
    )

ax.text(
    0.84,
    0.82,
    "308\nUNKNOWN",
    ha="center",
    va="center",
    transform=ax.transAxes,
    bbox=dict(
        boxstyle="round,pad=0.5",
        facecolor="white",
        edgecolor="black",
        linewidth=1.2,
    ),
)

ax.text(
    0.84,
    0.56,
    "292\nSubstantive diagnoses\neligible for Y_D",
    ha="center",
    va="center",
    transform=ax.transAxes,
    bbox=dict(
        boxstyle="round,pad=0.5",
        facecolor="white",
        edgecolor="black",
        linewidth=1.2,
    ),
)

ax.annotate(
    "",
    xy=(0.79, 0.82),
    xytext=(0.73, 0.72),
    xycoords=ax.transAxes,
    arrowprops=dict(arrowstyle="->", linewidth=1.2),
)

ax.annotate(
    "",
    xy=(0.79, 0.56),
    xytext=(0.73, 0.68),
    xycoords=ax.transAxes,
    arrowprops=dict(arrowstyle="->", linewidth=1.2),
)

ax.text(
    0.73,
    0.25,
    "56\nCorrect",
    ha="center",
    va="center",
    transform=ax.transAxes,
    bbox=dict(
        boxstyle="round,pad=0.5",
        facecolor="white",
        edgecolor="black",
        linewidth=1.2,
    ),
)

ax.text(
    0.93,
    0.25,
    "236\nIncorrect",
    ha="center",
    va="center",
    transform=ax.transAxes,
    bbox=dict(
        boxstyle="round,pad=0.5",
        facecolor="white",
        edgecolor="black",
        linewidth=1.2,
    ),
)

ax.annotate(
    "",
    xy=(0.73, 0.32),
    xytext=(0.82, 0.50),
    xycoords=ax.transAxes,
    arrowprops=dict(arrowstyle="->", linewidth=1.2),
)

ax.annotate(
    "",
    xy=(0.93, 0.32),
    xytext=(0.86, 0.50),
    xycoords=ax.transAxes,
    arrowprops=dict(arrowstyle="->", linewidth=1.2),
)

ax.text(
    0.5,
    0.04,
    "The 4,200 generations are repeated generations nested within 600 evidence cases "
    "and are not treated as independent modelling observations.",
    ha="center",
    va="center",
    transform=ax.transAxes,
    fontsize=9,
)

ax.set_title("Experimental Sample and Primary Diagnostic Outcome Flow")

save_figure(
    fig,
    "figure_01_experimental_sample_and_outcome_flow"
)

# ============================================================
# Figure 2
# Development model performance
# ============================================================

metrics = [
    ("brier_score", "Brier score", "Lower is better"),
    ("auroc", "AUROC", "Higher is better"),
    ("auprc", "AUPRC", "Higher is better"),
    ("ece_5_quantile_bins", "ECE", "Lower is better"),
]

panel_labels = ["(a)", "(b)", "(c)", "(d)"]

models = dev_performance["model"].tolist()

model_labels = {
    "M_E": r"$M_E$",
    "M_C": r"$M_C$",
    "M_H": r"$M_H$",
    "M_U": r"$M_U$",
    "M_EU": r"$M_{EU}$",
}

display_models = [
    model_labels.get(model, model)
    for model in models
]

fig, axes = plt.subplots(
    2,
    2,
    figsize=(10, 8),
)

for ax, panel, (column, label, direction) in zip(
    axes.flat,
    panel_labels,
    metrics,
):
    values = dev_performance[column].to_numpy()

    ax.bar(display_models, values)

    ax.set_title(
        f"{panel} {label}\n{direction}"
    )
    ax.set_xlabel("Reliability estimator")
    ax.set_ylabel(label)

    ax.grid(
        axis="y",
        alpha=0.25,
    )

    for i, value in enumerate(values):
        ax.text(
            i,
            value,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

fig.suptitle(
    "Reliability Estimator Performance on "
    "Development Out-of-Fold Predictions"
)

fig.tight_layout(
    rect=[0, 0, 1, 0.95]
)

save_figure(
    fig,
    "figure_02_development_model_performance"
)

# ============================================================
# Figure 3
# Development versus holdout comparison
# M_E and M_EU only
# ============================================================

dev = (
    dev_performance[
        dev_performance["model"].isin(["M_E", "M_EU"])
    ]
    .set_index("model")
)

hold = (
    holdout_performance[
        holdout_performance["model"].isin(["M_E", "M_EU"])
    ]
    .set_index("model")
)

metrics = [
    ("brier_score", "Brier score", "Lower is better"),
    ("auroc", "AUROC", "Higher is better"),
    ("auprc", "AUPRC", "Higher is better"),
    ("ece_5_quantile_bins", "ECE", "Lower is better"),
]

panel_labels = ["(a)", "(b)", "(c)", "(d)"]

fig, axes = plt.subplots(
    2,
    2,
    figsize=(10, 8.4),
)

x = np.arange(2)
width = 0.36

for ax, panel, (column, label, direction) in zip(
    axes.flat,
    panel_labels,
    metrics,
):

    dev_values = [
        dev.loc["M_E", column],
        dev.loc["M_EU", column],
    ]

    hold_values = [
        hold.loc["M_E", column],
        hold.loc["M_EU", column],
    ]

    bars1 = ax.bar(
        x - width / 2,
        dev_values,
        width,
        label="Development OOF",
    )

    bars2 = ax.bar(
        x + width / 2,
        hold_values,
        width,
        label="Final holdout",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [r"$M_E$", r"$M_{EU}$"]
    )

    ax.set_xlabel("Reliability estimator")
    ax.set_ylabel(label)

    ax.set_title(
        f"{panel} {label}\n{direction}"
    )

    ax.grid(
        axis="y",
        alpha=0.25,
    )

    for bars in (bars1, bars2):
        for bar in bars:
            height = bar.get_height()

            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f"{height:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

handles, labels = axes[0, 0].get_legend_handles_labels()

fig.legend(
    handles,
    labels,
    loc="lower center",
    ncol=2,
    frameon=True,
    bbox_to_anchor=(0.5, 0.01),
)

fig.suptitle(
    r"Development and Final Holdout Performance "
    r"of the $M_E$ and $M_{EU}$ Reliability Estimators"
)

fig.tight_layout(
    rect=[0, 0.07, 1, 0.95]
)

save_figure(
    fig,
    "figure_03_development_vs_holdout_estimator_performance"
)

# ============================================================
# Figure 4
# Paired Brier difference with scenario-bootstrap 95% CI
#
# Difference = M_EU - M_E
# Negative values favour M_EU.
# Positive values favour M_E.
# ============================================================

dev_brier = dev_paired[
    (dev_paired["reference_model"] == "M_E")
    & (dev_paired["comparison_model"] == "M_EU")
    & (dev_paired["metric"] == "brier_score")
].iloc[0]

hold_brier = holdout_paired[
    (holdout_paired["reference_model"] == "M_E")
    & (holdout_paired["comparison_model"] == "M_EU")
    & (holdout_paired["metric"] == "brier_score")
].iloc[0]

labels = [
    "Development OOF",
    "Final holdout",
]

points = np.array([
    dev_brier["point_difference"],
    hold_brier["point_difference"],
])

lower = np.array([
    dev_brier["ci_lower_95"],
    hold_brier["ci_lower_95"],
])

upper = np.array([
    dev_brier["ci_upper_95"],
    hold_brier["ci_upper_95"],
])

errors = np.vstack([
    points - lower,
    upper - points,
])

fig, ax = plt.subplots(
    figsize=(8, 3.9)
)

y = np.arange(len(labels))

ax.errorbar(
    points,
    y,
    xerr=errors,
    fmt="o",
    capsize=5,
    linewidth=1.4,
)

ax.axvline(
    0,
    linestyle="--",
    linewidth=1,
)

ax.set_yticks(y)
ax.set_yticklabels(labels)

ax.set_xlabel(
    r"Brier-score difference ($M_{EU}-M_E$)"
)

ax.set_title(
    r"Paired Scenario-Bootstrap Brier Difference "
    r"Between $M_{EU}$ and $M_E$"
)

ax.grid(
    axis="x",
    alpha=0.25,
)

ax.text(
    0.5,
    -0.24,
    r"Negative values favour $M_{EU}$; "
    r"positive values favour $M_E$.",
    transform=ax.transAxes,
    fontsize=9,
    ha="center",
)

fig.tight_layout(
    rect=[0, 0.07, 1, 1]
)

save_figure(
    fig,
    "figure_04_paired_brier_difference_development_and_holdout"
)

# ============================================================
# Figure 5
# Binned calibration of M_E and M_EU
# Development OOF versus final holdout
# ============================================================

ANALYSIS_OUTPUTS = ROOT / "analysis" / "outputs"

dev_calibration = pd.read_csv(
    ANALYSIS_OUTPUTS
    / "a8_8_evaluation_and_calibration"
    / "a8_8_oof_reliability_calibration_bins.csv"
)

holdout_calibration = pd.read_csv(
    ANALYSIS_OUTPUTS
    / "a8_12_feature_level_evaluation"
    / "a8_12_holdout_reliability_calibration_bins.csv"
)

fig, axes = plt.subplots(
    1,
    2,
    figsize=(11, 5),
    sharex=True,
    sharey=True,
)

model_display_labels = {
    "M_E": r"$M_E$",
    "M_EU": r"$M_{EU}$",
}


# ------------------------------------------------------------
# (a) Development OOF
# ------------------------------------------------------------

ax = axes[0]

for model in ["M_E", "M_EU"]:
    subset = (
        dev_calibration[
            dev_calibration["model"] == model
        ]
        .sort_values("mean_predicted_reliability")
    )

    ax.plot(
        subset["mean_predicted_reliability"],
        subset["observed_correctness"],
        marker="o",
        linewidth=1.5,
        label=model_display_labels[model],
    )

ax.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    linewidth=1,
    label="Perfect calibration",
)

ax.set_title("(a) Development out-of-fold")
ax.set_xlabel("Mean predicted reliability")
ax.set_ylabel("Observed diagnostic correctness")
ax.grid(alpha=0.25)


# ------------------------------------------------------------
# (b) Final holdout
# ------------------------------------------------------------

ax = axes[1]

for model in ["M_E", "M_EU"]:
    subset = (
        holdout_calibration[
            holdout_calibration["model"] == model
        ]
        .sort_values("mean_predicted_reliability")
    )

    ax.plot(
        subset["mean_predicted_reliability"],
        subset["observed_correctness"],
        marker="o",
        linewidth=1.5,
        label=model_display_labels[model],
    )

ax.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    linewidth=1,
    label="Perfect calibration",
)

ax.set_title("(b) Final holdout")
ax.set_xlabel("Mean predicted reliability")
ax.grid(alpha=0.25)


# ------------------------------------------------------------
# Shared formatting
# ------------------------------------------------------------

for ax in axes:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect(
        "equal",
        adjustable="box",
    )

handles, labels = axes[0].get_legend_handles_labels()

fig.legend(
    handles,
    labels,
    loc="lower center",
    ncol=3,
    frameon=True,
    bbox_to_anchor=(0.5, -0.01),
)

fig.suptitle(
    r"Binned Calibration of the $M_E$ and $M_{EU}$ "
    r"Reliability Estimators"
)

fig.text(
    0.5,
    0.055,
    "Points represent quantile-binned calibration summaries.",
    ha="center",
    fontsize=9,
)

fig.tight_layout(
    rect=[0, 0.12, 1, 0.94]
)

save_figure(
    fig,
    "figure_05_development_and_holdout_calibration"
)

# ============================================================
# Appendix Figure 1
# Diagnostic outcomes by evidence condition
# ============================================================

fig, ax = plt.subplots(figsize=(9, 5.5))

x = np.arange(len(evidence_condition))
width = 0.36

ax.bar(
    x - width / 2,
    evidence_condition["Correct"],
    width,
    label="Correct",
)

ax.bar(
    x + width / 2,
    evidence_condition["Incorrect"],
    width,
    label="Incorrect",
)

labels = (
    evidence_condition["evidence_condition"]
    .str.replace("_", " ", regex=False)
)

ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=20, ha="right")

ax.set_xlabel("Evidence condition")
ax.set_ylabel("Eligible evidence cases")
ax.set_title("Primary Diagnostic Outcome by Evidence Condition")

ax.legend()
ax.grid(axis="y", alpha=0.25)

fig.tight_layout()

save_figure(
    fig,
    "appendix_figure_01_outcome_by_evidence_condition",
    APPENDIX,
)


# ============================================================
# Appendix Figure 2
# Diagnostic outcomes by fault class
# ============================================================

fig, ax = plt.subplots(figsize=(10, 5.5))

x = np.arange(len(fault_class))
width = 0.36

ax.bar(
    x - width / 2,
    fault_class["Correct"],
    width,
    label="Correct",
)

ax.bar(
    x + width / 2,
    fault_class["Incorrect"],
    width,
    label="Incorrect",
)

labels = (
    fault_class["Fault class"]
    .str.replace("_", " ", regex=False)
)

ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=25, ha="right")

ax.set_xlabel("Ground-truth fault class")
ax.set_ylabel("Eligible evidence cases")
ax.set_title("Primary Diagnostic Outcome by Ground-Truth Fault Class")

ax.legend()
ax.grid(axis="y", alpha=0.25)

fig.tight_layout()

save_figure(
    fig,
    "appendix_figure_02_outcome_by_fault_class",
    APPENDIX,
)


# ============================================================
# Appendix Figure 3
# Case-level error categories
# ============================================================

fig, ax = plt.subplots(figsize=(9, 5.5))

plot_errors = error_categories[
    error_categories["Cases"] > 0
].copy()

labels = (
    plot_errors["Error category"]
    .str.replace("_", " ", regex=False)
)

y = np.arange(len(plot_errors))

ax.barh(y, plot_errors["Cases"])

ax.set_yticks(y)
ax.set_yticklabels(labels)

ax.set_xlabel("Evidence cases")
ax.set_ylabel("Error category")
ax.set_title("Case-Level Diagnostic Outcome Categories")

ax.grid(axis="x", alpha=0.25)

fig.tight_layout()

save_figure(
    fig,
    "appendix_figure_03_error_category_distribution",
    APPENDIX,
)


# ============================================================
# Appendix Figure 4
# Final holdout PIDRL policy states
# ============================================================

fig, ax = plt.subplots(figsize=(8, 5.5))

states = holdout_policy["decision_state"]
counts = holdout_policy["n_cases"]

ax.bar(states, counts)

ax.set_xlabel("PIDRL decision state")
ax.set_ylabel("Holdout evidence cases")
ax.set_title("Final Holdout PIDRL Decision-State Distribution")

ax.grid(axis="y", alpha=0.25)

for i, value in enumerate(counts):
    ax.text(
        i,
        value,
        str(int(value)),
        ha="center",
        va="bottom",
    )

fig.tight_layout()

save_figure(
    fig,
    "appendix_figure_04_holdout_pidrl_decision_states",
    APPENDIX,
)

print()
print("=" * 72)
print("DISSERTATION RESULTS FIGURE GENERATION COMPLETE")
print("Main figures: 5")
print("Appendix figures: 4")
print("Formats: PNG and PDF")
print("=" * 72)