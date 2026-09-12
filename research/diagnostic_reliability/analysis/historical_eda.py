"""
Focused exploratory data analysis of historical HOMEPOT telemetry.

Purpose
-------
This analysis characterises the existing HOMEPOT simulator population before
the dissertation's controlled diagnostic scenarios are designed.

It is intentionally limited to:
1. Core telemetry distributions.
2. Historical anomaly regimes.
3. Relationships between telemetry variables.
4. Health-state versus telemetry comparisons.
5. Temporal sampling behaviour.

This script does not assign causal diagnostic ground truth to historical data.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sqlalchemy import create_engine


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATABASE_URL = os.environ.get("HOMEPOT_DATABASE_URL")

CORE_METRICS = [
    "cpu_percent",
    "memory_percent",
    "disk_percent",
    "network_latency_ms",
    "error_rate",
]

EXTENDED_METRICS = [
    "transaction_count",
    "transaction_volume",
    "active_connections",
    "queue_depth",
]

ALL_METRICS = CORE_METRICS + EXTENDED_METRICS


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data():
    if not DATABASE_URL:
        raise RuntimeError(
            "HOMEPOT_DATABASE_URL environment variable is not set. "
            "Set it to the PostgreSQL connection URL for the HOMEPOT database."
        )

    engine = create_engine(DATABASE_URL)

    telemetry_query = """
    SELECT
        dm.id,
        dm.timestamp,
        dm.device_id,
        dm.cpu_percent,
        dm.memory_percent,
        dm.disk_percent,
        dm.network_latency_ms,
        dm.transaction_count,
        dm.transaction_volume,
        dm.error_rate,
        dm.active_connections,
        dm.queue_depth
    FROM device_metrics dm
    WHERE NOT (
        dm.device_id = 1
        AND dm.timestamp < (
            SELECT MIN(timestamp)::timestamp
            FROM health_checks
            WHERE device_id = 1
        )
    )
    ORDER BY dm.device_id, dm.timestamp;
    """

    health_query = """
    SELECT
        id,
        device_id,
        timestamp,
        is_healthy,
        CASE
            WHEN is_healthy = TRUE THEN 'Healthy'
            ELSE COALESCE(
                response_data->>'error',
                'Unhealthy - unspecified'
            )
        END AS health_state
    FROM health_checks
    ORDER BY device_id, timestamp;
    """

    telemetry = pd.read_sql(telemetry_query, engine)
    health = pd.read_sql(health_query, engine)

    engine.dispose()

    return telemetry, health


# ---------------------------------------------------------------------------
# Numerical summaries
# ---------------------------------------------------------------------------

def create_descriptive_summary(df):
    summary = df[ALL_METRICS].describe(
        percentiles=[0.05, 0.25, 0.50, 0.75, 0.95, 0.99]
    ).T

    summary.to_csv(OUTPUT_DIR / "telemetry_descriptive_summary.csv")


def classify_anomalies(df):
    result = df.copy()

    result["cpu_high"] = result["cpu_percent"] > 90
    result["latency_high"] = result["network_latency_ms"] > 500
    result["error_high"] = result["error_rate"] > 0.05

    result["anomaly_count"] = (
        result[["cpu_high", "latency_high", "error_high"]]
        .sum(axis=1)
    )

    return result


def create_anomaly_summary(df):
    summary = pd.DataFrame(
        {
            "condition": [
                "CPU > 90%",
                "Latency > 500 ms",
                "Error rate > 0.05",
                "No threshold anomaly",
                "Two simultaneous anomalies",
                "Three simultaneous anomalies",
            ],
            "count": [
                int(df["cpu_high"].sum()),
                int(df["latency_high"].sum()),
                int(df["error_high"].sum()),
                int((df["anomaly_count"] == 0).sum()),
                int((df["anomaly_count"] == 2).sum()),
                int((df["anomaly_count"] == 3).sum()),
            ],
        }
    )

    summary["percentage"] = 100 * summary["count"] / len(df)
    summary.to_csv(OUTPUT_DIR / "anomaly_summary.csv", index=False)


# ---------------------------------------------------------------------------
# Distribution figures
# ---------------------------------------------------------------------------

def plot_core_distributions(df):
    plot_specs = [
        ("cpu_percent", "CPU utilisation (%)", "cpu_distribution.png"),
        ("memory_percent", "Memory utilisation (%)", "memory_distribution.png"),
        (
            "network_latency_ms",
            "Network latency (ms)",
            "latency_distribution.png",
        ),
        ("error_rate", "Error rate", "error_rate_distribution.png"),
    ]

    for column, xlabel, filename in plot_specs:
        values = df[column].dropna()

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(values, bins=60)
        ax.set_title(f"Historical HOMEPOT {xlabel}")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Number of observations")
        ax.grid(alpha=0.2)

        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / filename, dpi=180)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Correlation analysis
# ---------------------------------------------------------------------------

def create_correlation_analysis(df):
    correlations = df[ALL_METRICS].corr(method="pearson")
    correlations.to_csv(OUTPUT_DIR / "telemetry_correlations.csv")

    fig, ax = plt.subplots(figsize=(9, 7))

    image = ax.imshow(
        correlations.values,
        vmin=-1,
        vmax=1,
        aspect="auto",
    )

    ax.set_xticks(range(len(correlations.columns)))
    ax.set_xticklabels(
        correlations.columns,
        rotation=45,
        ha="right",
    )

    ax.set_yticks(range(len(correlations.index)))
    ax.set_yticklabels(correlations.index)

    ax.set_title("Historical HOMEPOT telemetry correlations")

    fig.colorbar(image, ax=ax, label="Pearson correlation")
    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / "telemetry_correlation_matrix.png",
        dpi=180,
    )
    plt.close(fig)


# ---------------------------------------------------------------------------
# Health-state comparison
# ---------------------------------------------------------------------------

def pair_health_and_telemetry(telemetry, health):
    telemetry_sorted = telemetry.sort_values(
        ["timestamp", "device_id"]
    ).copy()

    health_sorted = health.sort_values(
        ["timestamp", "device_id"]
    ).copy()

    telemetry_sorted["timestamp"] = pd.to_datetime(
        telemetry_sorted["timestamp"],
        utc=True,
    )

    health_sorted["timestamp"] = pd.to_datetime(
        health_sorted["timestamp"],
        utc=True,
    )

    paired_parts = []

    for device_id in sorted(health_sorted["device_id"].unique()):
        device_health = (
            health_sorted[health_sorted["device_id"] == device_id]
            .sort_values("timestamp")
            .copy()
        )

        device_telemetry = (
            telemetry_sorted[
                telemetry_sorted["device_id"] == device_id
            ]
            .sort_values("timestamp")
            .copy()
        )

        device_health = device_health.rename(
            columns={"timestamp": "health_timestamp"}
        )

        device_telemetry = device_telemetry.rename(
            columns={"timestamp": "metric_timestamp"}
        )

        paired_device = pd.merge_asof(
            device_health,
            device_telemetry,
            left_on="health_timestamp",
            right_on="metric_timestamp",
            direction="nearest",
            tolerance=pd.Timedelta(seconds=5),
            suffixes=("_health", "_metric"),
        )

        paired_parts.append(paired_device)

    paired = pd.concat(paired_parts, ignore_index=True)

    paired["time_difference_seconds"] = (
        paired["metric_timestamp"] - paired["health_timestamp"]
    ).abs().dt.total_seconds()

    return paired


def create_health_state_summary(paired):
    summary = (
        paired.groupby("health_state")[ALL_METRICS]
        .mean()
        .round(4)
    )

    summary["n"] = paired.groupby("health_state").size()
    summary.to_csv(OUTPUT_DIR / "health_state_telemetry_summary.csv")


def plot_health_state_comparisons(paired):
    metrics = [
        ("cpu_percent", "Mean CPU utilisation (%)"),
        ("disk_percent", "Mean disk utilisation (%)"),
        ("network_latency_ms", "Mean network latency (ms)"),
        ("error_rate", "Mean error rate"),
    ]

    for metric, ylabel in metrics:
        means = (
            paired.groupby("health_state")[metric]
            .mean()
            .sort_values()
        )

        fig, ax = plt.subplots(figsize=(9, 5))
        means.plot(kind="bar", ax=ax)

        ax.set_title(f"{ylabel} by health state")
        ax.set_xlabel("Health state")
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=25)

        fig.tight_layout()
        fig.savefig(
            OUTPUT_DIR / f"health_state_{metric}.png",
            dpi=180,
        )
        plt.close(fig)


# ---------------------------------------------------------------------------
# Temporal sampling analysis
# ---------------------------------------------------------------------------

def create_sampling_summary(df):
    ordered = df.sort_values(["device_id", "timestamp"]).copy()

    ordered["interval_seconds"] = (
        ordered.groupby("device_id")["timestamp"]
        .diff()
        .dt.total_seconds()
    )

    summary = (
        ordered.groupby("device_id")["interval_seconds"]
        .agg(["count", "min", "median", "mean", "max"])
        .round(4)
    )

    summary.to_csv(OUTPUT_DIR / "sampling_interval_summary.csv")

    intervals = ordered["interval_seconds"].dropna()

    # Restrict visualisation to active collection intervals.
    active = intervals[intervals <= 10]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(active, bins=60)
    ax.set_title("HOMEPOT active telemetry sampling intervals")
    ax.set_xlabel("Seconds between observations")
    ax.set_ylabel("Number of intervals")
    ax.grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / "sampling_interval_distribution.png",
        dpi=180,
    )
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Loading historical HOMEPOT data...")
    telemetry, health = load_data()

    telemetry["timestamp"] = pd.to_datetime(
        telemetry["timestamp"],
        utc=True,
    )

    print(f"Telemetry rows: {len(telemetry):,}")
    print(f"Health-check rows: {len(health):,}")

    if len(telemetry) != 70114:
        raise RuntimeError(
            f"Expected 70,114 main-population telemetry rows, "
            f"found {len(telemetry):,}."
        )

    if len(health) != 70114:
        raise RuntimeError(
            f"Expected 70,114 health-check rows, "
            f"found {len(health):,}."
        )

    print("Creating descriptive summary...")
    create_descriptive_summary(telemetry)

    print("Classifying historical anomaly regimes...")
    telemetry = classify_anomalies(telemetry)
    create_anomaly_summary(telemetry)

    print("Creating distribution figures...")
    plot_core_distributions(telemetry)

    print("Creating correlation analysis...")
    create_correlation_analysis(telemetry)

    print("Pairing health checks with telemetry...")
    paired = pair_health_and_telemetry(telemetry, health)

    matched = paired["metric_timestamp"].notna().sum()
    print(f"Successfully paired: {matched:,}/{len(health):,}")

    print("Creating health-state analysis...")
    create_health_state_summary(paired)
    plot_health_state_comparisons(paired)

    print("Creating sampling analysis...")
    create_sampling_summary(telemetry)

    print()
    print("EDA complete.")
    print(f"Outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()