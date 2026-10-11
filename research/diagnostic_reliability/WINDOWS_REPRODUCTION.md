# Windows Research Reproduction Guide

## Scope

This guide provides Windows PowerShell instructions for setting up and verifying the HOMEPOT diagnostic-reliability dissertation research.

**Verification status:** These Windows instructions have not been independently executed on Windows. The equivalent Python installation, research tests, PostgreSQL evidence preparation, Gate B checkpoint and reporting regeneration were successfully verified on macOS.

The guide focuses on the dissertation research workflow rather than installing the complete HOMEPOT application.

## 1. Prerequisites

Install the following software:

- Git
- Python 3.11 (64-bit)
- PostgreSQL, with the PostgreSQL server running
- Ollama, only if performing new LLM-dependent experimental runs

For new LLM-dependent runs, the research configuration uses the `gemma3:12b` model. Ensure that the machine has sufficient memory and computational resources.

Open **PowerShell** and clone the repository:

```powershell
git clone https://github.com/JamesJTG/Homepot-Dissertation.git
cd Homepot-Dissertation
```

## 2. Create the Python environment

From the repository root:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r backend\requirements.txt
.\.venv\Scripts\python.exe -m pip install numpy pandas matplotlib
```

These commands invoke the virtual environment's Python interpreter directly, avoiding the need to change PowerShell's script execution policy.

## 3. Configure the dedicated PostgreSQL database

Create a separate PostgreSQL database named:

```text
homepot_dissertation_db
```

Use an appropriately authorised PostgreSQL account to create the database and grant the research user the permissions required to create tables and write experimental evidence.

For example, after configuring a PostgreSQL user named `homepot_user`, execute the following SQL as an administrator:

```sql
CREATE DATABASE homepot_dissertation_db OWNER homepot_user;
```

Set the database connection for the current PowerShell session:

```powershell
$env:DATABASE__URL = "postgresql://homepot_user:YOUR_PASSWORD@localhost:5432/homepot_dissertation_db"
```

Replace `YOUR_PASSWORD` with the configured PostgreSQL password.

The research code uses HOMEPOT's database service to initialise the required schema. The fresh-database initialisation and evidence-preparation path was verified on macOS, but has not been independently verified on Windows.

**Do not run destructive database reset or seed scripts against an existing dissertation database.** Use a newly created, dedicated research database.

## 4. Run the automated research tests

From the repository root:

```powershell
$env:PYTHONPATH = ".;backend/src"
.\.venv\Scripts\python.exe -m pytest research/diagnostic_reliability/tests -q
```

The verified macOS result was:

```text
460 passed
```

Windows results may differ and should be recorded rather than assumed.

## 5. Run the Gate B checkpoint

With `DATABASE__URL` and `PYTHONPATH` configured:

```powershell
.\.venv\Scripts\python.exe -m research.diagnostic_reliability.gate_b_checkpoint
```

Expected final success indicator:

```text
GATE B EXPERIMENTAL-VALIDITY CHECKPOINT: PASS
```

The checkpoint tests the complete/fresh, stale, incomplete and discontinuous evidence conditions. It does not require Ollama.

## 6. Prepare experimental evidence without LLM generation

Use a separate output directory:

```powershell
.\.venv\Scripts\python.exe -m research.diagnostic_reliability.main_experiment --run-dir research/diagnostic_reliability/outputs/windows_reproduction_dry_run --max-scenarios 2 --prepare-only
```

The verified macOS run prepared eight evidence cases across two scenarios without performing LLM generation.

The main experiment requires a clean Git working tree and a database named exactly `homepot_dissertation_db`.

## 7. Inspect or regenerate dissertation results

The repository contains frozen analysis evidence and dissertation reporting outputs.

To regenerate the tables and figures from the frozen analysis inputs:

```powershell
.\.venv\Scripts\python.exe research/diagnostic_reliability/analysis/generate_dissertation_results_artifacts.py
.\.venv\Scripts\python.exe research/diagnostic_reliability/analysis/generate_dissertation_results_figures.py
```

The verified macOS regeneration produced 53 reporting artefacts: 34 CSV files, nine PNG figures, nine PDF figures and one results manifest.

PDF creation and modification timestamps may differ between executions. Reporting regeneration may modify tracked PDF files and make the Git working tree dirty. If planning to execute the main experiment, perform its preparation or execution checks before regenerating the reports, or use a separate clean checkout.

## 8. New Ollama-dependent experimental runs

A new experimental run requires a working Ollama installation and the configured research model.

For example:

```powershell
ollama pull gemma3:12b
```

Follow the research README for the experimental execution command and configuration.

**Verification limitation:** A complete new Ollama-dependent experiment has not been independently reproduced in the examiner-style checkout. The original frozen results should not be presented as guaranteed outputs of a new run.

## 9. Reproducibility boundaries

Successful research tests, Gate B validation and reporting regeneration provide different levels of verification.

They do not independently reproduce all original LLM generations, reliability-model fitting, statistical analyses or the complete experimental evaluation.

Windows execution remains unverified. Any platform-specific installation, database, path or model-execution issues encountered during reproduction should be documented rather than silently treated as successful verification.