## macOS: Research Setup, Execution and Verification

This section describes how to prepare and run the dissertation's diagnostic reliability research on macOS. It is separate from the installation instructions for the full HOMEPOT application.

The research workflow supports four distinct activities:

1. Running the automated research tests.
2. Preparing experimental evidence without generating LLM responses.
3. Running a new diagnostic reliability experiment.
4. Regenerating dissertation tables and figures from frozen analysis outputs.

These activities have different dependencies and must not be treated as equivalent forms of reproduction.

### 1. Prerequisites

- macOS and Terminal.
- Git.
- Python 3.11 or later.
- PostgreSQL, configured for HOMEPOT, for database-backed evidence preparation and experimental execution.
- Ollama, with the `gemma3:12b` model, for generating new diagnostic responses.

The general HOMEPOT AI configuration may specify a different model. The dissertation experiment uses `gemma3:12b`, as defined in its research configuration.

The frozen dissertation tables and figures can be inspected without running PostgreSQL or Ollama.

### 2. Install the Python environment

From the cloned repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r backend/requirements.txt
python -m pip install numpy pandas matplotlib
```

The additional scientific packages support the dissertation's analysis and reporting scripts.

Verify the principal imports:

```bash
PYTHONPATH=.:backend/src python -c "import sqlalchemy, ollama, yaml, numpy, pandas, matplotlib; import homepot.database; import research.diagnostic_reliability; print('Research Python imports: OK')"
```

Expected result:

```text
Research Python imports: OK
```

This verifies imports only. It does not establish database connectivity or confirm that the full experiment can execute.

### 3. Run the automated research tests

```bash
PYTHONPATH=.:backend/src python -m pytest research/diagnostic_reliability/tests -q
```

The tests cover research components including evidence representation, scenario construction, experimental configuration, parsing, uncertainty measures, experiment storage and execution logic.

**Verification:** pytest should finish with a passing test summary. Any failures must be investigated rather than treated as successful reproduction.

Some tests may require additional environment configuration. The complete suite has not yet been verified in a clean macOS installation.

### 4. Run the Gate B validation checkpoint

For a configured HOMEPOT PostgreSQL environment:

```bash
PYTHONPATH=.:backend/src python -m research.diagnostic_reliability.gate_b_checkpoint
```

Expected success indicator:

```text
M6.2 CHECKPOINT: PASS
```

This checkpoint exercises the research integration with HOMEPOT's evidence validation mechanism. It is not a substitute for running the complete experiment.

### 5. Prepare a small experimental run without LLM generation

The main experiment module supports a preparation-only mode that persists validation-controlled evidence cases without making LLM calls.

Use a separate directory to avoid changing the frozen dissertation run:

```bash
PYTHONPATH=.:backend/src python -m research.diagnostic_reliability.main_experiment \
  --run-dir research/diagnostic_reliability/outputs/reproduction_dry_run \
  --max-scenarios 2 \
  --prepare-only
```

This command requests evidence preparation for the first two frozen manifest scenarios.

**Verification:** inspect the specified run directory for newly created evidence and run artefacts. A successful preparation-only run does not establish that Ollama inference or downstream statistical analysis has succeeded.

The database and evidence-validation prerequisites must be configured before running this command.

### 6. Run a small experiment with Ollama

Install and start Ollama using the instructions for your macOS installation, then obtain the dissertation model:

```bash
ollama pull gemma3:12b
```

Verify that Ollama can access the model:

```bash
ollama list
```

With PostgreSQL and the research environment configured, run a small experiment in a separate directory:

```bash
PYTHONPATH=.:backend/src python -m research.diagnostic_reliability.main_experiment \
  --run-dir research/diagnostic_reliability/outputs/reproduction_llm_run \
  --max-scenarios 2
```

This invokes the actual research experiment rather than merely testing imports or preparing evidence.

**Verification:** check the terminal completion status and the generated run artefacts. The experiment reports `A8.5 run completed successfully.` on successful completion.

Model execution requires sufficient local computing resources. Results from a new run are not guaranteed to be numerically identical to the original frozen experiment.

**Do not use the original frozen run directory for exploratory reproduction.** The experiment supports resuming durable runs, so separate directories prevent accidental alteration of the original evidence.

### 7. Inspect the frozen dissertation results

The published results are organised under:

```text
research/diagnostic_reliability/dissertation_outputs/
├── tables/
├── figures/
└── manifests/
```

The tables contain the final reported diagnostic outcomes, reliability model evaluations, holdout comparisons, policy decisions and integrity summaries.

The figures provide visual representations of selected results in PNG and PDF formats.

The manifest records information about the dissertation results artefact generation.

Inspecting these files does not require rerunning the experiment.

### 8. Regenerate dissertation tables and figures

The repository contains two reporting scripts:

```text
research/diagnostic_reliability/analysis/generate_dissertation_results_artifacts.py

research/diagnostic_reliability/analysis/generate_dissertation_results_figures.py
```

Both scripts read intermediate analysis files from:

```text
research/diagnostic_reliability/analysis/outputs/
```

**These intermediate files are required.** The presence of the reporting scripts alone is insufficient to guarantee that a fresh repository checkout can regenerate the dissertation results.

When the frozen analysis inputs are available, the scripts can be invoked from the repository root:

```bash
PYTHONPATH=.:backend/src python \
  research/diagnostic_reliability/analysis/generate_dissertation_results_artifacts.py

PYTHONPATH=.:backend/src python \
  research/diagnostic_reliability/analysis/generate_dissertation_results_figures.py
```

The table generator reports completion and records the generated table count. The figure generator reports five main figures and four appendix figures, in PNG and PDF formats.

**Important:** these scripts write into the dissertation output directories. Preserve a copy of the frozen published outputs before executing them if the objective is to compare regenerated files against the originals.

Generating these tables and figures does not retrain the reliability estimators, rerun the LLM experiment or repeat model selection.

### 9. Verify results

Verification should be performed at the appropriate level:

| Activity | Verification |
|---|---|
| Environment installation | Required Python imports succeed |
| Automated testing | pytest reports passing tests |
| Gate B checkpoint | `M6.2 CHECKPOINT: PASS` |
| Evidence preparation | Expected artefacts appear in the separate dry-run directory |
| LLM experiment | Experiment completes and produces durable run artefacts |
| Frozen result inspection | Tables, figures and manifest are present |
| Reporting regeneration | Scripts complete and regenerate the expected output files |

A fresh experiment is a new execution of the research method, not necessarily an exact numerical reproduction of the original results.

For exact reproduction of the reported analysis, the original frozen intermediate inputs, configuration and evaluation artefacts must be available.

### 10. Reproducibility boundaries

The dissertation's original experiment and evaluation were completed and frozen before preparation of the publication documentation.

The instructions above identify the supported research entry points and their intended use. They have not yet been independently verified from a clean macOS checkout.

The original results should not be described as independently reproduced merely because the scripts are available.

In particular, successful regeneration of dissertation tables and figures depends on the availability of the frozen `analysis/outputs/` inputs. If those files are excluded from the published repository, readers can inspect the supplied final outputs but cannot regenerate them using the reporting scripts alone.

The HOMEPOT application and the dissertation's experimental research workflow are distinct. Installing the full application is not equivalent to reproducing the dissertation experiment.