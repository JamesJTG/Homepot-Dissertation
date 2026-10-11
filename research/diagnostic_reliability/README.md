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

### Dedicated experimental database configuration

The diagnostic-reliability experiment requires a separate PostgreSQL database named `homepot_dissertation_db`. It deliberately refuses to run against the normal HOMEPOT database, `homepot_db`.

Before executing Gate B or the main experiment, ensure PostgreSQL is running, the dedicated database exists, the required schema has been initialised, and the configured database user has appropriate permissions.

HOMEPOT supports a database URL override through the `DATABASE__URL` environment variable. For example:

```bash
export DATABASE__URL="postgresql://homepot_user:YOUR_PASSWORD@localhost:5432/homepot_dissertation_db"
```

Replace `YOUR_PASSWORD` with the password configured for your local PostgreSQL user. Do not commit credentials or a populated `.env` file.

The experiment must be executed from a clean Git checkout because it enforces a repository-integrity safeguard. Uncommitted changes and untracked files can prevent execution. Use a fresh Git clone for reproduction and separate run directories for new experiments.

A successful database connection does not, by itself, establish that the required experimental schema or evidence-validation services are ready.

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

From the repository root, with the Python virtual environment activated, execute:

```bash
PYTHONPATH=.:backend/src python -m pytest research/diagnostic_reliability/tests -q
```

The automated tests cover research components including evidence representation, scenario construction, experimental configuration, parsing, uncertainty measures, experiment storage and execution logic.

**Expected result:** The test suite should complete without failures.

**Verified result:** All 460 diagnostic-reliability research tests passed in a separate examiner-style Git checkout of commit `b8d4781` on macOS using a newly created Python 3.11.15 virtual environment.

The documented dependencies installed successfully, and the required research modules imported without errors.

```text
460 passed in 0.72s
```

This verifies fresh-environment dependency installation and automated research testing on macOS. It does not establish Windows compatibility or reproduce the complete Ollama-dependent experiment.

**Verified fresh-environment result:** The documented Python dependencies were successfully installed into a newly created Python 3.11.15 virtual environment in a separate examiner-style repository checkout on macOS.

All required research imports succeeded, and the complete automated research test suite passed:

```text
460 passed in 0.72s
```

This verifies dependency installation, research module imports and automated test execution in a fresh Python environment. It does not establish compatibility with every operating system or reproduce the complete Ollama-dependent experiment.

Any test failures in a new environment should be investigated rather than treated as successful reproduction.

### 4. Run the Gate B validation checkpoint
### 4. Run the Gate B validation checkpoint

Gate B verifies that the evidence validation mechanism distinguishes complete, fresh telemetry from deliberately degraded evidence.

Configure the dedicated dissertation PostgreSQL database before running the checkpoint:

```bash
export DATABASE__URL="postgresql://homepot_user:YOUR_PASSWORD@localhost:5432/homepot_dissertation_db"

PYTHONPATH=.:backend/src python -m research.diagnostic_reliability.gate_b_checkpoint
```

Replace `YOUR_PASSWORD` with the password configured for your PostgreSQL user.

Expected final success indicator:

```text
GATE B EXPERIMENTAL-VALIDITY CHECKPOINT: PASS
```

The checkpoint verifies four evidence conditions:

| Evidence condition | Expected validation result |
|---|---|
| `COMPLETE_FRESH` | Pass |
| `STALE` | Fail freshness |
| `INCOMPLETE` | Fail completeness |
| `DISCONTINUOUS` | Fail continuity |

**Verified result:** The checkpoint passed in both the separate reproduction checkout and the original development repository. All eight experimental-validity checks returned `true`.

Without the `DATABASE__URL` override, HOMEPOT may connect to its default `homepot_db` database, which may have an incompatible schema. Use the dedicated dissertation database rather than modifying an existing application database.

The checkpoint does not reproduce the complete LLM-dependent experiment.

**Database warning:** Without the `DATABASE__URL` override, HOMEPOT may connect to its default `homepot_db` database. An existing application database may have an incompatible schema. Do not reset or modify an existing database to resolve this issue. Use a separately configured research database instead.

This checkpoint exercises the research integration with HOMEPOT's evidence validation mechanism. It does not reproduce the complete Ollama-dependent experiment.

### 5. Prepare a small experimental run without LLM generation

The main experiment module supports a preparation-only mode that persists validation-controlled evidence cases without making LLM calls.
Ensure DATABASE__URL is configured as described in the dedicated experimental database section above.
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
Ensure DATABASE__URL is configured as described in the dedicated experimental database section above.
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
| Gate B checkpoint | `GATE B EXPERIMENTAL-VALIDITY CHECKPOINT: PASS` |
| Evidence preparation | Expected artefacts appear in the separate dry-run directory |
| LLM experiment | Experiment completes and produces durable run artefacts |
| Frozen result inspection | Tables, figures and manifest are present |
| Reporting regeneration | Scripts complete and regenerate the expected output files |

A fresh experiment is a new execution of the research method, not necessarily an exact numerical reproduction of the original results.

For exact reproduction of the reported analysis, the original frozen intermediate inputs, configuration and evaluation artefacts must be available.

### 10. Reproducibility boundaries

The dissertation's original experiment and evaluation were completed and frozen before preparation of the repository's publication documentation.

The repository supports several distinct levels of research verification and reproduction:

1. **Automated testing:** Running the research test suite to verify the behaviour of individual components and research workflows.
2. **Experimental evidence preparation:** Preparing and persisting diagnostic evidence cases using the dedicated experimental database, without invoking LLM generation.
3. **Experimental execution:** Running a new diagnostic-reliability experiment using the configured LLM and experimental infrastructure.
4. **Frozen-results inspection:** Examining the published dissertation tables, figures, manifests and intermediate analysis evidence.
5. **Reporting regeneration:** Recreating dissertation tables and figures from the frozen analysis inputs supplied in the repository.

These activities are not equivalent. In particular, successful reporting regeneration does not mean that the original LLM generations, reliability model fitting, model selection or statistical evaluation have been repeated.

#### Verified reproducibility results

A separate examiner-style Git checkout of commit `b8d4781` was tested on macOS using Python 3.11.15. The verification included a newly created virtual environment.

The following checks completed successfully:

- **Fresh Python environment:** The documented dependencies installed successfully in a new Python 3.11.15 virtual environment, and the required research modules imported without errors.
- **Automated tests:** All 460 diagnostic-reliability research tests passed in the fresh environment (`460 passed in 0.72s`).
- **Fresh PostgreSQL database:** A separate, initially empty PostgreSQL database was created and initialised using HOMEPOT's database service, producing 25 tables.
- **Fresh-database evidence preparation:** Experimental device provisioning and evidence preparation successfully produced four evidence cases for one scenario in the newly initialised database.
- **Dedicated research database:** The existing `homepot_dissertation_db` database was accessed successfully using the `DATABASE__URL` override.
- **Preparation-only experiment:** Eight evidence cases across two scenarios were successfully prepared and persisted without LLM generation.
- **Gate B checkpoint:** The checkpoint passed in the separate reproduction checkout. All eight experimental-validity checks returned `true`, including the expected validation failures for stale, incomplete and discontinuous evidence.
- **Reporting scripts:** Both dissertation reporting scripts executed successfully using the committed frozen analysis inputs.
- **Reporting completeness:** All 53 expected dissertation reporting artefacts were regenerated.
- **CSV outputs:** All 29 dissertation tables and five consolidated tables were byte-identical to their committed versions.
- **PNG figures:** All nine regenerated PNG figures were byte-identical to their committed versions.
- **PDF figures:** All nine regenerated PDF figures matched their committed versions after excluding creation and modification date metadata.
- **Generation manifest:** The regenerated results manifest was byte-identical to its committed version.

These results verify the documented Python environment, automated tests, fresh-database evidence preparation, Gate B validation behaviour and frozen-results reporting process in the tested macOS environment.

They do not establish reproduction of the complete original LLM-dependent experiment or downstream statistical evaluation from scratch.

#### Requirements and limitations

The following activities remain unverified:

- Running a new diagnostic-reliability experiment with Ollama and the `gemma3:12b` model in the separate reproduction checkout.
- Repeating the original full-scale experiment, including LLM generations and downstream statistical evaluation.
- Executing the reproduction workflow on Windows or Linux.
- Repeating the complete setup on a physically independent machine.

The fresh PostgreSQL integration test used a separate temporary database rather than the exact database name enforced by the main experimental runner. It therefore verifies database schema initialisation and evidence preparation, but not full experimental execution from a completely fresh database.

The original experiment's outputs are supplied as frozen research evidence. New LLM generations may produce different results because generation is not necessarily deterministic.

The reporting scripts require the frozen intermediate files under:

```text
research/diagnostic_reliability/analysis/outputs/
```

The necessary A8 research evidence files were included in commit `b8d4781`.

The experiment enforces two safeguards:

- **Dedicated database requirement:** The main experiment refuses to run unless the connected database is named `homepot_dissertation_db`.
- **Git working-tree integrity:** The experiment requires a clean Git checkout. Uncommitted modifications or untracked files may prevent execution.

Reporting regeneration may modify tracked PDF files because PDF creation and modification timestamps can change. Run experimental checks before regenerating reports, or use a separate clean checkout.

New experimental runs should use separate output directories to protect the frozen dissertation evidence.

The HOMEPOT application and dissertation research workflow are distinct. Running the complete application is not equivalent to reproducing the dissertation experiment.

These verification results apply to the tested macOS environment and should not be interpreted as independent verification across other operating systems or hardware configurations.

#### Requirements and limitations

The following activities remain unverified:

- Running a new diagnostic-reliability experiment with Ollama and the `gemma3:12b` model in the separate reproduction checkout.
- Repeating the original full-scale experiment, including all LLM generations and downstream statistical evaluation.
- Executing the reproduction workflow on Windows or Linux.
- Repeating the fresh database and Python installation procedures on a completely independent machine.

The fresh PostgreSQL integration test used a separate temporary database rather than the exact database name required by the main experimental runner. Therefore, it verifies schema initialisation and experimental evidence preparation, but not a complete fresh installation followed by full experimental execution.

The original experiment's outputs are provided as frozen research evidence. A new experimental run may produce different diagnostic responses and results because LLM generation is not necessarily deterministic.

The reporting scripts require the frozen intermediate files under:

```text
research/diagnostic_reliability/analysis/outputs/
```

The necessary A8 research evidence files were included in commit `b8d4781`, allowing the reporting scripts to regenerate the published dissertation tables and figures in the tested clean checkout.

The experiment enforces two safeguards:

- **Dedicated database requirement:** The main experiment refuses to run unless the connected PostgreSQL database is named `homepot_dissertation_db`.
- **Git working-tree integrity:** The experiment requires a clean Git checkout to protect experimental provenance. Uncommitted modifications or untracked files may prevent execution.

The reporting scripts may update tracked PDF files because PDF creation and modification timestamps can change. If reporting regeneration is performed before a new experimental run, the checkout's Git status should be checked and any unintended generated-file modifications resolved before starting the experiment.

New experimental runs should use separate output directories to avoid altering the original frozen research evidence.

The HOMEPOT application and the dissertation's experimental research workflow remain distinct. Installing or running the full HOMEPOT application is not equivalent to reproducing the dissertation experiment.

The verification results reported here are specific to the tested macOS environment and should not be interpreted as independent verification across all operating systems or hardware configurations.