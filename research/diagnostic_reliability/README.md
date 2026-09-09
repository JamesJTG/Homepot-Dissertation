# Diagnostic Reliability Experiment Harness

This package is the dissertation-specific experimental extension for HOMEPOT.
It is intentionally isolated from the production `/ai/query` path.

## M6 vertical slice

Implemented:

- five-class controlled fault taxonomy plus `UNKNOWN` diagnostic decision;
- deterministic scenario generation with fixed seeds;
- evidence-quality transformations separated from fault generation;
- Gate B configuration reuse for freshness/completeness/continuity manipulations;
- model-visible context that excludes private ground truth and answer-bearing anomaly labels;
- fixed experimental prompt and strict JSON response schema;
- snapshot data structure with private ground-truth metadata separated from `model_context`;
- objective correctness / decision-appropriateness scoring;
- repeated-generation runner using an injectable LLM callable;
- diagnostic disagreement entropy and modal-fraction calculations;
- unit tests that do not require a live HOMEPOT database or Ollama.

## Next integration step

M6.2 will connect the vertical slice to a HOMEPOT database session and the real
validation envelope, materialise `COMPLETE_FRESH` and `STALE` telemetry, assert
Gate B PASS/FAIL behaviour, freeze the resulting context, and then attach the
local Ollama service for the first pilot generations.

## M6.2 real Gate B integration

`validation_integration.py` bridges the controlled scenarios to HOMEPOT's real
`DeviceMetrics` and `HealthCheck` tables and invokes the production
`DataIntegrityGate` with thresholds loaded from `ai/config.yaml`.  The module
keeps fault generation and evidence-quality manipulation outside Gate B, so the
experiment can verify the intended chain `Q -> Gate B -> E` without implementing
a substitute validator.

The integration requires the normal HOMEPOT backend/database dependencies and
is intentionally separate from the lightweight core tests.  A local integration
checkpoint will be provided once the dedicated experiment-device fixture/CLI is
complete.

### Local M6.2 checkpoint

After the normal HOMEPOT backend dependencies and PostgreSQL database are
available, run from the repository root:

```bash
PYTHONPATH=.:backend/src python -m research.diagnostic_reliability.gate_b_checkpoint
```

Expected final line:

```text
M6.2 CHECKPOINT: PASS
```

This proves, using HOMEPOT's production Gate B and real ORM tables, that the
same ground-truth CPU-saturation scenario passes when fresh and fails the
freshness check when deliberately made stale.
