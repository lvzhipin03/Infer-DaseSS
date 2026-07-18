# Project Documentation Redesign

## Goal

Make the repository understandable and runnable on a new development machine or GPU server without relying on the current server's absolute paths or prior conversation context. Keep benchmark evidence separate from the operational project guide.

## Root README

`README.md` is the single entry point for the project. It will contain:

1. Project purpose and current Phase 2 status.
2. A system architecture diagram showing the shared handwritten Qwen2 forward used by toy and real weights.
3. A repository map that assigns one responsibility to each major directory and entry point.
4. The external-checkpoint contract: real weights are not committed, required files are listed, and `MODEL_PATH` selects the checkpoint on every server.
5. Dependency boundaries: minimal toy, real inference, optional Transformers oracle, and public benchmark dependencies.
6. Copy-paste workflows for toy inference, real inference, correctness validation, oracle comparison, smoke benchmark, and full benchmark.
7. A portable environment-variable template using `PROJECT_ROOT`, `MODEL_PATH`, `PYTHON_BIN`, and optional SSH variables.
8. The local-development/server-execution workflow, including exact-source synchronization exclusions.
9. Current implementation boundaries and links to architecture, verification, plan, and benchmark documents.

Commands must quote variables, must not embed credentials, and must mark the current `/ai/...` paths as examples rather than required locations.

## Benchmark Results Document

Create `docs/benchmark-results.md` as the evidence record. It will contain:

1. The accepted full-run score and its exact result directory.
2. Hardware, software, dtype, attention backend, model path, benchmark profile, workload, and timing configuration.
3. The exact no-`--limit` command used for the accepted run.
4. Score components and suite metrics, including decode batch 1/2/4 values extracted from `final_summary.json`.
5. Correctness/runtime interpretation against the bundled public baseline.
6. A run history distinguishing the one-case smoke, the externally interfered first full run, the serving-only recovery check, and the accepted final full run.
7. A reproducibility checklist and the paths of `final_summary.json`, text summary, and CSV evidence.

The document will state explicitly that benchmark scripts, public data, baseline summary, and scoring rules were not changed.

## Validation

- Every referenced local path must exist.
- Every command must use the documented variables consistently.
- The accepted metrics must be read from the server's retained `final_summary.json`, not copied only from terminal output.
- README links must resolve to repository files.
- `git diff --check` and the repository test/static-validation commands must remain clean.
