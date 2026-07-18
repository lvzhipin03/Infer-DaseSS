# Project Documentation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the stale root README with a portable project/run guide and add a separate evidence-backed benchmark results document.

**Architecture:** Keep operational documentation in `README.md` and measured performance evidence in `docs/benchmark-results.md`. Use shell variables for every machine-specific path, link the two documents, and source accepted metrics from the retained server `final_summary.json`.

**Tech Stack:** Markdown, Bash command examples, Git, server JSON/CSV benchmark artifacts.

## Global Constraints

- Do not modify benchmark scripts, public data, baseline summary, scoring rules, or source implementation as part of this documentation task.
- Do not place SSH passwords, private-key passphrases, or other credentials in Git.
- Treat `/ai/projects/Infer-DaseSS` and `/ai/llm/models/models/Qwen/Qwen2.5-0.5B-Instruct` as current-server examples, not required paths.
- Quote shell variables and make a new server runnable by changing `PROJECT_ROOT`, `MODEL_PATH`, and `PYTHON_BIN` only.
- Record the accepted no-`--limit` run separately from smoke and failed/interfered diagnostic runs.

---

### Task 1: Capture authoritative benchmark evidence

**Files:**
- Read on server: `student_release/results/phase2_full_retry_20260718/final_summary.json`
- Read on server: `student_release/results/phase2_full_retry_20260718/final_summary.txt`
- Read on server: `student_release/results/phase2_full_retry_20260718/student/results.csv`

**Interfaces:**
- Produces: verified environment, score components, suite metrics, decode batch 1/2/4 metrics, and artifact paths for Task 3.

- [ ] **Step 1: Read the accepted summary JSON**

Use a read-only Python command on the server to print `overall`, every suite's primary summary, and `decode_throughput.by_batch_size`.

- [ ] **Step 2: Cross-check the text summary and artifacts**

Confirm `FINAL SCORE: 86.03 / 100`, `limit=None`, `timed_repeats=3`, six suites, runtime success 1.0, and that JSON, text, and CSV files exist.

---

### Task 2: Rewrite the portable root guide

**Files:**
- Modify: `README.md`

**Interfaces:**
- Consumes: repository entry points and the external checkpoint contract.
- Produces: a new-server quick start driven by `PROJECT_ROOT`, `MODEL_PATH`, and `PYTHON_BIN`.

- [ ] **Step 1: Replace the phase-oriented README structure**

Write sections for project purpose/status, architecture, repository map, toy versus real weights, dependencies, and implementation boundaries. Link `docs/images/qwen-system-architecture.png` and explain eager versus SDPA and left-padded KV-cache generation.

- [ ] **Step 2: Add portable configuration and workflows**

Define:

```bash
export PROJECT_ROOT=/path/to/Infer-DaseSS
export MODEL_PATH=/path/to/Qwen2.5-0.5B-Instruct
export PYTHON_BIN="$PROJECT_ROOT/.venv-real/bin/python"
```

Add exact install, toy run, real run, static validator, Transformers oracle, smoke, and full benchmark commands using those variables.

- [ ] **Step 3: Add local-development/server-execution guidance**

Document generic SSH/SCP variables and exact exclusions for `.git`, virtual environments, caches, `references`, ZIP packages, and benchmark results. Never include credentials.

---

### Task 3: Create the benchmark evidence record

**Files:**
- Create: `docs/benchmark-results.md`
- Modify: `README.md`

**Interfaces:**
- Consumes: Task 1 metrics.
- Produces: the accepted Phase 2 score record linked from the README.

- [ ] **Step 1: Record reproducible environment and command**

Document A800 80GB, Python 3.11, PyTorch 2.10.0+cu128, FP16, SDPA, external checkpoint, six suites, public defaults, `timed_repeats=3`, process isolation, and no `--limit`.

- [ ] **Step 2: Record scores and run history**

Add the 86.03 accepted score table, suite diagnostics, exact artifact paths, and a history table distinguishing smoke 84.28, externally interfered full 76.33, serving recovery, and accepted full 86.03. State that benchmark code/data/scoring were unchanged.

- [ ] **Step 3: Link the evidence document from README**

Keep only the accepted score summary in root README and link to `docs/benchmark-results.md` for details.

---

### Task 4: Verify and commit documentation

**Files:**
- Verify: `README.md`
- Verify: `docs/benchmark-results.md`

**Interfaces:**
- Produces: committed portable documentation with valid links and commands.

- [ ] **Step 1: Validate repository paths and formatting**

Run:

```bash
test -f docs/images/qwen-system-architecture.png
test -f docs/benchmark-results.md
test -f student_release/scripts/run_inference_benchmark.py
rg -n 'TBD|TODO|FIXME|dase314@AI|InferOpt_2026' README.md docs/benchmark-results.md
git diff --check
```

Expect every `test` to succeed, no secret/placeholder matches, and a clean diff check.

- [ ] **Step 2: Re-run documentation-adjacent project gates**

```bash
python3 student_release/scripts/validate_engine.py --skip-load
python3 -m unittest tests.test_verification_cli tests.test_real_cli -v
```

Expect the static validator and focused CLI tests to pass.

- [ ] **Step 3: Commit**

```bash
git add README.md docs/benchmark-results.md
git commit -m "docs: add portable run guide and benchmark results"
```

