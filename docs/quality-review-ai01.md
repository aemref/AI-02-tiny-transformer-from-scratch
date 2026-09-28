# Cross-Project Quality Review: AI-02 vs AI-01

Review date: 2026-09-28 (Europe/Istanbul)  
AI-01 reference: `e4c5477` / immutable `v1.0.0`  
AI-02 reviewed source: `50cb286`

This review uses AI-01's checked-in release checklist and evidence as the
portfolio baseline. It compares engineering controls, not model metrics: a
medical tabular classifier and a tiny language model do not share a meaningful
performance score.

## Evidence inspected

- Both repositories had clean `main` worktrees with zero local/remote divergence
  at the start of the daily run.
- AI-01's `v1.0.0` tag resolves to its current `main` commit, and its release
  evidence records 123 passing tests plus artifact, API, benchmark, and container
  gates.
- AI-02 collected 102 tests at the reviewed source and already had lint, concept
  experiment, training reproduction, licensed data, and measured loss evidence.
- AI-02's local demo tests passed, but the demo was not yet included in CI or
  documented from the README at the review point.

## Gate comparison

| Quality dimension | AI-01 v1.0.0 | AI-02 at review | Decision |
| --- | --- | --- | --- |
| Licensed local data | Data card, provenance, license, validation | Corpus license, provenance limits, batching tests | Equivalent for project scope |
| Leakage boundary | Train/validation/test roles and artifact digest checks | Document-level train/validation split; full-corpus vocabulary exposure disclosed | Accept with explicit limitation |
| Core behavior | 123 tests; model, artifact, HTTP, and drift paths | 102 collected tests; tensors through training and generation | Accept after full suite passes |
| Reproducible measurements | Comparison, drift, and HTTP evidence | Three full loss curves plus SVG reproduction | Accept; metrics are not cross-project comparable |
| Demo | JSON CLI and audited generated animation | Deterministic JSON CLI, no synthetic screenshot or user claim | Accept after CI and README integration |
| Architecture and risk | Architecture, model/data cards, trust boundaries | Architecture and model card added in this run | Accept after link audit |
| Packaging | Versioned app, artifact verifier, Docker smoke | Installable wheel with packaged CC0 corpus | Accept without Docker; AI-02 has no service boundary |
| Automated release gate | Machine-readable release readiness in CI | Missing at review point | Must add before release |
| Clean-clone evidence | Published in release record | Missing at review point | Must execute and record |
| Immutable release | Annotated tag and GitHub Release | Version `0.1.0`; no tag or release | Must complete last |

## Findings

### Required before AI-02 release

1. Make release readiness machine-readable and fail closed on the application
   version, corpus license, experiment artifacts, documentation, and README
   commands.
2. Add the demo, release check, and wheel build to CI so the default branch tests
   the same public paths described to users.
3. Upgrade the README from a progress summary to a release entry point with
   architecture, model-card, demo, measured-result, and limitation links.
4. Run lint, all 102+ tests, every executable experiment, demo, bytecode compile,
   evidence regeneration, and wheel inspection from a clean clone.
5. Record only observed results, then create `v1.0.0` after `origin/main` and the
   local branch are proven identical.

### Scope differences that are not defects

- AI-01 needs HTTP validation, latency, container, unprivileged-runtime, and
  artifact-integrity gates because it exposes an inference service and stored
  model. AI-02 deliberately exposes neither, so copying those gates would add
  theater rather than risk coverage.
- AI-02 needs causal-mask, tensor-shape, gradient, RNG-isolation, context-window,
  and experiment-reproduction checks that do not apply to AI-01.
- Neither repository's local measurements establish production capacity or
  real-world model quality.

## Review decision

AI-02 matches the released AI-01 baseline in test-first implementation, licensed
inputs, measured evidence, architecture disclosure, and explicit limitations.
It is **not release-ready at this review point** because automated release gates,
clean-clone evidence, version `1.0.0`, and immutable GitHub release objects remain
open. Those items are the only approved path to close the roadmap; no benchmark,
user feedback, or deployment claim may be invented to substitute for them.
