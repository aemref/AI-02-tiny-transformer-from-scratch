# v1.0.0 Release Evidence

Verification date: 2026-09-28 (Europe/Istanbul)  
Verified clean-clone source commit: `45a9073`
Application version: `1.0.0`

This record contains only locally executed results. It does not claim useful
language generation, external users, deployment, or production performance.

## Source and model verification

| Gate | Observed result |
| --- | --- |
| Ruff | Passed for the complete repository |
| Full test suite | 106 passed in 1.10 seconds |
| Six concept experiments | All exited successfully and emitted JSON |
| Training reproduction | JSON and SVG matched the checked-in files byte-for-byte |
| Bytecode compilation | Passed for `src` and `tests` |
| Wheel build | `tiny_transformer_from_scratch-1.0.0-py3-none-any.whl` |
| Wheel inspection | 26 files; Python modules, corpus, and corpus license present |
| Clean-clone wheel SHA-256 | `58a7ee447161bb904b5433b7258f8a3175d48adf179e073a04597652e203b47f` |
| Release verifier | All four checks passed; `ready: true` |
| Diff and worktree checks | Passed; clean clone had no tracked or untracked changes |

The wheel archive hash differs from the pre-evidence build because wheel ZIP
metadata contains build timestamps; the package name, version, 26-file inventory,
corpus, and corpus license were inspected in both builds. No wheel is attached to
the GitHub Release, so the source tag is the immutable release artifact.

## Measured demo path

The default 60-step lower-rate demo used the prompt `attention uses` and emitted:

```text
preserve masking masking layers layers layers
```

Its train loss moved from `12.077672958374023` to `1.6941132545471191`; final
validation loss was `9.639105796813965`. The JSON also reported six training
documents, six validation documents, vocabulary size 132, context window 6, and
the fixed configuration. This is deterministic software evidence on the local
environment, not a statement of language quality.

## Release decision

The implementation, documentation, full test suite, concept experiments,
training reproduction, demo, release verifier, compilation, package inspection,
diff check, and worktree check passed from a clean clone. The source is eligible
for the annotated `v1.0.0` tag and GitHub Release once this evidence update passes
the same local gates and `origin/main` is confirmed not to have diverged.
