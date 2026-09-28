# v1.0.0 Release Evidence

Verification date: 2026-09-28 (Europe/Istanbul)  
Verified source commit: `684339e`  
Application version: `1.0.0`

This record contains only locally executed results. It does not claim useful
language generation, external users, deployment, or production performance.

## Source and model verification

| Gate | Observed result |
| --- | --- |
| Ruff | Passed for the complete repository |
| Behavioral suite before evidence record | 102 passed |
| Six concept experiments | All exited successfully and emitted JSON |
| Training reproduction | JSON and SVG matched the checked-in files byte-for-byte |
| Bytecode compilation | Passed for `src` and `tests` |
| Wheel build | `tiny_transformer_from_scratch-1.0.0-py3-none-any.whl` |
| Wheel inspection | 26 files; Python modules, corpus, and corpus license present |
| Wheel SHA-256 | `8a7e051a9adfec72390593018917cc5b01b7d2865d5d711fe0bce06883d06570` |
| Diff and worktree checks | Passed; worktree clean and only ahead of `origin/main` |

The 102-test run intentionally excluded the four release-contract tests because
this evidence document did not exist yet. The final clean-clone pass must include
all 106 tests and the machine-readable release command after this record is
committed.

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

The implementation, documentation, experiment, demo, compilation, and packaging
gates passed on committed source. Release remains pending until this evidence
record itself passes all 106 tests and every documented command from a clean
clone. The annotated tag and GitHub Release must not be created before that final
gate succeeds.
