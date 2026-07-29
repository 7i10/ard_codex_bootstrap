# Cross-host evaluation checkpoint portability

Date: 2026-07-29

## Symptom

A Bartoldson/Entropy AutoAttack moved from Hamster to Ferret failed before evaluation because the resolved training
config contained Hamster's absolute teacher checkpoint path. The checkpoint bytes and registered SHA-256 were identical.

## Root cause and correction

Three layers incorrectly treated the absolute storage path as scientific identity:

1. evaluation-versus-training teacher equality;
2. production RobustBench preflight;
3. tracker teacher-lineage collection.

Evaluation now permits the checkpoint path alone to differ. The local path is used for registry and byte-hash
preflight and is recorded in the evaluation manifest. The original resolved training config and its hash remain the
canonical checkpoint lineage. Architecture, registry ID, preprocessing, normalization, threat model, and
`checkpoint_sha256` must still match exactly.

Focused tests cover same-SHA relocation, different-SHA rejection, local preflight selection, and local teacher-lineage
selection. Partial preflight directories were archived rather than overwritten.
