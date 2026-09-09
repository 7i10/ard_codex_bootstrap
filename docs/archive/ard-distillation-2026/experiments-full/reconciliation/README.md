# Terminal reassignment evidence

These records reconcile completed cross-host work into each job's canonical
owning-host state. They are operational lineage, not new training or
evaluation results.

| Record | Owning host | Execution host | Evidence SHA-256 |
|---|---|---|---|
| `prod-h1-chen-entropy-s0.json` | Hamster | Hamster | `3a078605b79242d4a3f7f4a44fd264799ddd71e915c26357e841a6338a0a039d` |
| `prod-h1-bart-entropy-s0.json` | Hamster | Ferret | `0a7530c8de8d0f0eeb5b65b1a99cdd4f149b383eabf66f95bad901a202a96399` |
| `prod-f0-bart-rslad-s0.json` | Ferret | Ferret | `84337c37011d66211e56ea037434c0499b059bc78b82938275c263da8e9bf192` |
| `prod-f1-bart-joint-s0.json` | Ferret | Ferret | `13a9a0f97baa5442fecf009e828bb4574fa33a35d7967178ce32b6cc3c072696` |
| `prod-f0-chen-student-s0.json` | Ferret | Hamster | `2c6e7f283ba779f18ec0f6116fd854debcb6d86d8c2b695985f685ededa39c3e` |

Each document binds the scientific Git SHA, campaign identity, source and
execution host/GPU UUID, ordered successful phase events, sequence files,
prior canonical phase exits, best/last checkpoints, and evaluation result
digests. Chen/Student's later AutoAttack remains auxiliary evidence because
its campaign contract ended at PGD.

The Hamster and Ferret batch transaction IDs are respectively
`b2bb6bbaf4c83d3a6d58dd34d7313b2bcec0e4425765fe46acd83884fe685c99`
and
`6c1a629104c741dd6e646ac5e73ca719e6721e2233fa9426224b38012d39acda`.
Both journals are completed, exact re-import is a no-op, and the prior job
records remain in the host-local reassignment archives.

The AutoAttack source identity attached to these legacy results is explicitly
post-hoc. The immutable results still record library version `unknown`; see
[`../0002-autoattack-provenance-amendment.json`](../0002-autoattack-provenance-amendment.json)
for the bounded claim and limitation.
