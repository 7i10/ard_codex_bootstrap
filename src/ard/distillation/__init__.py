"""ImageNet adversarial-distillation verification tooling (plan 0103 Phase 2, batch D).

* ``crop_keys``: training batches that carry the exact crop key of every image.
* ``soft_label_bank``: FKD-style precomputed teacher soft labels on the exact
  deterministic training crops, and the bank-reading teacher stand-in.
* ``proxies``: cheap (student, teacher) compatibility proxies.
* ``teacher_sanity``: clean / PGD-10 / throughput sanity of a teacher.
"""
