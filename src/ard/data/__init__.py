"""Datasets whose batches retain stable source sample identifiers."""

from .datasets import (
    EpochSourceTransform,
    SyntheticCIFAR,
    TinyImageNetDataset,
    build_dataset,
    build_raw_dataset,
    build_train_validation_views,
    stratified_train_validation_split,
)
from .indexed import EpochShuffleSampler, IndexedBatch, IndexedDataset, SampleRef, collate_indexed

__all__ = [
    "EpochShuffleSampler",
    "EpochSourceTransform",
    "IndexedBatch",
    "IndexedDataset",
    "SampleRef",
    "SyntheticCIFAR",
    "TinyImageNetDataset",
    "build_dataset",
    "build_raw_dataset",
    "build_train_validation_views",
    "collate_indexed",
    "stratified_train_validation_split",
]
