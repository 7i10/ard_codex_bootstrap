"""Datasets whose batches retain stable source sample identifiers."""

from .datasets import SyntheticCIFAR, TinyImageNetDataset, build_dataset, stratified_train_validation_split
from .indexed import EpochShuffleSampler, IndexedBatch, IndexedDataset, SampleRef, collate_indexed

__all__ = [
    "EpochShuffleSampler",
    "IndexedBatch",
    "IndexedDataset",
    "SampleRef",
    "SyntheticCIFAR",
    "TinyImageNetDataset",
    "build_dataset",
    "collate_indexed",
    "stratified_train_validation_split",
]
