"""FKD-style precomputed teacher soft labels on the exact training crops.

Format ``ard-soft-label-bank-v1`` (one directory per teacher x augmentation seed):

* ``source_ids.npy`` -- int64 ``[N]``, the sorted training-partition source IDs.
* ``epoch-XXX/topk_index.npy`` -- uint16 ``[N, K]`` class indices, descending probability.
* ``epoch-XXX/topk_prob.npy`` -- float16 ``[N, K]`` teacher probabilities ``softmax(z)``.
* ``epoch-XXX/residual_mass.npy`` -- float16 ``[N]``, ``max(0, 1 - sum(top-K FP32 probs))``;
  exactly 0 when K equals the class count.
* ``epoch-XXX/crop_keys.npy`` -- int32 ``[N, 6]`` ``(epoch, top, left, height, width, flip)``.
* ``epoch-XXX/meta.json`` -- per-file SHA-256, crop-key checksum, label-quality by-products.
* ``manifest.json`` (+ ``manifest.json.sha256``) -- identity and every epoch's file digests.
  Its SHA-256 is what ``distillation.bank.manifest_sha256`` pins.

Reconstruction (marginal smoothing): the residual mass is spread uniformly over
the ``C - K`` classes outside the top K, the row is renormalized in FP32 (fp16
rounding), and the RSLAD teacher-clean "logits" are ``log p``.  Because softmax
is shift-invariant, ``softmax(log softmax(z)) == softmax(z)``: with K = C the
bank reproduces the online target up to fp16 storage rounding plus the teacher
forward's own FP32/TF32 kernel noise (batch size and cuDNN algorithm differ
between the builder and a training run; ``build_environment`` records the
builder's flags), so an online-versus-bank comparison isolates the top-K
truncation.  A class whose
stored probability underflowed fp16 to exactly zero gets ``log`` of the
smallest normal FP32 number (probability ~1e-38) instead of ``-inf``.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

BANK_FORMAT = "ard-soft-label-bank-v1"
STORAGE = "top_k_marginal_smoothing_v1"
_ARRAYS = ("topk_index.npy", "topk_prob.npy", "residual_mass.npy", "crop_keys.npy")


class SoftLabelBankError(RuntimeError):
    """A bank is incomplete, altered, or does not belong to this run."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical(value: Any) -> Any:
    return json.loads(json.dumps(value, sort_keys=True, default=str))


def ids_sha256(ids: Sequence[int]) -> str:
    return hashlib.sha256(np.asarray(ids, dtype=np.int64).tobytes()).hexdigest()


# --------------------------------------------------------------------------
# Compression / reconstruction (pure functions, shared by bank and rslad_advt).
# --------------------------------------------------------------------------


def compress_probabilities(probabilities: torch.Tensor, top_k: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``[B, C]`` FP32 probabilities -> (uint16-range int64 idx, fp16 probs, fp16 residual)."""
    if probabilities.ndim != 2 or not probabilities.is_floating_point():
        raise ValueError("probabilities must be a floating [batch, class] tensor")
    classes = probabilities.shape[1]
    if not 1 <= top_k <= classes or classes > 65536:
        raise ValueError(f"top_k must lie in [1, {classes}] and classes must fit uint16")
    values, indices = probabilities.float().topk(top_k, dim=1, largest=True, sorted=True)
    if top_k == classes:
        residual = torch.zeros(probabilities.shape[0], dtype=torch.float32, device=probabilities.device)
    else:
        residual = (1.0 - values.sum(dim=1)).clamp_min(0.0)
    return indices, values.to(torch.float16), residual.to(torch.float16)


def reconstruct_probabilities(
    indices: torch.Tensor, probabilities16: torch.Tensor, residual16: torch.Tensor, *, num_classes: int
) -> torch.Tensor:
    """Marginal-smoothing reconstruction to a renormalized FP32 ``[B, C]`` distribution."""
    batch, top_k = indices.shape
    if probabilities16.shape != (batch, top_k) or residual16.shape != (batch,):
        raise ValueError("bank rows are misaligned")
    tail = num_classes - top_k
    if tail < 0:
        raise ValueError("top_k exceeds the class count")
    fill = residual16.float() / tail if tail > 0 else torch.zeros_like(residual16, dtype=torch.float32)
    if tail == 0 and bool((residual16 != 0).any()):
        raise SoftLabelBankError("a full-K bank row has non-zero residual mass")
    full = fill[:, None].expand(batch, num_classes).clone()
    full.scatter_(1, indices.long(), probabilities16.float())
    total = full.sum(dim=1, keepdim=True)
    if not bool(torch.isfinite(full).all()) or bool((total <= 0).any()):
        raise SoftLabelBankError("bank row reconstructs to a non-finite or empty distribution")
    return full / total


def pseudo_logits(probabilities: torch.Tensor) -> torch.Tensor:
    """``log p`` (shift-equivalent logits); zero mass maps to log(FP32 tiny), never -inf."""
    return probabilities.clamp_min(torch.finfo(torch.float32).tiny).log()


def truncate_like_bank(probabilities: torch.Tensor, top_k: int) -> torch.Tensor:
    """Apply the bank's exact storage + reconstruction to a live distribution (rslad_advt parity)."""
    indices, values16, residual16 = compress_probabilities(probabilities, top_k)
    return reconstruct_probabilities(indices, values16, residual16, num_classes=probabilities.shape[1])


# --------------------------------------------------------------------------
# Identity.
# --------------------------------------------------------------------------


def bank_identity(config: Any) -> dict[str, Any]:
    """Everything that determines the stored rows, from a validated ExperimentConfig."""
    dataset, training, seeds, teacher = config.dataset, config.training, config.seeds, config.teacher
    if teacher is None:
        raise SoftLabelBankError("a soft-label bank identity requires a teacher")
    if dataset.name != "imagenet" or dataset.imagenet_heavy_augmentation:
        raise SoftLabelBankError("soft-label banks require ImageNet without heavy (global-RNG) augmentation")
    view_size = dataset.image_size if training.train_image_size is None else training.train_image_size
    return _canonical(
        {
            "format": BANK_FORMAT,
            "dataset": {
                "name": dataset.name,
                "split": dataset.split,
                "num_classes": dataset.num_classes,
                "content_sha256": dataset.content_sha256,
                "derived_from": None if dataset.derived_from is None else dataset.derived_from.model_dump(mode="json"),
            },
            "training_view": {
                "transform": "EpochImageNetTransform(RandomResizedCrop+hflip, seed+1000003*epoch+10007*source_id)",
                "split_seed": seeds.split,
                "validation_fraction": training.validation_fraction,
                "augmentation_seed": seeds.augmentation,
                "view_image_size": view_size,
                "jpeg_draft_decode": training.jpeg_draft_decode,
                "imagenet_heavy_augmentation": False,
            },
            "teacher": {
                "source": teacher.source,
                "registry_id": teacher.registry_id,
                "architecture": teacher.architecture,
                "checkpoint_sha256": teacher.checkpoint_sha256,
                "normalization": teacher.normalization.model_dump(mode="json"),
            },
            "target": {
                "probabilities": "softmax(teacher_logits / 1.0), FP32 teacher forward, eval mode",
                "storage": STORAGE,
                "index_dtype": "uint16",
                "prob_dtype": "float16",
                "residual_dtype": "float16",
            },
        }
    )


# --------------------------------------------------------------------------
# Building.
# --------------------------------------------------------------------------


def _write_array(path: Path, array: np.ndarray) -> str:
    with path.open("wb") as handle:
        np.save(handle, array, allow_pickle=False)
    return sha256_file(path)


class BankEpochWriter:
    """Accumulate one epoch's rows (in ``source_ids`` order) and write it atomically.

    The epoch directory appears only on :meth:`close` (written to ``.tmp`` then
    renamed); an existing complete epoch is never overwritten.
    """

    def __init__(self, output_dir: Path, *, epoch: int, source_ids: np.ndarray, top_k: int, num_classes: int) -> None:
        self.final = output_dir / f"epoch-{epoch:03d}"
        if self.final.exists():
            raise SoftLabelBankError(f"bank epoch already exists (non-overwrite): {self.final}")
        self.tmp = output_dir / f"epoch-{epoch:03d}.tmp"
        if self.tmp.exists():
            shutil.rmtree(self.tmp)
        self.tmp.mkdir(parents=True)
        self.epoch, self.source_ids, self.top_k, self.num_classes = epoch, source_ids, top_k, num_classes
        count = source_ids.shape[0]
        self.index = np.empty((count, top_k), dtype=np.uint16)
        self.prob = np.empty((count, top_k), dtype=np.float16)
        self.residual = np.empty((count,), dtype=np.float16)
        self.keys = np.empty((count, 6), dtype=np.int32)
        self.position = 0
        self.correct = self.true_mass = self.coverage = 0.0
        self.pixel_sentinel: dict[str, Any] | None = None

    def add(
        self, sample_ids: torch.Tensor, labels: torch.Tensor, crop_keys: torch.Tensor, probabilities: torch.Tensor
    ) -> None:
        rows, position = sample_ids.shape[0], self.position
        if position + rows > self.source_ids.shape[0] or not np.array_equal(
            sample_ids.cpu().numpy().astype(np.int64), self.source_ids[position : position + rows]
        ):
            raise SoftLabelBankError("bank rows arrived out of source-ID order")
        if not bool((crop_keys[:, 0] == self.epoch).all()):
            raise SoftLabelBankError(f"crop keys were drawn for an epoch other than {self.epoch}")
        probabilities = probabilities.float()
        if probabilities.shape != (rows, self.num_classes) or not bool(torch.isfinite(probabilities).all()):
            raise SoftLabelBankError("teacher probabilities are misshapen or non-finite")
        indices, values16, residual16 = compress_probabilities(probabilities, self.top_k)
        self.index[position : position + rows] = indices.cpu().numpy().astype(np.uint16)
        self.prob[position : position + rows] = values16.cpu().numpy()
        self.residual[position : position + rows] = residual16.cpu().numpy()
        self.keys[position : position + rows] = crop_keys.cpu().numpy().astype(np.int32)
        labels_device = labels.to(probabilities.device)
        self.correct += float((probabilities.argmax(1) == labels_device).sum().item())
        self.true_mass += float(probabilities.gather(1, labels_device[:, None]).sum().item())
        self.coverage += float(probabilities.topk(self.top_k, dim=1).values.sum().item())
        self.position += rows

    def close(self) -> dict[str, Any]:
        count = self.source_ids.shape[0]
        if self.position != count:
            raise SoftLabelBankError(f"bank epoch {self.epoch} has {self.position} rows, expected {count}")
        tmp = self.tmp
        digests = {
            "topk_index.npy": _write_array(tmp / "topk_index.npy", self.index),
            "topk_prob.npy": _write_array(tmp / "topk_prob.npy", self.prob),
            "residual_mass.npy": _write_array(tmp / "residual_mass.npy", self.residual),
            "crop_keys.npy": _write_array(tmp / "crop_keys.npy", self.keys),
        }
        meta = {
            "epoch": self.epoch,
            "rows": count,
            "top_k": self.top_k,
            "num_classes": self.num_classes,
            "files": digests,
            "crop_key_checksum": hashlib.sha256(
                self.source_ids.astype(np.int64).tobytes() + self.keys.tobytes()
            ).hexdigest(),
            "teacher_top1_accuracy_on_crops": self.correct / count,
            "teacher_true_class_mass_on_crops": self.true_mass / count,
            "teacher_top_k_mass_coverage": self.coverage / count,
            "pixel_sentinel": self.pixel_sentinel,
        }
        (tmp / "meta.json").write_text(json.dumps(meta, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        os.replace(tmp, self.final)
        return meta


def write_bank_epoch(
    output_dir: Path,
    *,
    epoch: int,
    source_ids: np.ndarray,
    batches: Iterable[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
    top_k: int,
    num_classes: int,
) -> dict[str, Any]:
    """Write one epoch from ``(sample_ids, labels, crop_keys, teacher_probabilities)`` batches."""
    writer = BankEpochWriter(output_dir, epoch=epoch, source_ids=source_ids, top_k=top_k, num_classes=num_classes)
    for sample_ids, labels, crop_keys, probabilities in batches:
        writer.add(sample_ids, labels, crop_keys, probabilities)
    return writer.close()


def write_source_ids(output_dir: Path, source_ids: Sequence[int]) -> np.ndarray:
    array = np.asarray(source_ids, dtype=np.int64)
    if array.ndim != 1 or array.size == 0 or not bool((np.diff(array) > 0).all()):
        raise SoftLabelBankError("bank source IDs must be a non-empty strictly increasing sequence")
    path = output_dir / "source_ids.npy"
    if path.exists():
        existing = np.load(path, allow_pickle=False)
        if not np.array_equal(existing, array):
            raise SoftLabelBankError(f"existing {path} holds different source IDs")
        return array
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_array(path, array)
    return array


def finalize_bank(
    output_dir: Path,
    *,
    identity: Mapping[str, Any],
    epochs: Sequence[int],
    top_k: int,
    source_git_sha: str | None,
    build_environment: Mapping[str, Any] | None = None,
) -> str:
    """Verify every epoch and write ``manifest.json`` once; return its SHA-256."""
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        raise SoftLabelBankError(f"bank manifest already exists (non-overwrite): {manifest_path}")
    source_ids_path = output_dir / "source_ids.npy"
    source_ids = np.load(source_ids_path, allow_pickle=False)
    epoch_records: dict[str, Any] = {}
    num_classes = int(identity["dataset"]["num_classes"])
    for epoch in sorted(set(epochs)):
        directory = output_dir / f"epoch-{epoch:03d}"
        meta_path = directory / "meta.json"
        if not meta_path.is_file():
            raise SoftLabelBankError(f"bank epoch {epoch} is missing or incomplete: {directory}")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta["epoch"] != epoch or meta["top_k"] != top_k or meta["rows"] != int(source_ids.shape[0]):
            raise SoftLabelBankError(f"bank epoch {epoch} metadata disagrees with the bank")
        if meta["num_classes"] != num_classes:
            raise SoftLabelBankError(f"bank epoch {epoch} class count disagrees with the identity")
        for name in _ARRAYS:
            if sha256_file(directory / name) != meta["files"][name]:
                raise SoftLabelBankError(f"bank epoch {epoch} file {name} does not match its recorded digest")
        epoch_records[str(epoch)] = {"meta_sha256": sha256_file(meta_path), **meta}
    manifest = {
        "format": BANK_FORMAT,
        "identity": dict(identity),
        "top_k": top_k,
        "num_classes": num_classes,
        "epochs": sorted(set(epochs)),
        "rows": int(source_ids.shape[0]),
        "source_ids_sha256": sha256_file(source_ids_path),
        "train_ids_sha256": ids_sha256(source_ids.tolist()),
        "epoch_records": epoch_records,
        "source_git_sha": source_git_sha,
        # Provenance only (library versions, TF32/cuDNN flags, batch size): the
        # teacher forward is not bitwise reproducible across these, which is why
        # online-vs-bank parity is "fp16 storage rounding plus FP32/TF32 kernel noise".
        "build_environment": None if build_environment is None else dict(build_environment),
    }
    data = (json.dumps(_canonical(manifest), sort_keys=True, indent=2) + "\n").encode()
    manifest_path.write_bytes(data)
    digest = hashlib.sha256(data).hexdigest()
    (output_dir / "manifest.json.sha256").write_text(digest + "\n", encoding="utf-8")
    return digest


# --------------------------------------------------------------------------
# Reading.
# --------------------------------------------------------------------------


def read_bank_manifest(
    root: Path,
    *,
    expected_manifest_sha256: str,
    expected_identity: Mapping[str, Any],
    expected_top_k: int,
    required_epochs: int,
) -> dict[str, Any]:
    """Verify the manifest digest, identity, K and epoch coverage; return the manifest."""
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise SoftLabelBankError(f"soft-label bank manifest is missing: {manifest_path}")
    data = manifest_path.read_bytes()
    observed = hashlib.sha256(data).hexdigest()
    if observed != expected_manifest_sha256:
        raise SoftLabelBankError(
            f"soft-label bank manifest SHA-256 mismatch: expected {expected_manifest_sha256}, observed {observed}"
        )
    manifest = json.loads(data)
    if manifest.get("format") != BANK_FORMAT:
        raise SoftLabelBankError("unsupported soft-label bank format")
    expected = _canonical(dict(expected_identity))
    if manifest["identity"] != expected:
        differing = sorted(
            key
            for key in set(expected) | set(manifest["identity"])
            if manifest["identity"].get(key) != expected.get(key)
        )
        raise SoftLabelBankError(
            "soft-label bank was built for a different run identity (differs in: " + ", ".join(differing) + ")"
        )
    if int(manifest["top_k"]) != expected_top_k:
        raise SoftLabelBankError(f"bank top_k {manifest['top_k']} differs from configured {expected_top_k}")
    missing = sorted(set(range(required_epochs)) - set(manifest["epochs"]))
    if missing:
        raise SoftLabelBankError(f"bank lacks training epochs {missing[:5]}{'...' if len(missing) > 5 else ''}")
    return manifest


@dataclass
class _EpochArrays:
    index: np.ndarray
    prob: np.ndarray
    residual: np.ndarray
    keys: np.ndarray


class SoftLabelBank:
    """A verified, read-only bank bound to one training run's identity."""

    def __init__(self, root: Path, manifest: Mapping[str, Any], source_ids: np.ndarray) -> None:
        self.root = root
        self.manifest = manifest
        self.top_k = int(manifest["top_k"])
        self.num_classes = int(manifest["num_classes"])
        self.source_ids = source_ids
        self._epochs: dict[int, _EpochArrays] = {}

    @classmethod
    def open(
        cls,
        root: Path,
        *,
        expected_manifest_sha256: str,
        expected_identity: Mapping[str, Any],
        expected_top_k: int,
        required_epochs: int,
        train_ids: Sequence[int],
    ) -> SoftLabelBank:
        manifest = read_bank_manifest(
            root,
            expected_manifest_sha256=expected_manifest_sha256,
            expected_identity=expected_identity,
            expected_top_k=expected_top_k,
            required_epochs=required_epochs,
        )
        source_ids_path = root / "source_ids.npy"
        if sha256_file(source_ids_path) != manifest["source_ids_sha256"]:
            raise SoftLabelBankError("bank source_ids.npy does not match its manifest digest")
        source_ids = np.load(source_ids_path, allow_pickle=False)
        if not np.array_equal(source_ids, np.asarray(sorted(int(i) for i in train_ids), dtype=np.int64)):
            raise SoftLabelBankError("bank source IDs differ from this run's training partition")
        return cls(root, manifest, source_ids)

    def _arrays(self, epoch: int) -> _EpochArrays:
        cached = self._epochs.get(epoch)
        if cached is not None:
            return cached
        record = self.manifest["epoch_records"].get(str(epoch))
        if record is None:
            raise SoftLabelBankError(f"bank has no epoch {epoch}")
        directory = self.root / f"epoch-{epoch:03d}"
        # Verified once per epoch, on first use, before any row is served.
        if sha256_file(directory / "meta.json") != record["meta_sha256"]:
            raise SoftLabelBankError(f"bank epoch {epoch} meta.json was altered")
        for name in _ARRAYS:
            if sha256_file(directory / name) != record["files"][name]:
                raise SoftLabelBankError(f"bank epoch {epoch} file {name} was altered")
        arrays = _EpochArrays(
            index=np.load(directory / "topk_index.npy", mmap_mode="r", allow_pickle=False),
            prob=np.load(directory / "topk_prob.npy", mmap_mode="r", allow_pickle=False),
            residual=np.load(directory / "residual_mass.npy", mmap_mode="r", allow_pickle=False),
            keys=np.load(directory / "crop_keys.npy", mmap_mode="r", allow_pickle=False),
        )
        # Keep only the epoch in use mapped.
        self._epochs = {epoch: arrays}
        return arrays

    def probabilities(self, sample_ids: torch.Tensor, crop_keys: torch.Tensor, *, epoch: int) -> torch.Tensor:
        """Reconstructed FP32 teacher probabilities for one batch; refuses any crop-key mismatch."""
        arrays = self._arrays(epoch)
        ids = sample_ids.detach().cpu().numpy().astype(np.int64)
        positions = np.searchsorted(self.source_ids, ids)
        if bool((positions >= self.source_ids.shape[0]).any()) or not np.array_equal(self.source_ids[positions], ids):
            raise SoftLabelBankError("batch contains source IDs that are not in the bank")
        stored_keys = np.asarray(arrays.keys[positions], dtype=np.int64)
        batch_keys = crop_keys.detach().cpu().numpy().astype(np.int64)
        if batch_keys.shape != stored_keys.shape or not np.array_equal(batch_keys, stored_keys):
            bad = (
                int(np.flatnonzero((batch_keys != stored_keys).any(axis=1))[0])
                if batch_keys.shape == stored_keys.shape
                else 0
            )
            raise SoftLabelBankError(
                f"crop key mismatch at epoch {epoch} for source ID {int(ids[bad])}: "
                f"batch {batch_keys.tolist()[bad] if batch_keys.ndim == 2 else batch_keys.tolist()} "
                f"vs bank {stored_keys[bad].tolist()}"
            )
        device = sample_ids.device
        index = torch.from_numpy(np.asarray(arrays.index[positions], dtype=np.int64)).to(device)
        prob = torch.from_numpy(np.asarray(arrays.prob[positions])).to(device)
        residual = torch.from_numpy(np.asarray(arrays.residual[positions])).to(device)
        return reconstruct_probabilities(index, prob, residual, num_classes=self.num_classes)


class SoftLabelBankTeacher(nn.Module):
    """Teacher stand-in whose clean-crop output comes from a verified bank.

    ``clean_logits`` serves the RSLAD teacher-clean target (``log p``).  A
    pixel forward is delegated to ``online_teacher`` when one is given
    (``rslad_advt`` needs the teacher on the adversarial example) and refused
    otherwise, so no code path can silently run a teacher that is not there.
    """

    def __init__(
        self, bank: SoftLabelBank, *, online_teacher: nn.Module | None = None, sentinel_view: Any | None = None
    ) -> None:
        super().__init__()
        self.bank = bank
        # The crop-keyed training view of this run.  Crop keys pin the crop
        # geometry; the pixel sentinel additionally pins how the pixels are
        # produced (decoder, resize filter, library versions): on the first
        # batch of every epoch a few stored crops are re-drawn through this
        # run's own transform and must hash to the bank's.
        self.sentinel_view = sentinel_view
        self._sentinel_verified: set[int] = set()
        self.online_teacher = online_teacher
        if online_teacher is not None:
            for parameter in online_teacher.parameters():
                parameter.requires_grad_(False)
        super().train(False)

    def train(self, mode: bool = True) -> SoftLabelBankTeacher:
        super().train(False)
        if self.online_teacher is not None:
            self.online_teacher.train(False)
        return self

    @property
    def top_k(self) -> int:
        return self.bank.top_k

    @property
    def has_pixel_forward(self) -> bool:
        return self.online_teacher is not None

    def clean_logits(self, batch: Any, *, epoch: int) -> torch.Tensor:
        crop_keys = getattr(batch, "crop_keys", None)
        if crop_keys is None:
            raise SoftLabelBankError("bank-mode training batches must carry crop keys (CropKeyedSubset)")
        if self.sentinel_view is not None and epoch not in self._sentinel_verified:
            self.verify_pixel_sentinel(epoch)
        return pseudo_logits(self.bank.probabilities(batch.sample_ids, crop_keys, epoch=epoch)).detach()

    def verify_pixel_sentinel(self, epoch: int) -> None:
        record = self.bank.manifest["epoch_records"].get(str(epoch), {})
        stored = record.get("pixel_sentinel")
        if not stored:
            raise SoftLabelBankError(f"bank epoch {epoch} has no pixel sentinel")
        observed = compute_pixel_sentinel(self.sentinel_view, stored["source_ids"], epoch=epoch)
        if observed["uint8_sha256"] != stored["uint8_sha256"]:
            raise SoftLabelBankError(
                f"bank epoch {epoch} pixel sentinel mismatch: this run's transform produces different pixels for "
                "the same crop keys (decoder / resize / library version differs from the bank build)"
            )
        self._sentinel_verified.add(epoch)

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        if self.online_teacher is None:
            raise SoftLabelBankError(
                "this soft-label bank teacher has no pixel forward; only the clean training crops are in the bank"
            )
        return self.online_teacher(pixels)


SENTINEL_COUNT = 8


def pixel_sentinel_digest(images: torch.Tensor) -> str:
    """SHA-256 of uint8-quantized crops (``ToTensor`` pixels are k/255, so this is exact)."""
    quantized = (images.detach().float().cpu() * 255.0).round().clamp(0, 255).to(torch.uint8).contiguous()
    return hashlib.sha256(quantized.numpy().tobytes()).hexdigest()


def compute_pixel_sentinel(view: Any, source_ids: Sequence[int], *, epoch: int) -> dict[str, Any]:
    """Re-draw the crops of ``source_ids`` through a crop-keyed training view at ``epoch``."""
    transform = view.transform
    if transform.epoch != epoch:
        raise SoftLabelBankError(f"training transform is at epoch {transform.epoch}, sentinel requested for {epoch}")
    images = torch.stack([view.dataset[int(source_id)][0] for source_id in source_ids])
    return {"source_ids": [int(i) for i in source_ids], "uint8_sha256": pixel_sentinel_digest(images)}


def entropy(probabilities: torch.Tensor) -> torch.Tensor:
    return -(probabilities * probabilities.clamp_min(torch.finfo(torch.float32).tiny).log()).sum(dim=1)


def kl_rows(target: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """Per-row ``KL(target || reference)`` for probability rows (FP32)."""
    return F.kl_div(reference.clamp_min(torch.finfo(torch.float32).tiny).log(), target, reduction="none").sum(dim=1)
