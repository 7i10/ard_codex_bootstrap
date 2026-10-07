"""Pinned ImageNet-1k robust teacher profiles (plan 0103 Phase 2, batch D).

Each profile fixes one external or in-project checkpoint by SHA-256, the
architecture that strict-loads it, the parameter count, the teacher's own
input normalization, and its threat model (Linf 4/255 in pixel space).  Inputs
reach every teacher as float pixels in ``[0, 1]`` at 224 px; the teacher's own
normalization is applied inside :class:`ard.models.teacher.TeacherAdapter`
(``preprocessing_owner: teacher_adapter``), never by the data pipeline.

Checkpoint files are never downloaded from here and never live in the
repository: a config names the local file and its digest, the digest must equal
the profile's, and the bytes that are hashed are the bytes deserialized.

Architecture notes
------------------
* ``salman2020_resnet50_linf_eps4``: torchvision ResNet-50.  The MadryLab
  ``robustness`` checkpoint stores the classifier under ``module.model.``, its
  input normalizer under ``module.normalizer.`` and an identical copy for the
  attacker under ``module.attacker.model.``; only ``module.model.`` is loaded,
  after checking the normalizer equals the ImageNet mean/std and the attacker
  copy equals the classifier tensor for tensor.
* ``singh2023_*_convstem``: Singh, Croce & Hein (NeurIPS 2023,
  arXiv:2303.01870).  The ConvStem modules below reproduce the structure of
  ``robustbench/model_zoo/architectures/convstem_models.py`` (RobustBench,
  MIT licence, pinned at 78fcc9e) so the upstream ``base_model.`` state dict
  strict-loads; the upstream repository ``nmndeep/revisiting-at`` ships no
  licence file, which is recorded in each profile.  ConvNeXt-T/B checkpoints
  were trained without input normalization (``params.json``
  ``model.add_normalization: 0``; RobustBench also evaluates them on raw
  pixels), so their profile is ``imagenet_raw_identity``.  The ViT-S
  checkpoint embeds its normalizer (``add_normalization: 1``): the ImageNet std
  and a mean within 5e-5 of the ImageNet mean; the adapter applies exactly those
  embedded FP32 values (``custom`` profile).
* ``ard_mobilenetv4_conv_medium_phase1_random``: this project's own PGD-AT
  MobileNetV4-Conv-Medium (run ``plan0103-phase1-mobilenetv4-conv-medium-random-v1``,
  ``last.pt``), the same-family teacher for MobileNetV4-Conv-Small.
"""

from __future__ import annotations

import hashlib
import io
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch
from torch import nn

from ard.config.schema import NormalizationConfig, TeacherConfig

IMAGENET_TEACHER_THREAT = {"norm": "linf", "epsilon": "4/255", "input_domain": "pixel_0_1"}
IMAGENET_TEACHER_RESOLUTION = 224

StateDictFormat = Literal["madrylab_robustness", "singh2023_base_model", "ard_training_checkpoint"]


class ImageNetTeacherRegistryError(RuntimeError):
    """An ImageNet teacher config or checkpoint violates its pinned profile."""


@dataclass(frozen=True)
class ImageNetTeacherSpec:
    registry_id: str
    architecture: str
    upstream_model_id: str
    source_url: str
    license: str
    checkpoint_filename: str
    checkpoint_sha256: str
    checkpoint_bytes: int
    expected_parameter_count: int
    normalization_profile: Literal["imagenet_standard", "imagenet_raw_identity", "custom"]
    # The interpolation the upstream evaluation uses for its resize; recorded
    # for the sanity script.  Training crops are always the student's own crops.
    native_eval_interpolation: Literal["bilinear", "bicubic"]
    state_dict_format: StateDictFormat
    published_clean_accuracy: float | None
    published_robust_accuracy: float | None
    published_robust_attack: str | None
    # Only for ``custom``: the exact FP32 statistics embedded in the checkpoint.
    custom_mean: tuple[float, float, float] | None = None
    custom_std: tuple[float, float, float] | None = None
    custom_provenance: str | None = None

    def normalization(self) -> NormalizationConfig:
        if self.normalization_profile == "custom":
            return NormalizationConfig(
                profile="custom", mean=self.custom_mean, std=self.custom_std, provenance=self.custom_provenance
            )
        return NormalizationConfig(profile=self.normalization_profile)

    def threat(self) -> dict[str, str]:
        return dict(IMAGENET_TEACHER_THREAT)


IMAGENET_TEACHER_SPECS: Mapping[str, ImageNetTeacherSpec] = {
    spec.registry_id: spec
    for spec in (
        ImageNetTeacherSpec(
            registry_id="salman2020_resnet50_linf_eps4",
            architecture="resnet50_imagenet",
            upstream_model_id="madrylab/robust-imagenet-models resnet50_linf_eps4.0",
            source_url=(
                "https://huggingface.co/madrylab/robust-imagenet-models/resolve/main/resnet50_linf_eps4.0.ckpt"
            ),
            license="MIT (huggingface.co/madrylab/robust-imagenet-models model card)",
            checkpoint_filename="resnet50_linf_eps4.0.ckpt",
            checkpoint_sha256="9e62216975c111b8081481533b6142dec34be69abe5d90b54d7f860015c263db",
            checkpoint_bytes=204818947,
            expected_parameter_count=25557032,
            normalization_profile="imagenet_standard",
            native_eval_interpolation="bilinear",
            state_dict_format="madrylab_robustness",
            published_clean_accuracy=63.86,
            published_robust_accuracy=None,
            published_robust_attack=None,
        ),
        ImageNetTeacherSpec(
            registry_id="singh2023_convnext_t_convstem",
            architecture="convnext_tiny_convstem_imagenet",
            upstream_model_id="Singh2023Revisiting_ConvNeXt-T-ConvStem (convnext_tiny_cvst_robust.pt)",
            source_url="https://nc.mlcloud.uni-tuebingen.de/index.php/s/BFLoMrMdn8iBk7Y",
            license="none: github.com/nmndeep/revisiting-at has no licence file (recorded 2026-10-08)",
            checkpoint_filename="convnext_tiny_cvst_robust.pt",
            checkpoint_sha256="0c725da1789c28168e5697c87b1adf7228527d07f7ad3b5b376271bb22da4ce3",
            checkpoint_bytes=114579981,
            expected_parameter_count=28627432,
            normalization_profile="imagenet_raw_identity",
            native_eval_interpolation="bicubic",
            state_dict_format="singh2023_base_model",
            published_clean_accuracy=72.7,
            published_robust_accuracy=49.5,
            published_robust_attack="AutoAttack Linf 4/255 (upstream README)",
        ),
        ImageNetTeacherSpec(
            registry_id="singh2023_vit_s_convstem",
            architecture="vit_s_convstem_imagenet",
            upstream_model_id="Singh2023Revisiting_ViT-S-ConvStem (vit_s_cvst_robust.pt)",
            source_url="https://nc.mlcloud.uni-tuebingen.de/index.php/s/agtDw3D7QXbDCmw",
            license="none: github.com/nmndeep/revisiting-at has no licence file (recorded 2026-10-08)",
            checkpoint_filename="vit_s_cvst_robust.pt",
            checkpoint_sha256="1d2197912aeebb1eaa443df6c79d921470966026bb21529abbaf59bdaecc56a6",
            checkpoint_bytes=91176941,
            expected_parameter_count=22777576,
            # The embedded normalizer is the ImageNet std but a mean that
            # differs from (0.485, 0.456, 0.406) by up to 4.8e-5; RobustBench
            # loads (and so evaluates with) these checkpoint values, so the
            # adapter applies exactly them.
            normalization_profile="custom",
            native_eval_interpolation="bicubic",
            state_dict_format="singh2023_base_model",
            published_clean_accuracy=72.5,
            published_robust_accuracy=48.1,
            published_robust_attack="AutoAttack Linf 4/255 (upstream README)",
            custom_mean=(0.48495230078697205, 0.4560000002384186, 0.40598925948143005),
            custom_std=(0.2290000021457672, 0.2240000069141388, 0.22499999403953552),
            custom_provenance=(
                "vit_s_cvst_robust.pt (sha256 1d2197912aee...) embedded base_model.normalize.mean/std, FP32 values"
            ),
        ),
        ImageNetTeacherSpec(
            registry_id="singh2023_convnext_b_convstem",
            architecture="convnext_base_convstem_imagenet",
            upstream_model_id="Singh2023Revisiting_ConvNeXt-B-ConvStem (convnext_b_cvst_robust.pt)",
            source_url="https://nc.mlcloud.uni-tuebingen.de/index.php/s/RQBEXagC7R7XweX",
            license="none: github.com/nmndeep/revisiting-at has no licence file (recorded 2026-10-08)",
            checkpoint_filename="convnext_b_cvst_robust.pt",
            checkpoint_sha256="ea83ada60aea2ac329d7de7eb00a3bed252b971c790207fb206b5dd2ed8162f8",
            checkpoint_bytes=355147777,
            expected_parameter_count=88753416,
            normalization_profile="imagenet_raw_identity",
            native_eval_interpolation="bicubic",
            state_dict_format="singh2023_base_model",
            published_clean_accuracy=75.9,
            published_robust_accuracy=56.1,
            published_robust_attack="AutoAttack Linf 4/255 at 224 px (upstream README)",
        ),
        ImageNetTeacherSpec(
            registry_id="ard_mobilenetv4_conv_medium_phase1_random",
            architecture="mobilenetv4_conv_medium_imagenet",
            upstream_model_id="plan0103-phase1-mobilenetv4-conv-medium-random-v1 last.pt (epoch 49)",
            source_url="Ferret:workspace-local/ard-runtime/ard_codex_bootstrap/runs/"
            "plan0103-phase1-mobilenetv4-conv-medium-random-v1/outputs/train/last.pt",
            license="project-owned",
            checkpoint_filename="last.pt",
            checkpoint_sha256="c1fb3e3fc264a560ab7d29e5a431996ebaa802eeee19379f47102594154ea400",
            checkpoint_bytes=78265635,
            expected_parameter_count=9715512,
            normalization_profile="imagenet_standard",
            native_eval_interpolation="bilinear",
            state_dict_format="ard_training_checkpoint",
            published_clean_accuracy=None,
            published_robust_accuracy=None,
            published_robust_attack=None,
        ),
    )
}


# --------------------------------------------------------------------------
# ConvStem modules (structure of RobustBench convstem_models.py, MIT licence).
# Module and parameter names are load-bearing: they must match the upstream
# ``base_model.`` state-dict keys exactly.
# --------------------------------------------------------------------------


class ChannelsFirstLayerNorm(nn.Module):
    """LayerNorm over the channel axis of an NCHW tensor (ConvNeXt reference form)."""

    def __init__(self, channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        mean = inputs.mean(1, keepdim=True)
        variance = (inputs - mean).pow(2).mean(1, keepdim=True)
        normalized = (inputs - mean) / torch.sqrt(variance + self.eps)
        return self.weight[:, None, None] * normalized + self.bias[:, None, None]


def _conv_ln_gelu(in_channels: int, out_channels: int, *, stride: int) -> list[nn.Module]:
    return [
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
        ChannelsFirstLayerNorm(out_channels),
        nn.GELU(),
    ]


class _Stem(nn.Module):
    def __init__(self, layers: list[nn.Module]) -> None:
        super().__init__()
        self.stem = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.stem(inputs)


def convnext_tiny_convstem(planes: int = 48) -> nn.Module:
    """ConvBlock1(48): two stride-2 conv/LN/GELU stages to 96 channels."""
    return _Stem([*_conv_ln_gelu(3, planes, stride=2), *_conv_ln_gelu(planes, planes * 2, stride=2)])


def convnext_base_convstem(planes: int = 64) -> nn.Module:
    """ConvBlock3(64): 64 -> 96 -> 128 channels, overall stride 4."""
    middle = int(planes * 1.5)
    return _Stem(
        [
            *_conv_ln_gelu(3, planes, stride=2),
            *_conv_ln_gelu(planes, middle, stride=2),
            *_conv_ln_gelu(middle, planes * 2, stride=1),
        ]
    )


def vit_convstem(planes: int = 48, end_size: int = 8) -> nn.Module:
    """ConvBlock(48, end_siz=8): four stride-2 conv/LN/GELU stages, then 1x1 to 384."""
    final = planes * end_size
    return _Stem(
        [
            *_conv_ln_gelu(3, planes, stride=2),
            *_conv_ln_gelu(planes, planes * 2, stride=2),
            *_conv_ln_gelu(planes * 2, planes * 4, stride=2),
            *_conv_ln_gelu(planes * 4, planes * 8, stride=2),
            nn.Conv2d(planes * 8, final, kernel_size=1, stride=1, padding=0),
        ]
    )


def build_imagenet_teacher_architecture(architecture: str) -> nn.Module:
    """Construct an untrained teacher network (no download, 1000 classes)."""
    import timm

    if architecture == "resnet50_imagenet":
        from torchvision import models

        return models.resnet50(weights=None, num_classes=1000)
    if architecture == "convnext_tiny_convstem_imagenet":
        model = timm.create_model("convnext_tiny", pretrained=False, num_classes=1000)
        model.stem = convnext_tiny_convstem()
        return model
    if architecture == "convnext_base_convstem_imagenet":
        model = timm.create_model("convnext_base", pretrained=False, num_classes=1000)
        model.stem = convnext_base_convstem()
        return model
    if architecture == "vit_s_convstem_imagenet":
        model = timm.create_model("deit_small_patch16_224", pretrained=False, num_classes=1000)
        model.patch_embed.proj = vit_convstem()
        return model
    if architecture == "mobilenetv4_conv_medium_imagenet":
        from .registry import build_architecture

        return build_architecture("mobilenetv4_conv_medium_imagenet", 1000)
    raise ImageNetTeacherRegistryError(f"unknown ImageNet teacher architecture: {architecture}")


# --------------------------------------------------------------------------
# State-dict extraction per checkpoint format.
# --------------------------------------------------------------------------


def _check_normalizer(mean: torch.Tensor, std: torch.Tensor, *, expected: NormalizationConfig, where: str) -> None:
    """The checkpoint's own normalizer must equal what the adapter will apply, in FP32, exactly.

    The adapter builds its buffers with ``torch.tensor(config.mean, float32)``,
    so FP32 equality here means the teacher sees exactly the normalization it
    was trained and published with.
    """
    assert expected.mean is not None and expected.std is not None
    expected_mean = torch.tensor(expected.mean, dtype=torch.float32)
    expected_std = torch.tensor(expected.std, dtype=torch.float32)
    observed_mean = mean.detach().flatten()
    observed_std = std.detach().flatten()
    if observed_mean.shape != (3,) or observed_std.shape != (3,):
        raise ImageNetTeacherRegistryError(f"{where}: embedded normalizer must hold 3 channel values")
    if not torch.equal(observed_mean.to(torch.float32), expected_mean) or not torch.equal(
        observed_std.to(torch.float32), expected_std
    ):
        raise ImageNetTeacherRegistryError(
            f"{where}: embedded normalizer {observed_mean.tolist()}/{observed_std.tolist()} differs from the "
            f"profile the adapter applies ({expected.profile}: {list(expected.mean)}/{list(expected.std)})"
        )


def _extract_madrylab(payload: Any, *, normalization: NormalizationConfig) -> dict[str, torch.Tensor]:
    if not isinstance(payload, Mapping) or not isinstance(payload.get("model"), Mapping):
        raise ImageNetTeacherRegistryError("MadryLab checkpoint must contain a 'model' state mapping")
    raw = payload["model"]
    model_prefix, attacker_prefix = "module.model.", "module.attacker.model."
    state = {key[len(model_prefix) :]: value for key, value in raw.items() if key.startswith(model_prefix)}
    attacker = {key[len(attacker_prefix) :]: value for key, value in raw.items() if key.startswith(attacker_prefix)}
    other = {
        key
        for key in raw
        if not key.startswith(model_prefix)
        and not key.startswith(attacker_prefix)
        and not key.startswith("module.normalizer.")
        and not key.startswith("module.attacker.normalize.")
    }
    if other:
        raise ImageNetTeacherRegistryError(f"MadryLab checkpoint has unexpected keys: {sorted(other)[:5]}")
    if not state:
        raise ImageNetTeacherRegistryError("MadryLab checkpoint holds no module.model. weights")
    for prefix in ("module.normalizer.", "module.attacker.normalize."):
        _check_normalizer(
            raw[f"{prefix}new_mean"], raw[f"{prefix}new_std"], expected=normalization, where=f"MadryLab {prefix}"
        )
    if set(attacker) != set(state) or any(not torch.equal(attacker[key], state[key]) for key in state):
        raise ImageNetTeacherRegistryError("MadryLab attacker copy differs from the classifier weights")
    return state


def _extract_singh(payload: Any, *, normalization: NormalizationConfig) -> dict[str, torch.Tensor]:
    if not isinstance(payload, Mapping) or not payload:
        raise ImageNetTeacherRegistryError("Singh et al. checkpoint must be a non-empty state mapping")
    prefix = "base_model."
    if any(not isinstance(key, str) or not key.startswith(prefix) for key in payload):
        raise ImageNetTeacherRegistryError("Singh et al. checkpoint keys must all start with 'base_model.'")
    state = {key[len(prefix) :]: value for key, value in payload.items()}
    embedded = "normalize.mean" in state or "normalize.std" in state
    if embedded == (normalization.profile == "imagenet_raw_identity"):
        raise ImageNetTeacherRegistryError(
            f"Singh et al. checkpoint {'embeds' if embedded else 'has no'} input normalizer, inconsistent with "
            f"profile {normalization.profile}"
        )
    if embedded:
        _check_normalizer(
            state.pop("normalize.mean"), state.pop("normalize.std"), expected=normalization, where="Singh et al."
        )
        if any(not key.startswith("model.") for key in state):
            raise ImageNetTeacherRegistryError("Singh et al. normalized checkpoint must nest weights under 'model.'")
        state = {key[len("model.") :]: value for key, value in state.items()}
    return state


def _extract_ard(payload: Any, *, normalization: NormalizationConfig) -> dict[str, torch.Tensor]:
    if not isinstance(payload, Mapping) or not isinstance(payload.get("model"), Mapping):
        raise ImageNetTeacherRegistryError("ARD training checkpoint must contain a 'model' state mapping")
    if payload.get("epoch_boundary") != "end":
        raise ImageNetTeacherRegistryError("ARD teacher checkpoint must be an epoch-boundary checkpoint")
    raw = payload["model"]
    _check_normalizer(
        raw["normalization.mean"], raw["normalization.std"], expected=normalization, where="ARD student adapter"
    )
    if any(not key.startswith("model.") for key in raw if not key.startswith("normalization.")):
        raise ImageNetTeacherRegistryError("ARD checkpoint student weights must be nested under 'model.'")
    return {key[len("model.") :]: value for key, value in raw.items() if key.startswith("model.")}


def read_verified_teacher_state(spec: ImageNetTeacherSpec, path: Path) -> dict[str, torch.Tensor]:
    """Hash the file once, deserialize exactly those bytes, return the bare network state."""
    if not path.is_file():
        raise ImageNetTeacherRegistryError(f"teacher checkpoint is missing; no download is attempted: {path}")
    data = path.read_bytes()
    observed = hashlib.sha256(data).hexdigest()
    if observed != spec.checkpoint_sha256:
        raise ImageNetTeacherRegistryError(
            f"teacher checkpoint hash mismatch for {spec.registry_id}: expected {spec.checkpoint_sha256}, "
            f"got {observed} ({path})"
        )
    if spec.state_dict_format == "ard_training_checkpoint":
        # Our own epoch-boundary checkpoint stores NumPy RNG state, which the
        # weights-only unpickler refuses; the bytes are digest-pinned above.
        payload = torch.load(io.BytesIO(data), map_location="cpu", weights_only=False)
        return _extract_ard(payload, normalization=spec.normalization())
    payload = torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
    if spec.state_dict_format == "madrylab_robustness":
        return _extract_madrylab(payload, normalization=spec.normalization())
    return _extract_singh(payload, normalization=spec.normalization())


def spec_for(registry_id: str | None) -> ImageNetTeacherSpec:
    try:
        return IMAGENET_TEACHER_SPECS[str(registry_id)]
    except KeyError as exc:
        raise ImageNetTeacherRegistryError(f"unknown ImageNet teacher registry ID: {registry_id!r}") from exc


def validate_imagenet_teacher_config(config: TeacherConfig) -> ImageNetTeacherSpec:
    """The config must restate the pinned profile exactly; nothing is inferred."""
    if config.source != "imagenet_registry":
        raise ImageNetTeacherRegistryError("ImageNet registry validation requires teacher.source=imagenet_registry")
    spec = spec_for(config.registry_id)
    mismatched = [
        name
        for name, ok in (
            ("architecture", config.architecture == spec.architecture),
            ("num_classes", config.num_classes == 1000),
            ("normalization", config.normalization == spec.normalization()),
            ("preprocessing_owner", config.preprocessing_owner == "teacher_adapter"),
            ("threat_norm", config.threat_norm == "linf"),
            ("threat_epsilon", config.threat_epsilon == "4/255"),
            ("checkpoint_sha256", config.checkpoint_sha256 == spec.checkpoint_sha256),
            (
                "checkpoint filename",
                config.checkpoint is not None and config.checkpoint.name == spec.checkpoint_filename,
            ),
        )
        if not ok
    ]
    if mismatched:
        raise ImageNetTeacherRegistryError(
            f"teacher config does not match ImageNet registry profile {spec.registry_id}: " + ", ".join(mismatched)
        )
    return spec


def load_imagenet_teacher_network(
    spec: ImageNetTeacherSpec,
    path: Path,
    *,
    architecture_builder: Callable[[str], nn.Module] = build_imagenet_teacher_architecture,
) -> nn.Module:
    model = architecture_builder(spec.architecture)
    observed = sum(parameter.numel() for parameter in model.parameters())
    if observed != spec.expected_parameter_count:
        raise ImageNetTeacherRegistryError(
            f"{spec.registry_id} parameter count mismatch: expected {spec.expected_parameter_count}, got {observed}"
        )
    state = read_verified_teacher_state(spec, path)
    model.load_state_dict(state, strict=True)
    for tensor in state.values():
        if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()):
            raise ImageNetTeacherRegistryError(f"{spec.registry_id} checkpoint holds non-finite weights")
    return model


def build_imagenet_teacher(config: TeacherConfig) -> Any:
    """Frozen, eval-only :class:`TeacherAdapter` for a registered ImageNet teacher."""
    from .teacher import TeacherAdapter, TeacherMetadata

    spec = validate_imagenet_teacher_config(config)
    assert config.checkpoint is not None
    model = load_imagenet_teacher_network(spec, config.checkpoint)
    metadata = TeacherMetadata(
        architecture=spec.architecture,
        num_classes=1000,
        normalization=spec.normalization(),
        checkpoint_sha256=spec.checkpoint_sha256,
        registry_id=spec.registry_id,
        upstream_model_id=spec.upstream_model_id,
        preprocessing_owner="teacher_adapter",
        preprocessing_profile=spec.normalization_profile,
        threat_model=spec.threat(),
    )
    return TeacherAdapter(model, metadata)


def eval_interpolation(spec: ImageNetTeacherSpec) -> Any:
    from torchvision.transforms import InterpolationMode

    return InterpolationMode.BICUBIC if spec.native_eval_interpolation == "bicubic" else InterpolationMode.BILINEAR
