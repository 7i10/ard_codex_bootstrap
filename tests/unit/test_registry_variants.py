"""Plan 0103 Phase 2 batch B: architecture-variant student ids (human-approved 2026-10-08).

Pins each variant's construction, measured parameter / MAC counts (``ard.models.complexity``, one
3x224x224 image, 1000 classes), forward shapes, random-init-only contract, the Phase 2 configs, and
that every pre-existing architecture id still builds exactly what it built before.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import timm
import torch
from pydantic import ValidationError
from torch import nn

from ard.config.loader import load_config
from ard.config.schema import CUDA_GRAPH_ARCHITECTURES, ModelConfig
from ard.models import build_architecture
from ard.models.complexity import count_macs, count_parameters
from ard.models.variants import VARIANT_BUILDERS, build_mobilenetv4_conv_small_variant

pytestmark = pytest.mark.t1

# id -> (params, MACs at 224, base id, base params, base MACs)
BASE = {
    "mobilenetv4_conv_small_imagenet": (3_774_024, 186_005_568),
    "deit_tiny_imagenet": (5_717_416, 1_253_683_200),
    "convnext_atto_imagenet": (3_695_520, 547_332_480),
}
VARIANTS = {
    "mobilenetv4_conv_small_silu_imagenet": (3_774_024, 186_005_568, "mobilenetv4_conv_small_imagenet"),
    "mobilenetv4_conv_small_gelu_imagenet": (3_774_024, 186_005_568, "mobilenetv4_conv_small_imagenet"),
    "mobilenetv4_conv_small_se_imagenet": (3_772_296, 185_999_424, "mobilenetv4_conv_small_imagenet"),
    "mobilenetv4_conv_small_silu_se_imagenet": (3_772_296, 185_999_424, "mobilenetv4_conv_small_imagenet"),
    "deit_tiny_convstem_imagenet": (5_826_280, 1_337_677_824, "deit_tiny_imagenet"),
    "convnext_atto_deep_narrow_imagenet": (2_758_636, 554_452_128, "convnext_atto_imagenet"),
    "convnext_atto_ols_imagenet": (3_702_912, 576_534_912, "convnext_atto_imagenet"),
}
# Variants whose parameter count is matched to the base within +-3% by design (the deeper-narrower
# ConvNeXt is MAC-matched with fewer params; ConvStem DeiT and OLS ConvNeXt are reported, not matched).
PARAM_MATCHED = {
    "mobilenetv4_conv_small_silu_imagenet",
    "mobilenetv4_conv_small_gelu_imagenet",
    "mobilenetv4_conv_small_se_imagenet",
    "mobilenetv4_conv_small_silu_se_imagenet",
    "deit_tiny_convstem_imagenet",
    "convnext_atto_ols_imagenet",
}
EXISTING_TIMM_IDS = {
    "mobilenetv4_conv_small_imagenet": "mobilenetv4_conv_small.e1200_r224_in1k",
    "efficientnet_b0_imagenet": "efficientnet_b0.ra_in1k",
    "mobilenetv4_conv_medium_imagenet": "mobilenetv4_conv_medium.e500_r224_in1k",
    "convnext_atto_imagenet": "convnext_atto.d2_in1k",
    "deit_tiny_imagenet": "deit_tiny_patch16_224.fb_in1k",
    "mobilevit_s_imagenet": "mobilevit_s.cvnets_in1k",
    "convnext_tiny_imagenet": "convnext_tiny",
}


def test_variant_ids_are_exactly_the_schema_additions() -> None:
    for architecture in VARIANTS:
        assert ModelConfig(architecture=architecture, num_classes=1000).architecture == architecture
    assert set(VARIANTS) == set(VARIANT_BUILDERS)


@pytest.mark.parametrize("architecture", sorted(BASE))
def test_base_complexity_is_pinned(architecture: str) -> None:
    model = build_architecture(architecture, num_classes=1000)
    assert (count_parameters(model), count_macs(model)) == BASE[architecture]


@pytest.mark.parametrize("architecture", sorted(VARIANTS))
def test_variant_complexity_is_pinned(architecture: str) -> None:
    params, macs, base = VARIANTS[architecture]
    model = build_architecture(architecture, num_classes=1000)
    assert count_parameters(model) == params
    assert count_macs(model) == macs
    base_params, base_macs = BASE[base]
    if architecture in PARAM_MATCHED:
        assert abs(params / base_params - 1) <= 0.03
    if architecture.startswith("mobilenetv4_conv_small_"):
        assert abs(macs / base_macs - 1) <= 0.001


def test_convnext_deep_narrow_is_mac_matched_with_fewer_params() -> None:
    params, macs, _ = VARIANTS["convnext_atto_deep_narrow_imagenet"]
    base_params, base_macs = BASE["convnext_atto_imagenet"]
    assert abs(macs / base_macs - 1) <= 0.03
    assert params < 0.8 * base_params
    model = build_architecture("convnext_atto_deep_narrow_imagenet", num_classes=1000)
    assert [len(stage.blocks) for stage in model.stages] == [3, 3, 8, 1]
    assert [stage.blocks[0].conv_dw.in_channels for stage in model.stages] == [36, 72, 144, 288]


@pytest.mark.parametrize("architecture", sorted(VARIANTS))
@pytest.mark.parametrize("image_size", [224, 112])
def test_variant_forward_shape(architecture: str, image_size: int) -> None:
    model = build_architecture(architecture, num_classes=10).eval()
    images = torch.rand(2, 3, image_size, image_size)
    if architecture == "deit_tiny_convstem_imagenet" and image_size != 224:
        # Same fixed-size contract as the base DeiT-Tiny (absolute 14x14 position embedding).
        with pytest.raises(ValueError, match="does not match the model"):
            model(images)
        with pytest.raises(AssertionError, match="doesn't match model"):
            build_architecture("deit_tiny_imagenet", num_classes=10).eval()(images)
        return
    with torch.no_grad():
        assert model(images).shape == (2, 10)


@pytest.mark.parametrize("architecture", sorted(VARIANTS))
def test_pretrained_is_refused_for_variants(architecture: str) -> None:
    with pytest.raises(ValueError, match="pretrained=True is not supported"):
        build_architecture(architecture, num_classes=1000, pretrained=True)
    with pytest.raises(ValidationError, match="pretrained=True is not supported"):
        ModelConfig(architecture=architecture, num_classes=1000, pretrained=True)


@pytest.mark.parametrize("architecture", sorted(VARIANTS))
def test_variants_are_not_cuda_graph_eligible(architecture: str) -> None:
    assert architecture not in CUDA_GRAPH_ARCHITECTURES


def _activation_types(model: nn.Module) -> set[type]:
    from timm.layers.activations import GELU as TimmGELU

    kinds = (nn.ReLU, nn.ReLU6, nn.SiLU, nn.GELU, TimmGELU, nn.Hardswish, nn.Hardsigmoid)
    return {type(module) for module in model.modules() if isinstance(module, kinds)}


@pytest.mark.parametrize(
    ("architecture", "activation"),
    [
        ("mobilenetv4_conv_small_silu_imagenet", nn.SiLU),
        ("mobilenetv4_conv_small_gelu_imagenet", "gelu"),
        ("mobilenetv4_conv_small_se_imagenet", nn.ReLU),
        ("mobilenetv4_conv_small_silu_se_imagenet", nn.SiLU),
    ],
)
def test_mobilenetv4_variant_uses_one_activation_everywhere(architecture: str, activation: type | str) -> None:
    if activation == "gelu":
        from timm.layers import get_act_layer

        activation = get_act_layer("gelu")  # timm's own GELU module (exact erf GELU)
    model = build_architecture(architecture, num_classes=1000)
    assert _activation_types(model) == {activation}
    assert _activation_types(build_architecture("mobilenetv4_conv_small_imagenet", num_classes=1000)) == {nn.ReLU}


@pytest.mark.parametrize("architecture", ["mobilenetv4_conv_small_se_imagenet", "mobilenetv4_conv_small_silu_se_imagenet"])
def test_mobilenetv4_se_is_efficientnet_b0_style(architecture: str) -> None:
    """SE in every UIB block, squeeze = 0.25 x block input channels, sigmoid gate, squeeze act = block act."""
    from timm.models._efficientnet_blocks import SqueezeExcite, UniversalInvertedResidual

    model = build_architecture(architecture, num_classes=1000)
    b0_se = build_architecture("efficientnet_b0_imagenet", num_classes=1000).blocks[1][0].se
    assert isinstance(b0_se, SqueezeExcite)
    assert b0_se.conv_reduce.out_channels == 4  # B0's 16 -> 24 block: 0.25 x 16 input channels, not 96 mid
    blocks = [module for module in model.modules() if isinstance(module, UniversalInvertedResidual)]
    assert len(blocks) == 12
    activation = nn.SiLU if "silu" in architecture else nn.ReLU
    for block in blocks:
        assert isinstance(block.se, SqueezeExcite)
        assert block.se.conv_reduce.out_channels == round(block.pw_exp.conv.in_channels * 0.25)
        assert block.se.conv_reduce.in_channels == block.pw_exp.conv.out_channels
        assert type(block.se.gate) is type(b0_se.gate)  # timm's exact sigmoid, as in EfficientNet-B0
        assert type(block.se.gate).__name__ == "Sigmoid"
        assert isinstance(block.se.act1, activation)
    assert model.conv_head.out_channels == 1152
    assert sum(isinstance(module, SqueezeExcite) for module in model.modules()) == 12


def test_mobilenetv4_variant_builder_reproduces_timm_base_exactly() -> None:
    """The copied arch_def with ReLU, no SE, 1280 head is timm's mobilenetv4_conv_small, tensor for tensor."""
    torch.manual_seed(0)
    reference = timm.create_model("mobilenetv4_conv_small", pretrained=False, num_classes=1000).state_dict()
    torch.manual_seed(0)
    rebuilt = build_mobilenetv4_conv_small_variant(num_classes=1000).state_dict()
    assert list(reference) == list(rebuilt)
    for key in reference:
        assert torch.equal(reference[key], rebuilt[key]), key


def test_mobilenetv4_unmatched_se_delta_is_documented() -> None:
    """Without the head trim SE adds 249,408 params (+6.61%), outside +-3% -- hence the 1152 head."""
    model = build_mobilenetv4_conv_small_variant(num_classes=1000, se=True)
    assert count_parameters(model) == 4_023_432
    assert count_parameters(model) - BASE["mobilenetv4_conv_small_imagenet"][0] == 249_408


def test_deit_convstem_keeps_the_token_grid_and_transformer() -> None:
    model = build_architecture("deit_tiny_convstem_imagenet", num_classes=1000)
    base = build_architecture("deit_tiny_imagenet", num_classes=1000)
    assert model.patch_embed.num_patches == base.patch_embed.num_patches == 196
    assert model.pos_embed.shape == base.pos_embed.shape == (1, 197, 192)
    convs = [module for module in model.patch_embed.modules() if isinstance(module, nn.Conv2d)]
    assert [(conv.out_channels, conv.kernel_size, conv.stride) for conv in convs] == [
        (24, (3, 3), (2, 2)),
        (48, (3, 3), (2, 2)),
        (96, (3, 3), (2, 2)),
        (192, (3, 3), (2, 2)),
        (192, (1, 1), (1, 1)),
    ]
    stem_free = {key: value.shape for key, value in model.state_dict().items() if not key.startswith("patch_embed.")}
    base_stem_free = {key: value.shape for key, value in base.state_dict().items() if not key.startswith("patch_embed.")}
    assert stem_free == base_stem_free


@pytest.mark.parametrize(("architecture", "timm_name"), sorted(EXISTING_TIMM_IDS.items()))
def test_existing_timm_ids_build_exactly_what_they_built_before(architecture: str, timm_name: str) -> None:
    torch.manual_seed(0)
    reference = timm.create_model(timm_name, pretrained=False, num_classes=1000).state_dict()
    torch.manual_seed(0)
    built = build_architecture(architecture, num_classes=1000).state_dict()
    assert list(reference) == list(built)
    for key in reference:
        assert torch.equal(reference[key], built[key]), key


CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs" / "scientific"
MNV4_BASE = "imagenet_mobilenetv4_pgd_at_random_init_lr0025_cg.yaml"
PHASE2_CONFIGS = {
    "imagenet_mobilenetv4_silu_pgd_at_phase2_random_lr0025.yaml": ("mobilenetv4_conv_small_silu_imagenet", MNV4_BASE),
    "imagenet_mobilenetv4_gelu_pgd_at_phase2_random_lr0025.yaml": ("mobilenetv4_conv_small_gelu_imagenet", MNV4_BASE),
    "imagenet_mobilenetv4_se_pgd_at_phase2_random_lr0025.yaml": ("mobilenetv4_conv_small_se_imagenet", MNV4_BASE),
    "imagenet_mobilenetv4_silu_se_pgd_at_phase2_random_lr0025.yaml": (
        "mobilenetv4_conv_small_silu_se_imagenet",
        MNV4_BASE,
    ),
    "imagenet_deit_tiny_convstem_pgd_at_phase2_adamw_random_lr5em4.yaml": (
        "deit_tiny_convstem_imagenet",
        "imagenet_deit_tiny_pgd_at_phase1_adamw_random_lr5em4.yaml",
    ),
    "imagenet_convnext_atto_deep_narrow_pgd_at_phase2_adamw_random_lr5em4.yaml": (
        "convnext_atto_deep_narrow_imagenet",
        "imagenet_convnext_atto_pgd_at_phase1_adamw_random_lr5em4.yaml",
    ),
    "imagenet_convnext_atto_ols_pgd_at_phase2_adamw_random_lr5em4.yaml": (
        "convnext_atto_ols_imagenet",
        "imagenet_convnext_atto_pgd_at_phase1_adamw_random_lr5em4.yaml",
    ),
}


@pytest.mark.parametrize("config_file", sorted(PHASE2_CONFIGS))
def test_phase2_configs_change_only_architecture_and_group(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, config_file: str
) -> None:
    """Cloned from the base model's Phase 1 chosen-LR config; MobileNetV4-S additionally drops
    training.cuda_graph (variants are not in CUDA_GRAPH_ARCHITECTURES)."""
    for key, value in {
        "ARD_SEED": "7",
        "ARD_IMAGENET_ROOT": str(tmp_path / "imagenet"),
        "ARD_NUM_WORKERS": "0",
        "ARD_JOB_OUTPUT_DIR": str(tmp_path / "job-output"),
        "ARD_RUN_ID": "config-test-run",
        "WANDB_ENTITY": "entity",
        "WANDB_PROJECT": "project",
    }.items():
        monkeypatch.setenv(key, value)
    architecture, base_file = PHASE2_CONFIGS[config_file]
    base = load_config(CONFIG_DIR / base_file).model_dump(mode="json")
    variant = load_config(CONFIG_DIR / config_file).model_dump(mode="json")
    assert variant["student"]["architecture"] == architecture
    assert variant["student"]["pretrained"] is False
    assert variant["tracking"]["group"] != base["tracking"]["group"]
    assert "cuda_graph" not in variant["training"]  # excluded from the dump only when false
    for payload in (base, variant):
        payload["student"]["architecture"] = None
        payload["tracking"]["group"] = None
        payload["training"].pop("cuda_graph", None)
    assert variant == base
