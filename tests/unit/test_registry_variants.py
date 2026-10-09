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
from ard.models.complexity import count_mac_terms, count_parameters, measure
from ard.models.variants import VARIANT_BUILDERS, build_mobilenetv4_conv_small_variant

pytestmark = pytest.mark.t1

MNV4S, DEIT, CONVNEXT = "mobilenetv4_conv_small_imagenet", "deit_tiny_imagenet", "convnext_atto_imagenet"
# id -> (params, conv+linear MACs, attention-matmul MACs) at 224, 1000 classes.
BASE = {
    MNV4S: (3_774_024, 186_005_568, 0),
    DEIT: (5_717_416, 1_074_851_328, 178_831_872),
    CONVNEXT: (3_695_520, 547_332_480, 0),
    # MobileViT folds patches into the batch dimension; the counter must count every folded sequence.
    "mobilevit_s_imagenet": (5_578_632, 1_416_770_560, 104_736_768),
}
# id -> (params, conv+linear MACs, attention MACs, base id)
VARIANTS = {
    "mobilenetv4_conv_small_silu_imagenet": (3_774_024, 186_005_568, 0, MNV4S),
    "mobilenetv4_conv_small_gelu_imagenet": (3_774_024, 186_005_568, 0, MNV4S),
    "mobilenetv4_conv_small_se_imagenet": (3_772_296, 185_999_424, 0, MNV4S),
    "mobilenetv4_conv_small_silu_se_imagenet": (3_772_296, 185_999_424, 0, MNV4S),
    "mobilenetv4_conv_small_se_fullhead_imagenet": (4_023_432, 186_250_304, 0, MNV4S),
    "deit_tiny_convstem_imagenet": (5_826_280, 1_158_845_952, 178_831_872, DEIT),
    "convnext_atto_deep_narrow_imagenet": (2_758_636, 554_452_128, 0, CONVNEXT),
    "convnext_atto_ols_imagenet": (3_702_912, 576_534_912, 0, CONVNEXT),
    "convnext_atto_convstem_imagenet": (3_701_400, 570_664_320, 0, CONVNEXT),
}
# Variants whose parameter count is within +-3% of the base (the deeper-narrower ConvNeXt is
# MAC-matched with fewer params; the SE full-head control is deliberately +6.61%).
PARAM_MATCHED = {
    "mobilenetv4_conv_small_silu_imagenet",
    "mobilenetv4_conv_small_gelu_imagenet",
    "mobilenetv4_conv_small_se_imagenet",
    "mobilenetv4_conv_small_silu_se_imagenet",
    "deit_tiny_convstem_imagenet",
    "convnext_atto_ols_imagenet",
    "convnext_atto_convstem_imagenet",
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
    assert (count_parameters(model), *count_mac_terms(model)) == BASE[architecture]


def test_conv_linear_convention_reproduces_the_phase1_table() -> None:
    """Plan 0103's Phase 1 table counts conv + linear only: DeiT-Tiny 1.075 G, MobileViT-S 1.417 G."""
    deit = measure(build_architecture(DEIT, num_classes=1000))
    mobilevit = measure(build_architecture("mobilevit_s_imagenet", num_classes=1000))
    assert round(deit.conv_linear_macs / 1e9, 3) == 1.075
    assert round(mobilevit.conv_linear_macs / 1e9, 3) == 1.417
    assert round(deit.macs / 1e9, 3) == 1.254
    assert round(mobilevit.macs / 1e9, 3) == 1.522


@pytest.mark.parametrize("architecture", sorted(VARIANTS))
def test_variant_complexity_is_pinned(architecture: str) -> None:
    params, conv_linear, attention, base = VARIANTS[architecture]
    model = build_architecture(architecture, num_classes=1000)
    assert count_parameters(model) == params
    assert count_mac_terms(model) == (conv_linear, attention)
    base_params, base_conv_linear, _ = BASE[base]
    if architecture in PARAM_MATCHED:
        assert abs(params / base_params - 1) <= 0.03
    if architecture.startswith("mobilenetv4_conv_small_"):
        assert abs(conv_linear / base_conv_linear - 1) <= 0.002


def test_deit_convstem_mac_delta_under_both_conventions() -> None:
    _, conv_linear, attention, _ = VARIANTS["deit_tiny_convstem_imagenet"]
    _, base_conv_linear, base_attention = BASE[DEIT]
    assert attention == base_attention
    assert round(100 * (conv_linear / base_conv_linear - 1), 2) == 7.81
    assert round(100 * ((conv_linear + attention) / (base_conv_linear + base_attention) - 1), 2) == 6.70


def test_convnext_deep_narrow_is_mac_matched_with_fewer_params() -> None:
    params, macs, _, _ = VARIANTS["convnext_atto_deep_narrow_imagenet"]
    base_params, base_macs, _ = BASE[CONVNEXT]
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


# Variants admitted by training.cuda_graph after their own parity tests (plan 0105, human-approved 2026-10-09:
# the ConvNeXt-Atto / DeiT-Tiny batch-B variants, SGD and AdamW). Every other variant stays refused.
_CUDA_GRAPH_VARIANTS = frozenset(
    {
        "convnext_atto_deep_narrow_imagenet",
        "convnext_atto_ols_imagenet",
        "convnext_atto_convstem_imagenet",
        "deit_tiny_convstem_imagenet",
    }
)


@pytest.mark.parametrize("architecture", sorted(VARIANTS))
def test_only_parity_tested_variants_are_cuda_graph_eligible(architecture: str) -> None:
    assert (architecture in CUDA_GRAPH_ARCHITECTURES) is (architecture in _CUDA_GRAPH_VARIANTS)


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
    """Without the head trim SE adds 249,408 params (+6.61%), outside +-3% -- hence the 1152 head of the
    matched arm, and the separate full-head control id."""
    model = build_architecture("mobilenetv4_conv_small_se_fullhead_imagenet", num_classes=1000)
    assert count_parameters(model) == 4_023_432
    assert count_parameters(model) - BASE[MNV4S][0] == 249_408
    assert model.conv_head.out_channels == 1280


def _state_shapes(model: nn.Module) -> dict[str, torch.Size]:
    return {key: value.shape for key, value in model.state_dict().items()}


@pytest.mark.parametrize("architecture", ["mobilenetv4_conv_small_silu_imagenet", "mobilenetv4_conv_small_gelu_imagenet"])
def test_activation_variants_are_tensor_identical_to_the_base_except_activation_modules(architecture: str) -> None:
    """Activations hold no tensors and draw no init RNG: same seed -> the same state_dict, value for value."""
    torch.manual_seed(0)
    base = build_architecture(MNV4S, num_classes=1000)
    torch.manual_seed(0)
    variant = build_architecture(architecture, num_classes=1000)
    base_state, variant_state = base.state_dict(), variant.state_dict()
    assert list(base_state) == list(variant_state)
    for key in base_state:
        assert torch.equal(base_state[key], variant_state[key]), key
    base_modules = dict(base.named_modules())
    differing = {
        name for name, module in variant.named_modules() if type(module) is not type(base_modules[name])
    }
    assert differing and all(type(base_modules[name]) is nn.ReLU for name in differing)


@pytest.mark.parametrize(
    ("architecture", "head_changes"),
    [
        ("mobilenetv4_conv_small_se_imagenet", True),
        ("mobilenetv4_conv_small_silu_se_imagenet", True),
        ("mobilenetv4_conv_small_se_fullhead_imagenet", False),
    ],
)
def test_se_variants_match_the_base_except_se_and_head(architecture: str, head_changes: bool) -> None:
    """Same tensors (names and shapes) as the base except the added ``.se.*`` and, for the matched arms,
    the narrowed head. Values differ under one seed only because SE's init draws shift the RNG order."""
    base, variant = _state_shapes(build_architecture(MNV4S, num_classes=1000)), _state_shapes(
        build_architecture(architecture, num_classes=1000)
    )
    head = ("conv_head.", "norm_head.", "classifier.")
    se_keys = {key for key in variant if ".se." in key}
    assert se_keys and not any(".se." in key for key in base)
    changed_head = {key for key in base if key.startswith(head) and base[key] != variant[key]}
    assert bool(changed_head) is head_changes
    assert {key: shape for key, shape in variant.items() if key not in se_keys and key not in changed_head} == {
        key: shape for key, shape in base.items() if key not in changed_head
    }


def test_convnext_convstem_is_singh_convblock1_scaled_to_atto() -> None:
    from timm.layers import LayerNorm2d

    model = build_architecture("convnext_atto_convstem_imagenet", num_classes=1000)
    base = build_architecture(CONVNEXT, num_classes=1000)
    layers = list(model.stem.stem)
    assert [type(layer) for layer in layers] == [nn.Conv2d, LayerNorm2d, nn.GELU] * 2
    convs = [layer for layer in layers if isinstance(layer, nn.Conv2d)]
    assert [(c.in_channels, c.out_channels, c.kernel_size, c.stride, c.padding) for c in convs] == [
        (3, 20, (3, 3), (2, 2), (1, 1)),
        (20, 40, (3, 3), (2, 2), (1, 1)),
    ]
    assert all(layer.eps == 1e-6 for layer in layers if isinstance(layer, LayerNorm2d))
    stem_free = {key: shape for key, shape in _state_shapes(model).items() if not key.startswith("stem.")}
    assert stem_free == {key: shape for key, shape in _state_shapes(base).items() if not key.startswith("stem.")}


def test_convnext_ols_stem_is_linear_not_a_convstem() -> None:
    """timm's overlap_tiered stem has no activation at all -- it is not Singh's ConvStem."""
    model = build_architecture("convnext_atto_ols_imagenet", num_classes=1000)
    assert not any(isinstance(module, (nn.GELU, nn.ReLU)) for module in model.stem.modules())
    assert [m.out_channels for m in model.stem.modules() if isinstance(m, nn.Conv2d)] == [24, 40]


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
    "imagenet_convnext_atto_deep_narrow_pgd_at_phase2_adamw_random_lr1em3.yaml": (
        "convnext_atto_deep_narrow_imagenet",
        "imagenet_convnext_atto_pgd_at_phase1_adamw_random_lr1em3.yaml",
    ),
    "imagenet_convnext_atto_ols_pgd_at_phase2_adamw_random_lr1em3.yaml": (
        "convnext_atto_ols_imagenet",
        "imagenet_convnext_atto_pgd_at_phase1_adamw_random_lr1em3.yaml",
    ),
    "imagenet_convnext_atto_convstem_pgd_at_phase2_adamw_random_lr1em3.yaml": (
        "convnext_atto_convstem_imagenet",
        "imagenet_convnext_atto_pgd_at_phase1_adamw_random_lr1em3.yaml",
    ),
    "imagenet_mobilenetv4_se_fullhead_pgd_at_phase2_random_lr0025.yaml": (
        "mobilenetv4_conv_small_se_fullhead_imagenet",
        MNV4_BASE,
    ),
}


def test_every_variant_has_exactly_one_phase2_config() -> None:
    assert sorted(architecture for architecture, _ in PHASE2_CONFIGS.values()) == sorted(VARIANTS)


@pytest.mark.parametrize("config_file", sorted(PHASE2_CONFIGS))
def test_phase2_adamw_configs_are_marked_do_not_launch(config_file: str) -> None:
    header = (CONFIG_DIR / config_file).read_text().split("schema_version")[0]
    base = PHASE2_CONFIGS[config_file][1]
    if base != MNV4_BASE and "convnext_atto" not in base:
        assert "DO NOT LAUNCH until the AdamW grid closes" in header
    if "convnext_atto" in base:  # ConvNeXt-Atto AdamW grid closed 2026-10-09 at 1e-3
        assert "LR set to the closed AdamW grid winner 1e-3" in header


@pytest.mark.parametrize("config_file", sorted(PHASE2_CONFIGS))
def test_phase2_configs_change_only_architecture_and_group(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, config_file: str
) -> None:
    """Cloned from the base model's Phase 1 chosen-LR config plus the Phase 2 template's
    training.selection_subset_size 5000 and weight_ema_decay 0.9999; MobileNetV4-S additionally drops training.cuda_graph
    (variants are not in CUDA_GRAPH_ARCHITECTURES)."""
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
    assert variant["training"]["selection_subset_size"] == 5000
    assert "selection_subset_size" not in base["training"]
    assert variant["training"]["weight_ema_decay"] == 0.9999  # EMA kept in every Phase 2 run
    assert base["training"].get("weight_ema_decay") is None
    for payload in (base, variant):
        payload["student"]["architecture"] = None
        payload["tracking"]["group"] = None
        payload["training"].pop("cuda_graph", None)
        payload["training"].pop("selection_subset_size", None)
        payload["training"].pop("weight_ema_decay", None)
    assert variant == base
