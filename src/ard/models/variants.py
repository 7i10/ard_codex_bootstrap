"""Plan 0103 Phase 2 architecture variants for adversarial-training ablations (human-approved 2026-10-08).

Each variant changes one named design factor of a Phase 1 base model (timm 1.0.9 definitions) and is
random-init only: no ImageNet checkpoint exists for these exact networks, and loading the base model's
weights into a modified network is not a defined initialization here. The registry refuses
``pretrained=True`` for every id built in this module.

Parameter / MAC numbers below are measured with :mod:`ard.models.complexity` (one 3x224x224 image,
1000 classes; MACs are conv+linear unless stated) and pinned in ``tests/unit/test_registry_variants.py``.
"""

from __future__ import annotations

from functools import partial

import torch
from torch import nn

# MobileNetV4-Conv-Small block definition, copied verbatim from timm 1.0.9
# ``timm.models.mobilenetv3._gen_mobilenet_v4`` ('small', non-hybrid). The copy exists only so SE can be
# appended per block; tests assert that building it with ReLU, no SE and the default 1280-wide head
# reproduces ``timm.create_model("mobilenetv4_conv_small")`` tensor for tensor under the same seed.
_MOBILENETV4_CONV_SMALL_ARCH_DEF: tuple[tuple[str, ...], ...] = (
    ("cn_r1_k3_s2_e1_c32", "cn_r1_k1_s1_e1_c32"),  # stage 0, 112x112 in
    ("cn_r1_k3_s2_e1_c96", "cn_r1_k1_s1_e1_c64"),  # stage 1, 56x56 in
    ("uir_r1_a5_k5_s2_e3_c96", "uir_r4_a0_k3_s1_e2_c96", "uir_r1_a3_k0_s1_e4_c96"),  # stage 2, 28x28 in
    (  # stage 3, 14x14 in
        "uir_r1_a3_k3_s2_e6_c128",
        "uir_r1_a5_k5_s1_e4_c128",
        "uir_r1_a0_k5_s1_e4_c128",
        "uir_r1_a0_k5_s1_e3_c128",
        "uir_r2_a0_k3_s1_e4_c128",
    ),
    ("cn_r1_k1_s1_c960",),  # stage 4, 7x7 in
)
MOBILENETV4_CONV_SMALL_NUM_FEATURES = 1280

# EfficientNet-B0 convention (Tan & Le 2019; timm ``_gen_efficientnet``): SE squeeze width is 0.25 x the
# *block input* channels (timm ``se_from_exp=False``), sigmoid gate, squeeze activation = the block's own
# activation. Applied to every UIB block (stages 2-3, where B0 also has SE in every MBConv); the plain
# ``cn`` conv-BN-act stem/stage-0/1/4 layers have no expansion and no SE, as in timm.
SE_RATIO_OF_BLOCK_INPUT = 0.25
# Conv-head width (``num_features``) that keeps the SE variants' parameter count within 0.05% of the
# base 3,774,024: SE adds 249,408 params at a 1280-wide head (4,023,432, +6.61%); each head channel
# costs 960 (conv_head) + 2 (BN) + 1000 (classifier weights) = 1962 params, so 1280 -> 1152 removes
# 251,136 -> 3,772,296 (-0.046%). 1152 = 9 x 128 keeps the head a multiple of 8. MACs at 224: SE adds
# 244,736, the narrower head removes 250,880 -> 185,999,424 vs 186,005,568 (-0.003%).
MOBILENETV4_CONV_SMALL_SE_NUM_FEATURES = 1152


def mobilenetv4_conv_small_arch_def(*, se: bool) -> list[list[str]]:
    suffix = f"_se{SE_RATIO_OF_BLOCK_INPUT}" if se else ""
    return [[block + suffix if block.startswith("uir_") else block for block in stage] for stage in _MOBILENETV4_CONV_SMALL_ARCH_DEF]


def build_mobilenetv4_conv_small_variant(
    *, num_classes: int, act_layer: str = "relu", se: bool = False, num_features: int = MOBILENETV4_CONV_SMALL_NUM_FEATURES
) -> nn.Module:
    """MobileNetV4-Conv-Small with the activation, SE and head width as the only free factors.

    Mirrors ``timm.models.mobilenetv3._gen_mobilenet_v4``'s model kwargs for 'small' at channel multiplier
    1.0 exactly (stem 32, BN, no layer scale, bias-free normed conv head) and constructs ``MobileNetV3``
    directly, so the weight initialization (``efficientnet_init_weights``) is timm's own.
    """
    from timm.layers import get_act_layer
    from timm.models._efficientnet_blocks import SqueezeExcite
    from timm.models._efficientnet_builder import decode_arch_def, resolve_bn_args, round_channels
    from timm.models.mobilenetv3 import MobileNetV3

    return MobileNetV3(
        block_args=decode_arch_def(mobilenetv4_conv_small_arch_def(se=se)),
        num_classes=num_classes,
        head_bias=False,
        head_norm=True,
        num_features=num_features,
        stem_size=32,
        fix_stem=False,
        round_chs_fn=partial(round_channels, multiplier=1.0),
        norm_layer=partial(nn.BatchNorm2d, **resolve_bn_args({})),
        act_layer=get_act_layer(act_layer),
        se_layer=SqueezeExcite,  # timm default gate nn.Sigmoid, as in timm efficientnet_b0
        se_from_exp=False,
        layer_scale_init_value=None,
    )


class ConvStemPatchEmbed(nn.Module):
    """Convolutional stem replacing ViT's 16x16 patchify embedding (drop-in for timm ``PatchEmbed``).

    Singh/Croce/Hein 2023 (arXiv:2303.01870, "ConvStem", their ViT-S stem scaled to DeiT-Tiny's width)
    after Xiao et al. 2021 ("Early convolutions help transformers see better"); the same layers as their
    released ``ConvBlock`` (robustbench ``convstem_models.py``; ``vit_s_cvst`` = ``ConvBlock(48, end_siz=8)``
    as ``patch_embed.proj``), here ``ConvBlock(24, end_siz=8)`` for width 192: four 3x3 stride-2 convs
    with channels 24, 48, 96, 192 (embed_dim/8 doubling), each followed by channels-first LayerNorm and
    GELU, then a 1x1 conv to ``embed_dim``. Total stride 16, so a 224 input yields the same 14x14 token
    grid as patch-16 and the 197-entry positional embedding stays valid. Same fixed-size contract as the
    base ``PatchEmbed`` (``strict_img_size``): any other input size is refused, exactly as the base is.
    """

    def __init__(
        self,
        img_size: int | tuple[int, int] = 224,
        patch_size: int | tuple[int, int] = 16,
        in_chans: int = 3,
        embed_dim: int = 192,
        bias: bool = True,
        dynamic_img_pad: bool = False,
        **kwargs: object,
    ) -> None:
        super().__init__()
        from timm.layers import LayerNorm2d, to_2tuple

        if kwargs or dynamic_img_pad or not bias:
            raise ValueError(f"ConvStemPatchEmbed supports only the plain fixed-size ViT configuration, got {kwargs}")
        self.img_size = to_2tuple(img_size)
        self.patch_size = to_2tuple(patch_size)
        if self.patch_size != (16, 16) or embed_dim % 8 or self.img_size[0] % 16 or self.img_size[1] % 16:
            raise ValueError("ConvStemPatchEmbed is defined for patch_size=16, embed_dim % 8 == 0 and img_size % 16 == 0")
        self.grid_size = (self.img_size[0] // 16, self.img_size[1] // 16)
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        widths = [embed_dim // 8, embed_dim // 4, embed_dim // 2, embed_dim]
        layers: list[nn.Module] = []
        previous = in_chans
        for width in widths:
            layers += [nn.Conv2d(previous, width, kernel_size=3, stride=2, padding=1), LayerNorm2d(width), nn.GELU()]
            previous = width
        layers.append(nn.Conv2d(previous, embed_dim, kernel_size=1))
        self.stem = nn.Sequential(*layers)

    def feat_ratio(self, as_scalar: bool = True) -> int | tuple[int, int]:
        return self.patch_size[0] if as_scalar else self.patch_size

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        height, width = images.shape[-2:]
        if (height, width) != self.img_size:
            raise ValueError(f"input size ({height}x{width}) does not match the model ({self.img_size[0]}x{self.img_size[1]})")
        return self.stem(images).flatten(2).transpose(1, 2)


def build_deit_tiny_convstem(*, num_classes: int) -> nn.Module:
    import timm

    return timm.create_model("deit_tiny_patch16_224", pretrained=False, num_classes=num_classes, embed_layer=ConvStemPatchEmbed)


CONVNEXT_ATTO_DEEP_NARROW_DEPTHS = (3, 3, 8, 1)
CONVNEXT_ATTO_DEEP_NARROW_DIMS = (36, 72, 144, 288)


def build_convnext_atto_deep_narrow(*, num_classes: int) -> nn.Module:
    import timm

    return timm.create_model(
        "convnext_atto",
        pretrained=False,
        num_classes=num_classes,
        depths=CONVNEXT_ATTO_DEEP_NARROW_DEPTHS,
        dims=CONVNEXT_ATTO_DEEP_NARROW_DIMS,
    )


def build_convnext_atto_ols(*, num_classes: int) -> nn.Module:
    """timm ``convnext_atto_ols``: ``stem_type='overlap_tiered'`` -- conv3x3/s2 (3->24) then conv3x3/s2
    (24->40), with NO norm or activation between them and one LayerNorm2d after. Its stem is
    therefore an overlapping *linear* stem, not Singh/Croce/Hein's ConvStem (which puts LN + GELU after
    each conv); that one is ``convnext_atto_convstem_imagenet`` below."""
    import timm

    return timm.create_model("convnext_atto_ols", pretrained=False, num_classes=num_classes)


class ConvNeXtConvStem(nn.Module):
    """Singh/Croce/Hein 2023 ConvStem for ConvNeXt, scaled to Atto.

    Same structure as their released ``ConvBlock1`` (``.external/robustbench/robustbench/model_zoo/
    architectures/convstem_models.py``; ``convnext_t_cvst`` uses ``ConvBlock1(48)`` for ConvNeXt-T's
    96-wide stage 0): conv3x3/s2 (3 -> w/2), channels-first LayerNorm (eps 1e-6), GELU, conv3x3/s2
    (w/2 -> w), LayerNorm, GELU, where w = stage-0 width (40 for Atto, so ``ConvBlock1(20)``). Replaces
    the whole timm stem (4x4/s4 patchify conv + LayerNorm2d) as their code does; total stride 4 is
    unchanged, so every stage sees the same spatial sizes. timm ``LayerNorm2d`` computes the same
    channels-first LayerNorm as their hand-written one.
    """

    def __init__(self, in_chans: int, width: int) -> None:
        super().__init__()
        from timm.layers import LayerNorm2d

        if width % 2:
            raise ValueError("ConvNeXtConvStem needs an even stage-0 width")
        half = width // 2
        self.stem = nn.Sequential(
            nn.Conv2d(in_chans, half, kernel_size=3, stride=2, padding=1),
            LayerNorm2d(half, eps=1e-6),
            nn.GELU(),
            nn.Conv2d(half, width, kernel_size=3, stride=2, padding=1),
            LayerNorm2d(width, eps=1e-6),
            nn.GELU(),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.stem(images)


def build_convnext_atto_convstem(*, num_classes: int) -> nn.Module:
    """ConvNeXt-Atto with its stem replaced after construction, as ``get_convstem_models`` does (the new
    stem keeps PyTorch's default conv init, like theirs)."""
    import timm

    model = timm.create_model("convnext_atto", pretrained=False, num_classes=num_classes)
    model.stem = ConvNeXtConvStem(3, model.stages[0].blocks[0].conv_dw.in_channels)
    return model


# id -> builder(num_classes) for every Phase 2 variant; all random-init only.
VARIANT_BUILDERS = {
    "mobilenetv4_conv_small_silu_imagenet": partial(build_mobilenetv4_conv_small_variant, act_layer="silu"),
    "mobilenetv4_conv_small_gelu_imagenet": partial(build_mobilenetv4_conv_small_variant, act_layer="gelu"),
    "mobilenetv4_conv_small_se_imagenet": partial(
        build_mobilenetv4_conv_small_variant, se=True, num_features=MOBILENETV4_CONV_SMALL_SE_NUM_FEATURES
    ),
    "mobilenetv4_conv_small_silu_se_imagenet": partial(
        build_mobilenetv4_conv_small_variant,
        act_layer="silu",
        se=True,
        num_features=MOBILENETV4_CONV_SMALL_SE_NUM_FEATURES,
    ),
    "deit_tiny_convstem_imagenet": build_deit_tiny_convstem,
    "convnext_atto_deep_narrow_imagenet": build_convnext_atto_deep_narrow,
    "convnext_atto_ols_imagenet": build_convnext_atto_ols,
    "convnext_atto_convstem_imagenet": build_convnext_atto_convstem,
    # SE control with the untrimmed 1280-wide head (+6.61% params): the matched SE arm above changes
    # two things at once (adds SE, narrows the head 1280 -> 1152); this id isolates SE alone. Run only
    # if the matched SE effect lands in the ambiguous 0.5-1.5 pt band (reviewer, 2026-10-08).
    "mobilenetv4_conv_small_se_fullhead_imagenet": partial(build_mobilenetv4_conv_small_variant, se=True),
}
