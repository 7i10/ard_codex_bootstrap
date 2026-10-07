# ruff: noqa: E501
"""Existing configs keep byte-identical config hashes after plan 0103 Phase 2 batch D.

The golden digests were computed with the source at master 6b8a68a (before the
distillation fields existed) under the fixed environment below: every config
that has a teacher, an RSLAD-family method, or is a Phase 1 ImageNet cell.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ard.config.loader import load_config, resolved_config_dict
from ard.engine.checkpoint import config_digest

ROOT = Path(__file__).resolve().parents[2]
ENVIRONMENT = {
    "ARD_SEED": "7",
    "ARD_CIFAR10_ROOT": "/x/cifar10",
    "ARD_IMAGENET_ROOT": "/x/imagenet",
    "ARD_IMAGENET_TRAIN_ROOT": "/x/imagenet_train_derived",
    "ARD_IMAGENET100_ROOT": "/x/imagenet100",
    "ARD_TEACHER_CHEN2021_LTD_WRN34_10_CHECKPOINT": "/x/t.pt",
    "ARD_TEACHER_CHEN2021_LTD_WRN34_10_CHECKPOINT_SHA256": "a" * 64,
    "ARD_TEACHER_BARTOLDSON2024_ADVERSARIAL_WRN94_16_CHECKPOINT": "/x/t.pt",
    "ARD_TEACHER_BARTOLDSON2024_ADVERSARIAL_WRN94_16_CHECKPOINT_SHA256": "b" * 64,
    "ARD_PER_RANK_BATCH_SIZE": "128",
    "ARD_NUM_WORKERS": "0",
    "ARD_DEVICE": "cpu",
    "ARD_OUTPUT_ROOT": "/x/outputs",
    "ARD_JOB_OUTPUT_DIR": "/x/job-output",
    "ARD_RUN_ID": "config-test-run",
    "ARD_STAGEWISE_SWITCH_EPOCH": "100",
    "ARD_STAGEWISE_LATE_POLICY": "crop_re",
    "WANDB_ENTITY": "entity",
    "WANDB_GROUP_STAGEWISE": "stagewise-group",
    "WANDB_PROJECT": "project",
    "WANDB_GROUP_CHEN": "chen-group",
    "WANDB_GROUP_BARTOLDSON": "bartoldson-group",
    "WANDB_GROUP_BARTOLDSON_ORACLE": "bartoldson-oracle-group",
    "ARD_FROZEN_ORACLE_MANIFEST": "/x/frozen-oracle.json",
    "ARD_FROZEN_ORACLE_MANIFEST_SHA256": "c" * 64,
    "ARD_STAGE1_CHECKPOINT": "/x/stage1/last.pt",
    "ARD_STAGE1_CHECKPOINT_SHA256": "d" * 64,
}

GOLDEN = {
    "configs/experiments/synthetic_rslad.yaml": "338533df8003019eb239f92a30ae1465bf12cf329bc869b8c56b1bad7c752216",
    "configs/experiments/synthetic_rslad_entropy.yaml": "2aecea527a367c43830743e54f384af926246732b05c17783e51f982e474169c",
    "configs/experiments/synthetic_rslad_joint.yaml": "b7389290a86bd7456cb9735fe0d41314bea1745f67e79de488847185f9bd405b",
    "configs/experiments/synthetic_rslad_observed.yaml": "90a777312b260f13b5ad41cfa8dad207d8bac60451ebbae1445b8b1e30cd484d",
    "configs/experiments/synthetic_rslad_student.yaml": "afaf10567dd7abc5528be5bbb9826ca08eca1e3ea3e0b78d3dc5d9653c5b05bd",
    "configs/pilot/cifar10_r18_rslad_bartoldson2024_adversarial_wrn94_16.yaml": "7c0a3d46990269ddf10caff57d5326ead8d65c2df20a3654505378db7e83c892",
    "configs/pilot/cifar10_r18_rslad_chen2021_ltd_wrn34_10.yaml": "e1ef2257ee01242eeffac6cc45ad9867349ed519086289d7ab14b2c3cd2c3de1",
    "configs/pilot/single_gpu/cifar10_r18_rslad_bartoldson_1ep.yaml": "0947b4af591ca1d4d6b97bc704302b3fdbeb493486d59fb55cf0c34b0df19e33",
    "configs/pilot/single_gpu/cifar10_r18_rslad_chen_1ep.yaml": "91b81d42c4f3d097b16fcbe7ec95ec687e1c52084d1b68bd580cdfde4b537f6e",
    "configs/pilot/single_gpu/cifar10_r18_rslad_joint_chen_3ep.yaml": "55e34cd80792712b88cd03e8052a3700d863d331a7bae796225bf99117a5b9cc",
    "configs/production/cifar10_r18_rslad_bartoldson2024_adversarial_wrn94_16.yaml": "bcabf517b5a7c54d9ef12ca7bc651f0495a42738e75a4ba3085c6e938da26cd5",
    "configs/production/cifar10_r18_rslad_chen2021_ltd_wrn34_10.yaml": "a0a5c7d88161dc84eb8a0cdd8d421e073e361b1c4263670417f90315eb3f04fa",
    "configs/production/cifar10_r18_rslad_entropy_bartoldson2024_adversarial_wrn94_16.yaml": "599686b6b4b10562574c21032492159809989d8a528600c856b98c9956d4bdb7",
    "configs/production/cifar10_r18_rslad_entropy_chen2021_ltd_wrn34_10.yaml": "faa15220a64d53449efe1a12bae2b30d8cff9824c6f81987a524dd15cfaab552",
    "configs/production/cifar10_r18_rslad_joint_bartoldson2024_adversarial_wrn94_16.yaml": "162dddd8bb1179ceaef53670b0153ff7d78cb65afc5e5d4943e5a836ea50660c",
    "configs/production/cifar10_r18_rslad_joint_chen2021_ltd_wrn34_10.yaml": "0eeab413b173c6fa0c6007c05ca2a915fcc68fd59f091f438ed63db971eed955",
    "configs/production/cifar10_r18_rslad_student_bartoldson2024_adversarial_wrn94_16.yaml": "e3dfbfe9a9ff9b8945a584c809bed70c2d5b292d85840e14c98d66c6ff9016c2",
    "configs/production/cifar10_r18_rslad_student_chen2021_ltd_wrn34_10.yaml": "31aa31bb9aa2c228b1b90433fbd13565595faf978d8e78f216791dd05323ed89",
    "configs/production/single_gpu/cifar10_r18_rslad_bartoldson.yaml": "d1d6caa0234087f4f49f98e6f42998e65dcc37a2e14cfefb1ab8effeda32633e",
    "configs/production/single_gpu/cifar10_r18_rslad_chen.yaml": "b7d0c167315d8b24056f69f535b7ef41b1307695b0a20be23d56ff5674c62389",
    "configs/production/single_gpu/cifar10_r18_rslad_entropy_bartoldson.yaml": "04996cefa2e8dfb21affc3728b93e0b9936c68e026eedc428cf77faf3de0b6f1",
    "configs/production/single_gpu/cifar10_r18_rslad_entropy_chen.yaml": "0f43af454a64069a7687c870b12106f3b8446f5790800c13a434bca4b81d0211",
    "configs/production/single_gpu/cifar10_r18_rslad_joint_bartoldson.yaml": "a65d79fc10f1efb1a4461bc52f97032e3f8b11f59d6923e179b046ae8be210ab",
    "configs/production/single_gpu/cifar10_r18_rslad_joint_chen.yaml": "dc46cea44467f7ea9f1b33aa1c7da949beb315619cfd79e2f2c3876e1df9edd8",
    "configs/production/single_gpu/cifar10_r18_rslad_student_bartoldson.yaml": "da3c823a6937a50fe252a337c5a23210ae3782c419a15accf04db3961ea66a31",
    "configs/production/single_gpu/cifar10_r18_rslad_student_chen.yaml": "bab5b883c933ca69ecf05d901e3b63b08fed4aab5b88417fd0ff550d32bb97e6",
    "configs/scientific/cifar10_r18_rslad_crop_re_chen2021_ltd_wrn34_10.yaml": "fe35654cb1f20f11df6320e40df4c17efa61aaba1d82e712714348bdf9214bee",
    "configs/scientific/cifar10_r18_rslad_cropshift_chen2021_ltd_wrn34_10.yaml": "e2a904df530155e3d352afb9d01e53c4b9b191ce18a3438c61098027c59efed2",
    "configs/scientific/cifar10_r18_rslad_cropshift_prefix_chen2021_ltd_wrn34_10.yaml": "e7c145d07e948f59a7b89f390976e20a3fe43f26cc5d1e7e5ca721dd6465c3ff",
    "configs/scientific/cifar10_r18_rslad_idbh_weak_chen2021_ltd_wrn34_10.yaml": "eb4d4d830177483797b5ecf0186b311436f324f6a72f51b2fc8447f1eec50fed",
    "configs/scientific/cifar10_r18_rslad_observed_bartoldson.yaml": "d1d6caa0234087f4f49f98e6f42998e65dcc37a2e14cfefb1ab8effeda32633e",
    "configs/scientific/cifar10_r18_rslad_observed_chen.yaml": "b7d0c167315d8b24056f69f535b7ef41b1307695b0a20be23d56ff5674c62389",
    "configs/scientific/cifar10_r18_rslad_stagewise_augmentation_chen2021_ltd_wrn34_10.yaml": "7fcb8f14f7f1a8706fd1f550a6ba98d2c79b1d7419b8fb5e468f53db9142ee57",
    "configs/scientific/imagenet_convnext_atto_pgd_at_phase1_adamw_random_lr1em3.yaml": "858ba416e1ab8ffdc12b09106aba8ef4278c81dcba915243156e84329fb93164",
    "configs/scientific/imagenet_convnext_atto_pgd_at_phase1_adamw_random_lr1p25em4.yaml": "0513f10540be68ccb3224840179d6305c1b28b197ddccb7e7702aae57620b980",
    "configs/scientific/imagenet_convnext_atto_pgd_at_phase1_adamw_random_lr2em3.yaml": "ac48619122fabcbb0716fe1b6de43e22bf8ac0f0bc7005da05e1e237cf8a946e",
    "configs/scientific/imagenet_convnext_atto_pgd_at_phase1_adamw_random_lr2p5em4.yaml": "b6e22b3464e1c729735f0d8731f149c2fc2891cae9adbdf6312267ff1af7cd1b",
    "configs/scientific/imagenet_convnext_atto_pgd_at_phase1_adamw_random_lr5em4.yaml": "0aa0e37eff9ed6cd39efb0c797957b169f12ae8a1cc18e454548d96e2f264cb6",
    "configs/scientific/imagenet_convnext_atto_pgd_at_phase1_random.yaml": "21d7d84fae696c201c2888f3aec29725d43e0e56849007dd629c77bac188a743",
    "configs/scientific/imagenet_deit_tiny_pgd_at_phase1_adamw_random_lr1em3.yaml": "371fc43ea9c35807499af0eba17c56a8ffe10f6ccf0cd7930f46c51185fc3d08",
    "configs/scientific/imagenet_deit_tiny_pgd_at_phase1_adamw_random_lr1p25em4.yaml": "3ae6ae0db4dbac7622de31eca2f30689d093f86773bd1ceb4bd163da9dc052b1",
    "configs/scientific/imagenet_deit_tiny_pgd_at_phase1_adamw_random_lr2p5em4.yaml": "34eee272379aa2f7a391857788119db5d35daaa972bbdec42fcbc88699968893",
    "configs/scientific/imagenet_deit_tiny_pgd_at_phase1_adamw_random_lr5em4.yaml": "d65e69941df8009170d755adbe4a872f62dfee373196ed867ded0a437224a2b2",
    "configs/scientific/imagenet_deit_tiny_pgd_at_phase1_random.yaml": "736df81c8a9dc633da1912054c09b54af33a49eafe0b3359bb9b07a8a41928df",
    "configs/scientific/imagenet_efficientnet_b0_pgd_at_phase1_random.yaml": "963e50d3a81d562748bd50d854c9456964e8c0e5e0d9d248707edb70c5b64bc6",
    "configs/scientific/imagenet_efficientnet_b0_pgd_at_phase1_random_lr00125.yaml": "d97ca33b6cc0d173dbcbef44c1e03e26cb98363a1253d92b51f1f242bf3416d7",
    "configs/scientific/imagenet_efficientnet_b0_pgd_at_phase1_random_lr005.yaml": "c832d536d2001f7c2047318b4cb604540073db29644d9900b2f5298356981a52",
    "configs/scientific/imagenet_mobilenetv4_conv_medium_pgd_at_phase1_pretrained_lr0005.yaml": "86be1bdd56799f15a51a34f3a00bdc496e0ea71f83217bdc2fd2622b50ede388",
    "configs/scientific/imagenet_mobilenetv4_conv_medium_pgd_at_phase1_pretrained_lr0015.yaml": "62dd9eee5e96b5e73dfcefe616d2dbbe261743a7b9e8916f33c3ae815feda570",
    "configs/scientific/imagenet_mobilenetv4_conv_medium_pgd_at_phase1_pretrained_lr005.yaml": "0b990da95045b5879429b9f37917a6b44f78025a1027086f8ad4b028cb1649dd",
    "configs/scientific/imagenet_mobilenetv4_conv_medium_pgd_at_phase1_random.yaml": "1f01c7e12d9d8392ff4c68662125427a6e3f8a4b5106ec67403ee483a6a114d3",
    "configs/scientific/imagenet_mobilenetv4_conv_medium_pgd_at_phase1_random_lr00125.yaml": "6b3199cfbf9eb478188b6f70d9908c2f6985d1ce155eae2589c353091dfbae4a",
    "configs/scientific/imagenet_mobilenetv4_conv_medium_pgd_at_phase1_random_lr005.yaml": "3f3c6daaa21a20f950ae161f1fdec752bc82a328c7a233f08837dbb13a4e5d71",
    "configs/scientific/imagenet_mobilenetv4_pgd_at_random_init_lr0025_cg.yaml": "1621d65270539ac56702405c6d9ed414908fc64bfc62b5880efda0f76577df86",
    "configs/scientific/imagenet_mobilevit_s_pgd_at_phase1_random.yaml": "5e50ac5063e030ce4db230032ee832a09207b52ad6ebadfa6bcab77adf1ad2a7",
}


def test_existing_config_hashes_are_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    for key, value in ENVIRONMENT.items():
        monkeypatch.setenv(key, value)
    changed = {}
    for relative, expected in GOLDEN.items():
        resolved = resolved_config_dict(load_config(ROOT / relative))
        assert "distillation" not in resolved
        observed = config_digest(resolved)
        if observed != expected:
            changed[relative] = observed
    assert not changed
