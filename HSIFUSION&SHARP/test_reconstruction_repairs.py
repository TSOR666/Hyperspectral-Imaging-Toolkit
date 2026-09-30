"""Mechanism and counter tests for the assessment repairs (CPU, synthetic data)."""
import pytest
import torch
from torch import nn

from reconstruction_layers import FullResolutionSpectralBlock, unpack_grouped_projection
from sharp_v322_hardened import (ImprovedCrossAttentionFusion, MultiScaleAttention,
                                OptimizedSparseAttention)
from hsifusion_v252_complete import RobustEnhancedSpectralAttention
from optimized_dataloader import MSTPlusPlusLoss, OptimizedTrainDataset, OptimizedValDataset
from unified_training import UnifiedTrainer, UnifiedTrainingConfig, build_model, parse_args
from unified_inference import load_checkpoint_model
from test_unified_infra import mini_arad


def small_kwargs(family):
    if family == 'sharp':
        return dict(base_dim=8, depths=[1, 1, 1, 1], heads=[1, 2, 4, 8],
                    mlp_ratios=[2., 2., 2., 2.], drop_path_rate=0., max_global_tokens=16,
                    sparse_window_size=9, sparse_landmark_tokens=4, sparse_q_block_size=64)
    return dict(base_channels=8, depths=[1, 1, 1, 1], num_heads=1, mlp_ratio=2.,
                drop_path=0., use_moe=False, cross_attention_max_tokens=16)


@pytest.mark.parametrize('parts', [2, 3])
def test_grouped_projection_all_parts_have_the_same_input_support(parts):
    conv = nn.Conv2d(8, 8 * parts, 1, groups=4, bias=False)
    nn.init.ones_(conv.weight)
    x = torch.zeros(1, 8, 2, 2)
    x[:, :2] = 1
    corrected = unpack_grouped_projection(conv(x), 4, parts, 'group_major')
    for value in corrected:
        assert torch.all(value[:, :2] == 2)
        assert torch.count_nonzero(value[:, 2:]) == 0
    assert not torch.equal(corrected[0], unpack_grouped_projection(conv(x), 4, parts)[0])


@pytest.mark.parametrize('kind', ['multiscale', 'sparse', 'pooled_spectral', 'skip_kv'])
def test_corrected_layout_is_wired_to_each_grouped_projection(kind):
    if kind == 'multiscale':
        module = MultiScaleAttention(16, 4, max_global_tokens=4, qkv_layout='group_major')
    elif kind == 'sparse':
        module = OptimizedSparseAttention(16, 4, qkv_layout='group_major')
    elif kind == 'pooled_spectral':
        module = RobustEnhancedSpectralAttention(16, qkv_layout='group_major', strict_failures=True)
    else:
        module = ImprovedCrossAttentionFusion(16, 4, max_global_tokens=4, qkv_layout='group_major')
    x = torch.rand(1, 16, 4, 4, requires_grad=True)
    y = module(x, x) if kind == 'skip_kv' else module(x)
    y.square().mean().backward()
    assert module.qkv_layout == 'group_major'
    assert torch.isfinite(y).all() and torch.isfinite(x.grad).all()


def test_aligned_skip_transmits_a_local_detail_without_pooling():
    fusion = ImprovedCrossAttentionFusion(8, 2, max_global_tokens=4, aligned_skip=True)
    nn.init.zeros_(fusion.proj.weight)
    nn.init.zeros_(fusion.proj.bias)
    nn.init.dirac_(fusion.skip_proj.weight)
    nn.init.zeros_(fusion.skip_proj.bias)
    x, skip = torch.zeros(1, 8, 8, 8), torch.zeros(1, 8, 8, 8)
    baseline = fusion(x, skip)
    skip[:, 3, 2, 5] = 0.7
    delta = fusion(x, skip) - baseline
    assert delta[0, 3, 2, 5].item() == pytest.approx(0.7)
    assert torch.count_nonzero(delta) == 1


@pytest.mark.parametrize('shape', [(8, 8), (16, 24)])
def test_fixed_local_operator_does_not_switch_with_resolution(shape):
    module = OptimizedSparseAttention(8, 2, attention_mode='local_landmark',
                                      window_size=9, landmark_tokens=4, q_block_size=32)
    assert module.attention_operator(*shape) == 'local_landmark'
    output = module(torch.rand(1, 8, *shape))
    assert output.shape[-2:] == shape and torch.isfinite(output).all()


def test_fixed_exact_operator_refuses_to_silently_fall_back():
    module = OptimizedSparseAttention(8, 2, attention_mode='exact_topk', exact_topk_max_tokens=16)
    with pytest.raises(ValueError, match='token limit'):
        module(torch.rand(1, 8, 8, 8))


def test_full_resolution_spectral_branches_learn_on_first_update():
    block = FullResolutionSpectralBlock(8, gate_init=0.01)
    x = torch.rand(1, 8, 8, 8)
    block(x).square().mean().backward()
    for parameter in (block.qkv.weight, block.ffn_in.weight, block.gamma, block.gamma2):
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0


def test_strict_spectral_failure_is_observable(monkeypatch):
    module = RobustEnhancedSpectralAttention(16, strict_failures=True)
    def fail(*args):
        raise RuntimeError('projection failed')
    monkeypatch.setattr(module.scale_convs[0], 'forward', fail)
    with pytest.raises(RuntimeError, match='projection failed'):
        module(torch.rand(1, 16, 4, 4))
    with pytest.raises(FloatingPointError, match='input'):
        module(torch.full((1, 16, 4, 4), float('nan')))


def test_exact_mrae_matches_reference_and_rejects_zero_targets():
    target = torch.tensor([1e-5, .1, .8]).view(1, 3, 1, 1)
    pred = torch.tensor([2e-5, .2, .4]).view_as(target).requires_grad_()
    loss = MSTPlusPlusLoss(eps=0)(pred, target)
    assert loss.item() == pytest.approx(2.5 / 3)
    loss.backward()
    assert torch.isfinite(pred.grad).all()
    with pytest.raises(ValueError, match='strictly positive'):
        MSTPlusPlusLoss(eps=0)(pred, torch.zeros_like(target))


def test_dense_linear_multiscale_projection_handles_nchw():
    module = MultiScaleAttention(8, 2, use_conv_proj=False, max_global_tokens=4)
    x = torch.rand(1, 8, 4, 6, requires_grad=True)
    y = module(x)
    y.square().mean().backward()
    assert y.shape == x.shape and torch.isfinite(x.grad).all()


def test_reference_floor_patch_grid_has_no_shifted_edge_duplicates(mini_arad):
    floor = OptimizedTrainDataset(str(mini_arad), crop_size=32, stride=13, memory_mode='lazy',
                                  patch_grid='floor', augment=False)
    ceiling = OptimizedTrainDataset(str(mini_arad), crop_size=32, stride=13, memory_mode='lazy',
                                    patch_grid='ceil', augment=False)
    assert floor.patches_per_image == [9, 9]
    assert ceiling.patches_per_image == [16, 16]
    assert floor._get_image_and_patch(8)[1:3] == (26, 26)


def test_validation_failure_restores_ema_weights_and_training_mode(mini_arad, tmp_path, monkeypatch):
    import unified_training
    cfg = UnifiedTrainingConfig(model='sharp', model_size='tiny', model_kwargs=small_kwargs('sharp'),
                                data_root=str(mini_arad), output_dir=str(tmp_path),
                                batch_size=1, patch_size=64, stride=64, num_workers=0,
                                max_optimizer_steps=1, amp='off', device='cpu', ema_decay=.9)
    trainer = UnifiedTrainer(cfg)
    trainer.model.train()
    before = {name: value.clone() for name, value in trainer._orig_model.state_dict().items()}
    for value in trainer.ema_state.values():
        value.add_(.1)
    def fail(*args):
        raise RuntimeError('validation failed')
    monkeypatch.setattr(unified_training, 'evaluate_scene', fail)
    with pytest.raises(RuntimeError, match='validation failed'):
        trainer.validate(0)
    assert trainer.model.training
    for name, value in trainer._orig_model.state_dict().items():
        assert torch.equal(value, before[name])


def test_validation_preserves_targets_even_when_training_cache_is_half(mini_arad):
    import h5py
    import numpy as np
    dataset = OptimizedValDataset(str(mini_arad), memory_mode='float16')
    _, truth = dataset[0]
    with h5py.File(dataset.hsi_files[0]) as handle:
        source = np.asarray(handle['cube'], dtype=np.float32).transpose(0, 2, 1)
    assert np.array_equal(truth.numpy(), source)
    assert dataset.hsi_data[0].dtype == np.float32


def test_missing_pair_requires_an_explicit_exclusion(mini_arad, monkeypatch):
    import builtins
    import io
    from optimized_dataloader import _paired_paths
    original_open = builtins.open
    def split_with_missing(path, *args, **kwargs):
        if str(path).endswith('train_list.txt'):
            return io.StringIO('ARAD_1K_0001\nmissing_scene\n')
        return original_open(path, *args, **kwargs)
    monkeypatch.setattr(builtins, 'open', split_with_missing)
    with pytest.raises(FileNotFoundError, match='Missing RGB/HSI pair'):
        _paired_paths(str(mini_arad), 'train')
    rgb, hsi = _paired_paths(str(mini_arad), 'train', exclude_samples=['missing_scene'])
    assert len(rgb) == len(hsi) == 1


@pytest.mark.parametrize('family', ['sharp', 'hsifusion'])
def test_radiometry_paths_respond_to_brightness_and_backpropagate(family):
    kwargs = dict(small_kwargs(family), conv_norm=False, rgb_skip=True, fullres_spectral=True)
    if family == 'sharp':
        kwargs.update(head_mode='linear', output_activation='none', spectral_head_rank=0,
                      qkv_layout='group_major', aligned_skip=True, sparse_attention_mode='local_landmark')
    else:
        kwargs.update(qkv_layout='group_major', strict_spectral_failures=True, layer_scale_init=.01)
    model = build_model(family, 'tiny', kwargs).eval()
    rgb = torch.rand(1, 3, 64, 64)
    a, b = model(rgb), model(.5 * rgb)
    assert (a - b).square().mean().sqrt() > 1e-3
    a.square().mean().backward()
    assert model.input_skip.weight.grad.abs().sum() > 0
    assert model.spectral_output.qkv.weight.grad.abs().sum() > 0


@pytest.mark.parametrize('family', ['sharp', 'hsifusion'])
def test_successful_update_budget_accumulation_and_checkpoint_roundtrip(family, mini_arad, tmp_path):
    kwargs = dict(small_kwargs(family), qkv_layout='group_major')
    cfg = UnifiedTrainingConfig(model=family, model_size='tiny', model_kwargs=kwargs,
                                data_root=str(mini_arad), output_dir=str(tmp_path),
                                experiment_name=family, batch_size=1, patch_size=64, stride=64,
                                num_workers=0, memory_mode='lazy', max_optimizer_steps=3,
                                updates_per_epoch=2, val_interval_steps=2, val_crop_border=0,
                                checkpoint_interval_steps=1, accumulate_steps=2, health_interval_steps=2,
                                amp='off', device='cpu', loss_mode='exact', mrae_eps=0,
                                auxiliary_loss=False, gradient_clip=0)
    trainer = UnifiedTrainer(cfg)
    trainer.train()
    assert trainer.optimizer_step == trainer.scheduler.last_epoch == 3
    assert trainer.iteration == 6 and trainer.loader_cycles == 2
    checkpoint = cfg.experiment_path() / 'last.pth'
    payload = torch.load(checkpoint, weights_only=False)
    assert payload['resolved_model_config']['qkv_layout'] == 'group_major'
    assert payload['run_manifest']['data']['validation_storage'] == 'float32'
    assert payload['total_optimizer_steps'] == 3
    loaded, _ = load_checkpoint_model(str(checkpoint))
    from pathlib import Path
    from hsi_benchmark.models import ModelRequest, load_model_adapter
    adapter = load_model_adapter(ModelRequest(family, 'auto', checkpoint),
                                 repository_root=Path(__file__).resolve().parent.parent,
                                 device=torch.device('cpu'), mst_root=None, trust_checkpoint=False,
                                 allow_partial=False, prefer_ema=False, use_amp=False,
                                 normalization_override='auto', sampling_steps=20, latent_mode='mean')
    assert adapter.normalization == 'unit'
    trainer.model.eval()
    rgb = torch.rand(1, 3, 64, 64)
    with torch.no_grad():
        assert torch.equal(loaded(rgb), trainer.model(rgb))
        assert torch.equal(adapter.model(rgb), trainer.model(rgb))
    cfg.resume_from = str(checkpoint)
    resumed = UnifiedTrainer(cfg)
    assert resumed.optimizer_step == 3
    resumed.train()  # A completed budget must not take another step.
    cfg.model_kwargs['qkv_layout'] = 'legacy'
    with pytest.raises(ValueError, match='semantics'):
        UnifiedTrainer(cfg)
    cfg.model_kwargs['qkv_layout'] = 'group_major'
    cfg.rgb_normalization = 'scene_minmax'
    with pytest.raises(ValueError, match='data split/preprocessing'):
        UnifiedTrainer(cfg)


def test_recipe_overrides_are_explicit_and_default_budget_is_loader_independent():
    from pathlib import Path
    for path in (Path(__file__).parent / 'recipes').glob('*.json'):
        cfg = parse_args(['--recipe', str(path), '--max_optimizer_steps', '3', '--model_size', 'tiny'])
        assert cfg.max_optimizer_steps == 3 and cfg.model_size == 'tiny'
        assert cfg.loss_mode == 'exact' and cfg.memory_mode == 'lazy'
    assert UnifiedTrainingConfig().max_optimizer_steps == 300000
    with pytest.raises(ValueError, match='FP32'):
        UnifiedTrainingConfig(loss_mode='exact')


@pytest.mark.parametrize('family', ['sharp', 'hsifusion'])
def test_shared_reference_builder_and_odd_image_validation(family, tmp_path):
    import json
    from diagnose_reconstruction import build_model as build_reference_model
    from train_reconstruction_reference import validate_reference
    config = tmp_path / 'recipe.json'
    config.write_text(json.dumps({'model': family, 'model_size': 'tiny',
                                  'model_kwargs': small_kwargs(family)}))
    model, resolved = build_reference_model(family, config)
    assert model._requires_reconstruction_padding
    assert resolved['in_channels'] == 3
    rgb, target = torch.rand(1, 3, 63, 67), torch.full((1, 31, 63, 67), .5)
    model.train()
    result = validate_reference(model, [(rgb, target)], device='cpu', border=0)
    assert result['raw_mrae'] >= 0 and model.training
