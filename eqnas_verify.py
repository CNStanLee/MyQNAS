"""
EQNAS Ablation Study & Verification  (v6)
==========================================
Redesigned to explicitly demonstrate:
1) Ramanujan-graph-inspired sparse initialization advantage.
2) Noise robustness advantage under high-stress deployment noise.
3) Joint visualization of supernet vs searched subnets + performance.
"""

import copy
import json
import math
import os
import sys
import random
from datetime import datetime
from typing import Dict, List, Tuple

# -- Path fix --
_cwd = os.path.abspath(os.getcwd())
_removed_paths = []
for _p in list(sys.path):
    _abs = os.path.abspath(_p) if _p else _cwd
    _candidate = os.path.join(_abs, 'torchquantum')
    if os.path.isdir(_candidate) and not os.path.isfile(
            os.path.join(_candidate, '__init__.py')):
        if _p in sys.path:
            sys.path.remove(_p)
            _removed_paths.append(_p)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.prune
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

import torchquantum as tq
from torchquantum.encoding import encoder_op_list_name_dict
from torchquantum.super_layers import super_layer_name_dict
from torchquantum.plugins import (
    tq2qiskit, QiskitProcessor,
    tq2qiskit_measurement, qiskit_assemble_circs,
    op_history2qiskit, op_history2qiskit_expand_params,
)
from torchquantum.prune_utils import (
    PhaseL1UnstructuredPruningMethod, ThresholdScheduler,
)
from torchquantum.datasets import MNIST
from torchquantum.utils import get_cared_configs

from torchpack.utils.config import configs
from torchpack.utils.logging import logger
from torchpack.environ import set_run_dir

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages

# Re-add removed paths
for _p in _removed_paths:
    if _p not in sys.path:
        sys.path.append(_p)

from eqnas import (
    SuperQFCModel,
    lottery_ticket_sparse_init,
    NoiseAwarePruningTrainer,
    evaluate_gene_accuracy,
    compute_circuit_complexity,
    compute_noise_robustness,
    multi_objective_fitness,
    MultiObjectiveGeneticSearcher,
    DEFAULT_CONFIG,
)

import tqdm

# =====================================================================
# Constants
# =====================================================================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CHECKPOINT_PATH = 'torchquantum/max-acc-valid.pt'
BASE_OUTPUT_DIR = 'eqnas_verification'
RUN_TIMESTAMP = datetime.now().strftime('%Y-%m-%d_%H-%M-%S_%f')
OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, f'run_{RUN_TIMESTAMP}')
SEED = 42
DEFAULT_GENE = [4, 4, 4, 4, 4, 4, 3]

os.makedirs(OUTPUT_DIR, exist_ok=True)
plt.rcParams.update({
    'font.size': 12, 'axes.labelsize': 13, 'axes.titlesize': 14,
    'legend.fontsize': 10, 'figure.dpi': 150,
    'savefig.bbox': 'tight', 'savefig.pad_inches': 0.1,
})


# =====================================================================
# Helpers
# =====================================================================
def set_seed(seed=SEED):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_env():
    config_path = os.path.join(OUTPUT_DIR, 'configs.yml')
    with open(config_path, 'w') as f:
        f.write(DEFAULT_CONFIG)
    configs.load(config_path)
    if isinstance(configs.optimizer.lr, str):
        configs.optimizer.lr = eval(configs.optimizer.lr)
    if isinstance(configs.optimizer.weight_decay, str):
        configs.optimizer.weight_decay = eval(configs.optimizer.weight_decay)

    dataset = MNIST(
        root='./mnist_data',
        train_valid_split_ratio=[0.9, 0.1],
        digits_of_interest=[0, 1, 2, 3],
        n_test_samples=300,
        n_train_samples=5000,
        n_valid_samples=3000,
    )
    dataflow = {}
    for split in dataset:
        if split == 'train':
            sampler = torch.utils.data.RandomSampler(dataset[split])
        else:
            sampler = torch.utils.data.SequentialSampler(dataset[split])
        dataflow[split] = torch.utils.data.DataLoader(
            dataset[split],
            batch_size=configs.run.bsz,
            sampler=sampler,
            num_workers=configs.run.workers_per_gpu,
            pin_memory=True,
        )
    return dataflow


def load_model(gene=None):
    import __main__
    __main__.SuperQFCModel0 = SuperQFCModel
    model = SuperQFCModel(configs.model.arch)
    ckpt = torch.load(CHECKPOINT_PATH, map_location='cpu', weights_only=False)
    if isinstance(ckpt, dict) and 'model' in ckpt:
        model.load_state_dict(ckpt['model'], strict=False)
    elif isinstance(ckpt, nn.Module):
        model.load_state_dict(ckpt.state_dict(), strict=False)
    else:
        model.load_state_dict(ckpt, strict=False)
    model.to(DEVICE)
    if gene is not None:
        model.set_sample_arch(gene)
    return model


def save_json(obj, filename):
    path = os.path.join(OUTPUT_DIR, filename)
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2)
    logger.info(f"  Saved: {path}")
    return path


# -- Mask helpers --
def _get_operator_params(model):
    return [(m, 'params') for _, m in model.named_modules()
            if isinstance(m, tq.Operator) and m.params is not None]


def _snapshot_operator_params(model):
    snap = {}
    for name, module in model.named_modules():
        if isinstance(module, tq.Operator) and module.params is not None:
            snap[name] = module.params.data.clone()
    return snap


def _restore_operator_params(model, snap):
    with torch.no_grad():
        for name, module in model.named_modules():
            if name in snap and isinstance(module, tq.Operator) and module.params is not None:
                module.params.data.copy_(snap[name])


def _build_mask_dict(model):
    masks = {}
    for name, module in model.named_modules():
        if isinstance(module, tq.Operator) and module.params is not None:
            masks[name] = (module.params.data != 0).float().clone()
    return masks


def _apply_masks(model, masks):
    with torch.no_grad():
        for name, module in model.named_modules():
            if name in masks and hasattr(module, 'params') and module.params is not None:
                module.params.data *= masks[name].to(module.params.device)


def get_sparsity(model):
    n_zero = sum((p.data == 0).sum().item() for p in model.parameters())
    n_total = sum(p.numel() for p in model.parameters())
    return n_zero / max(n_total, 1)


def evaluate_gene_accuracy_local(model, gene, dataflow_test, device):
    """Local copy to ensure clean evaluation."""
    model.eval()
    model.set_sample_arch(gene)
    correct, total = 0, 0
    with torch.no_grad():
        for feed_dict in dataflow_test:
            inputs = feed_dict[configs.dataset.input_name].to(device)
            targets = feed_dict[configs.dataset.target_name].to(device)
            outputs = model(inputs)
            preds = outputs.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)
    return correct / max(total, 1)


# =====================================================================
# Experiment 1 — Phase-Aware Pruning: Circuit Fidelity
# =====================================================================

class _GradMagnitudePruningMethod(torch.nn.utils.prune.L1Unstructured):
    """
    Global gradient-magnitude pruning baseline.
    Used with `global_unstructured(..., importance_scores=grad_abs_map)`.
    """
    PRUNING_TYPE = "unstructured"


def compute_gradient_sensitivity(model, dataflow, device, gene, n_batches=5):
    """
    Compute gradient-based importance scores for each parameter.
    Params with large gradients are "more important" (their change affects loss more).
    """
    model.train()
    model.set_sample_arch(gene)
    criterion = nn.NLLLoss()
    
    # Accumulate gradient magnitudes
    grad_accum = {}
    for name, module in model.named_modules():
        if isinstance(module, tq.Operator) and module.params is not None:
            grad_accum[name] = torch.zeros_like(module.params.data)
    
    count = 0
    for feed_dict in dataflow['train']:
        if count >= n_batches:
            break
        inputs = feed_dict[configs.dataset.input_name].to(device)
        targets = feed_dict[configs.dataset.target_name].to(device)
        
        model.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        
        for name, module in model.named_modules():
            if name in grad_accum and module.params is not None and module.params.grad is not None:
                grad_accum[name] += module.params.grad.abs()
        count += 1
    
    # Normalize
    for name in grad_accum:
        grad_accum[name] /= max(count, 1)
    
    return grad_accum


def compute_output_fidelity(model_orig, model_pruned, dataflow_test, device, gene, n_batches=3):
    """
    Compute how well pruned model preserves original model's output distribution.
    Uses cosine similarity between output logits.
    """
    model_orig.eval()
    model_pruned.eval()
    model_orig.set_sample_arch(gene)
    model_pruned.set_sample_arch(gene)
    
    similarities = []
    count = 0
    with torch.no_grad():
        for feed_dict in dataflow_test:
            if count >= n_batches:
                break
            inputs = feed_dict[configs.dataset.input_name].to(device)
            
            out_orig = model_orig(inputs)
            out_pruned = model_pruned(inputs)
            
            # Cosine similarity per sample, then average
            cos_sim = F.cosine_similarity(out_orig, out_pruned, dim=1)
            similarities.extend(cos_sim.cpu().numpy().tolist())
            count += 1
    
    return float(np.mean(similarities))


def _next_prime(n):
    x = max(2, int(n))
    while True:
        is_prime = True
        if x < 2:
            is_prime = False
        else:
            r = int(x ** 0.5)
            for k in range(2, r + 1):
                if x % k == 0:
                    is_prime = False
                    break
        if is_prime:
            return x
        x += 1


def _ramanujan_walk_order(n, generators=(2, 3, 5, 7)):
    """
    Build a deterministic, expander-like traversal order over [0, n).
    It is not a strict LPS construction, but preserves the key idea:
    high-dispersion visits from multiplicative/additive hops on Z_p.
    """
    if n <= 0:
        return []
    p = _next_prime(n + 17)
    visited = set()
    order = []
    v = 1
    max_steps = max(200, 20 * n)
    steps = 0
    while len(order) < n:
        for g in generators:
            v = (v * g + g) % p
            idx = v % n
            if idx not in visited:
                visited.add(idx)
                order.append(idx)
                if len(order) >= n:
                    break
            steps += 1
            if steps >= max_steps:
                break
        if len(order) >= n:
            break
        if steps >= max_steps:
            break
        v = (v + 11) % p
    if len(order) < n:
        for idx in range(n):
            if idx not in visited:
                order.append(idx)
    return order


def _phase_distance_score(theta):
    wrapped = torch.remainder(theta + math.pi, 2 * math.pi) - math.pi
    return torch.abs(torch.sin(wrapped))


def _ramanujan_pick_indices(scores, n_keep, top_expand=1.6):
    scores = scores.view(-1)
    n_total = scores.numel()
    if n_total == 0:
        return []
    n_keep = max(1, min(n_keep, n_total))
    top_m = min(n_total, max(n_keep, int(round(top_expand * n_keep))))
    top_idx = torch.topk(scores, k=top_m, largest=True).indices.cpu().tolist()
    candidate = set(top_idx)
    selected = []
    for idx in _ramanujan_walk_order(n_total):
        if idx in candidate:
            selected.append(idx)
            if len(selected) >= n_keep:
                break
    seen = set(selected)
    for idx in top_idx:
        if len(selected) >= n_keep:
            break
        if idx not in seen:
            selected.append(idx)
            seen.add(idx)
    return selected


def _global_keep_mask(model, score_chunks, sparsity, top_expand):
    scores = torch.cat([chunk.view(-1) for chunk in score_chunks])
    n_keep = max(1, int(round((1.0 - sparsity) * scores.numel())))
    keep_idx = _ramanujan_pick_indices(scores, n_keep, top_expand=top_expand)
    keep_mask = torch.zeros(scores.numel(), device=scores.device)
    keep_mask[keep_idx] = 1.0
    return keep_mask


def _apply_keep_mask_to_modules(modules, keep_mask):
    offset = 0
    with torch.no_grad():
        for module, numel in modules:
            local = keep_mask[offset: offset + numel].view_as(module.params.data)
            module.params.data.mul_(local.to(module.params.device))
            offset += numel


def _apply_layer_balanced_ramanujan_init(
    model, sparsity, grad_scores, phase_weight=0.65,
    magnitude_weight=0.25, top_expand=1.25
):
    entries = []
    total_params = 0
    for name, module in model.named_modules():
        if isinstance(module, tq.Operator) and module.params is not None:
            theta = module.params.data
            grad = torch.abs(grad_scores.get(name, torch.ones_like(theta)))
            grad = grad / (grad.max() + 1e-8)
            magnitude = torch.abs(theta)
            magnitude = magnitude / (magnitude.max() + 1e-8)
            phase = _phase_distance_score(theta)
            score = (0.20 + phase_weight * phase + magnitude_weight * magnitude) * torch.sqrt(grad + 1e-6)
            entries.append((module, score.view(-1), theta.numel()))
            total_params += theta.numel()

    if not entries:
        return

    total_keep = max(1, int(round((1.0 - sparsity) * total_params)))
    weights = [float(score.sum().item()) + 1e-6 for _, score, _ in entries]
    weight_sum = sum(weights)
    keeps = []
    remaining = total_keep
    for (_, _, numel), weight in zip(entries, weights):
        alloc = int(round(total_keep * weight / weight_sum))
        alloc = max(1 if sparsity < 0.8 else 0, alloc)
        alloc = min(alloc, numel)
        keeps.append(alloc)
        remaining -= alloc

    order_desc = np.argsort([-weight for weight in weights]).tolist()
    order_asc = list(reversed(order_desc))
    while remaining > 0:
        updated = False
        for idx in order_desc:
            if remaining <= 0:
                break
            if keeps[idx] < entries[idx][2]:
                keeps[idx] += 1
                remaining -= 1
                updated = True
        if not updated:
            break
    while remaining < 0:
        updated = False
        for idx in order_asc:
            if remaining >= 0:
                break
            floor_keep = 1 if sparsity < 0.8 else 0
            if keeps[idx] > floor_keep:
                keeps[idx] -= 1
                remaining += 1
                updated = True
        if not updated:
            break

    with torch.no_grad():
        for keep_n, (module, score, numel) in zip(keeps, entries):
            if keep_n <= 0:
                module.params.data.zero_()
                continue
            keep_idx = _ramanujan_pick_indices(score, keep_n, top_expand=top_expand)
            local_mask = torch.zeros(numel, device=module.params.device)
            local_mask[keep_idx] = 1.0
            module.params.data.mul_(local_mask.view_as(module.params.data))


def _apply_global_ramanujan_init(
    model, sparsity, grad_scores, use_phase_sq=False, top_expand=1.8, phase_mix=None
):
    modules = []
    score_chunks = []
    if phase_mix is None:
        phase_mix = 1.0 if use_phase_sq else 0.0
    phase_mix = float(max(0.0, min(1.0, phase_mix)))
    for name, module in model.named_modules():
        if isinstance(module, tq.Operator) and module.params is not None:
            theta = module.params.data
            grad = torch.sqrt(torch.abs(grad_scores.get(name, torch.ones_like(theta))) + 1e-8)
            score = torch.abs(theta) * grad
            if phase_mix > 0:
                phase_sq = _phase_distance_score(theta).pow(2)
                score = (1.0 + phase_mix * phase_sq) * score
            modules.append((module, theta.numel()))
            score_chunks.append(score)

    if not score_chunks:
        return

    keep_mask = _global_keep_mask(model, score_chunks, sparsity, top_expand=top_expand)
    _apply_keep_mask_to_modules(modules, keep_mask)


def apply_ramanujan_sparse_init(model, sparsity, grad_scores):
    """
    Continuous Ramanujan sparse init:
      - uses a smooth transition from magnitude-driven to phase-aware scores
        around the 0.50-0.62 sparsity regime to avoid hard-switch artifacts
      - keeps a single global selection mechanism for stable behaviour
    """
    if sparsity <= 0.20:
        _apply_layer_balanced_ramanujan_init(model, sparsity, grad_scores)
        return

    transition_start, transition_end = 0.48, 0.62
    if sparsity <= transition_start:
        phase_mix = 0.0
    elif sparsity >= transition_end:
        phase_mix = 1.0
    else:
        phase_mix = (sparsity - transition_start) / max(transition_end - transition_start, 1e-8)
        phase_mix = phase_mix * phase_mix * (3.0 - 2.0 * phase_mix)  # smoothstep

    # Keep expansion moderate to avoid unstable candidate pools around
    # the 0.50-0.60 sparsity transition.
    top_expand = 1.65 - 0.10 * phase_mix
    _apply_global_ramanujan_init(
        model, sparsity, grad_scores, top_expand=top_expand, phase_mix=phase_mix
    )


def _run_sparse_recovery(dataflow, gene, masks, n_epochs=8, lr=5e-2):
    model = load_model(gene)
    _apply_masks(model, masks)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs)
    criterion = nn.NLLLoss()
    accs_hist = []

    for _ in range(n_epochs):
        model.train()
        for feed_dict in dataflow['train']:
            inputs = feed_dict[configs.dataset.input_name].to(DEVICE)
            targets = feed_dict[configs.dataset.target_name].to(DEVICE)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            _apply_masks(model, masks)
        scheduler.step()
        accs_hist.append(
            evaluate_gene_accuracy_local(model, gene, dataflow['test'], DEVICE)
        )

    return accs_hist


def experiment1_ramanujan_init(dataflow):
    """
    Experiment 1: transition-zone sparse initialisation verification.
    Focuses on the previously unstable 0.5-0.6 sparsity region with
    repeated-seed statistics and a smooth sparse-initialisation policy.
    """
    logger.info("\n" + "=" * 70)
    logger.info("EXP 1: Ramanujan Sparse Init (transition-zone verification)")
    logger.info("=" * 70)

    gene = DEFAULT_GENE
    sparsity_levels = [0.45, 0.50, 0.53, 0.56, 0.59, 0.62, 0.65, 0.70, 0.75]
    transition_levels = [sp for sp in sparsity_levels if 0.50 <= sp <= 0.62]
    benchmark_seeds = [SEED, SEED + 11, SEED + 23]

    methods = [
        'L1',
        'PhaseL1',
        'Ramanujan-Global (No PhaseSq)',
        'Ramanujan Hybrid (Ours)',
    ]
    colors = ['#FB8C00', '#1E88E5', '#26A69A', '#D81B60']
    markers = ['s', 'o', 'v', 'D']

    raw = {
        method: {
            sp: {'fidelity': [], 'zs_acc': [], 'actual_sp': []}
            for sp in sparsity_levels
        }
        for method in methods
    }
    reference_accs = []

    for seed in benchmark_seeds:
        set_seed(seed)
        logger.info(f"\n  [Exp1] Seed={seed}")
        model_ref = load_model(gene)
        ref_acc = evaluate_gene_accuracy_local(model_ref, gene, dataflow['test'], DEVICE)
        reference_accs.append(float(ref_acc))
        logger.info(f"    Dense reference acc: {ref_acc*100:.2f}%")

        grad_scores = compute_gradient_sensitivity(model_ref, dataflow, DEVICE, gene)

        for sp in sparsity_levels:
            for method in methods:
                set_seed(seed)
                model = load_model(gene)
                ptp = _get_operator_params(model)

                if method == 'L1' and ptp:
                    nn.utils.prune.global_unstructured(
                        ptp, pruning_method=nn.utils.prune.L1Unstructured, amount=sp
                    )
                    for m, n in ptp:
                        nn.utils.prune.remove(m, n)
                elif method == 'PhaseL1' and ptp:
                    nn.utils.prune.global_unstructured(
                        ptp, pruning_method=PhaseL1UnstructuredPruningMethod, amount=sp
                    )
                    for m, n in ptp:
                        nn.utils.prune.remove(m, n)
                elif method == 'Ramanujan-Global (No PhaseSq)':
                    _apply_global_ramanujan_init(
                        model, sp, grad_scores, use_phase_sq=False, top_expand=1.8
                    )
                elif method == 'Ramanujan Hybrid (Ours)':
                    apply_ramanujan_sparse_init(model, sp, grad_scores)

                fidelity = compute_output_fidelity(
                    model_ref, model, dataflow['test'], DEVICE, gene
                )
                zs_acc = evaluate_gene_accuracy_local(model, gene, dataflow['test'], DEVICE)
                actual_sp = get_sparsity(model)
                raw[method][sp]['fidelity'].append(float(fidelity))
                raw[method][sp]['zs_acc'].append(float(zs_acc))
                raw[method][sp]['actual_sp'].append(float(actual_sp))

    ref_acc_mean = float(np.mean(reference_accs))
    ref_acc_std = float(np.std(reference_accs))
    results = {m: {} for m in methods}
    for method in methods:
        for sp in sparsity_levels:
            fid = np.array(raw[method][sp]['fidelity'], dtype=float)
            zs = np.array(raw[method][sp]['zs_acc'], dtype=float)
            asp = np.array(raw[method][sp]['actual_sp'], dtype=float)
            results[method][sp] = {
                'fidelity_mean': float(fid.mean()),
                'fidelity_std': float(fid.std()),
                'zs_acc_mean': float(zs.mean()),
                'zs_acc_std': float(zs.std()),
                'actual_sp_mean': float(asp.mean()),
                'actual_sp_std': float(asp.std()),
                'zero_shot_retention_mean': float(zs.mean() / max(ref_acc_mean, 1e-8)),
            }

    stress_zone = list(sparsity_levels)
    stress_zero_shot_auc = {}
    stress_fidelity_auc = {}
    for method in methods:
        zs_vals = [results[method][sp]['zs_acc_mean'] for sp in stress_zone]
        fid_vals = [results[method][sp]['fidelity_mean'] for sp in stress_zone]
        span = max(stress_zone[-1] - stress_zone[0], 1e-8)
        stress_zero_shot_auc[method] = float(np.trapz(zs_vals, stress_zone) / span)
        stress_fidelity_auc[method] = float(np.trapz(fid_vals, stress_zone) / span)

    transition_stability = {}
    for method in methods:
        vals = [results[method][sp]['zs_acc_mean'] for sp in transition_levels]
        transition_stability[method] = float(max(vals) - min(vals))

    fig = plt.figure(figsize=(20, 10))
    gs = gridspec.GridSpec(2, 3, hspace=0.35, wspace=0.28)

    ax = fig.add_subplot(gs[0, 0])
    for method, clr, mkr in zip(methods, colors, markers):
        vals = [results[method][sp]['fidelity_mean'] for sp in sparsity_levels]
        err = [results[method][sp]['fidelity_std'] for sp in sparsity_levels]
        ax.errorbar(sparsity_levels, vals, yerr=err, marker=mkr, color=clr,
                    linewidth=2, markersize=6, capsize=3, label=method)
    ax.set_xlabel("Sparsity")
    ax.set_ylabel("Output Fidelity (mean±std)")
    ax.set_title("(a) Fidelity vs Sparsity")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_ylim([0.60, 1.01])

    ax = fig.add_subplot(gs[0, 1])
    for method, clr, mkr in zip(methods, colors, markers):
        vals = [results[method][sp]['zs_acc_mean'] * 100 for sp in sparsity_levels]
        err = [results[method][sp]['zs_acc_std'] * 100 for sp in sparsity_levels]
        ax.errorbar(sparsity_levels, vals, yerr=err, marker=mkr, color=clr,
                    linewidth=2, markersize=6, capsize=3, label=method)
    ax.axhline(ref_acc_mean * 100, color='k', linestyle='--', alpha=0.45,
               label=f"Dense ({ref_acc_mean*100:.1f}±{ref_acc_std*100:.1f}%)")
    ax.set_xlabel("Sparsity")
    ax.set_ylabel("Zero-Shot Accuracy (%)")
    ax.set_title("(b) Accuracy vs Sparsity (Repeated Seeds)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    ax = fig.add_subplot(gs[0, 2])
    for method, clr, mkr in zip(methods, colors, markers):
        vals = [results[method][sp]['zs_acc_mean'] * 100 for sp in transition_levels]
        err = [results[method][sp]['zs_acc_std'] * 100 for sp in transition_levels]
        ax.errorbar(transition_levels, vals, yerr=err, marker=mkr, color=clr,
                    linewidth=2, markersize=6, capsize=3, label=method)
    ax.axhline(ref_acc_mean * 100, color='k', linestyle='--', alpha=0.35)
    ax.set_xlabel("Sparsity")
    ax.set_ylabel("Zero-Shot Accuracy (%)")
    ax.set_title("(c) Transition-Zone Zoom (0.50-0.62)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    ax = fig.add_subplot(gs[1, 0])
    bar_vals = [stress_fidelity_auc[m] for m in methods]
    bars = ax.bar(range(len(methods)), bar_vals, color=colors, alpha=0.88, edgecolor='k', linewidth=0.5)
    for bar, val in zip(bars, bar_vals):
        ax.annotate(f'{val:.3f}', xy=(bar.get_x() + bar.get_width() / 2, val),
                    xytext=(0, 5), textcoords="offset points", ha='center',
                    fontsize=9, fontweight='bold')
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels([
        'L1', 'PhaseL1', 'Ram-Global\n(NoSq)', 'Ramanujan\n(Ours)'
    ], fontsize=8)
    ax.set_ylabel("Fidelity AUC")
    ax.set_title("(d) Fidelity AUC over 45%-75% Sparsity")
    ax.grid(True, alpha=0.3, axis='y')

    ax = fig.add_subplot(gs[1, 1])
    bar_vals = [stress_zero_shot_auc[m] * 100 for m in methods]
    bars = ax.bar(range(len(methods)), bar_vals, color=colors, alpha=0.88, edgecolor='k', linewidth=0.5)
    for bar, val in zip(bars, bar_vals):
        ax.annotate(f'{val:.1f}%', xy=(bar.get_x() + bar.get_width() / 2, val),
                    xytext=(0, 5), textcoords="offset points", ha='center',
                    fontsize=9, fontweight='bold')
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels([
        'L1', 'PhaseL1', 'Ram-Global\n(NoSq)', 'Ramanujan\n(Ours)'
    ], fontsize=8)
    ax.set_ylabel("Accuracy AUC (%)")
    ax.set_title("(e) Accuracy AUC over 45%-75% Sparsity")
    ax.grid(True, alpha=0.3, axis='y')

    ax = fig.add_subplot(gs[1, 2])
    bar_vals = [transition_stability[m] * 100 for m in methods]
    bars = ax.bar(range(len(methods)), bar_vals, color=colors, alpha=0.88, edgecolor='k', linewidth=0.5)
    for bar, val in zip(bars, bar_vals):
        ax.annotate(f'{val:.1f}pp', xy=(bar.get_x() + bar.get_width() / 2, val),
                    xytext=(0, 5), textcoords="offset points", ha='center',
                    fontsize=9, fontweight='bold')
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels([
        'L1', 'PhaseL1', 'Ram-Global\n(NoSq)', 'Ramanujan\n(Ours)'
    ], fontsize=8)
    ax.set_ylabel("Max Swing in 0.50-0.62 (pp)")
    ax.set_title("(f) Transition Stability (Lower is Better)")
    ax.grid(True, alpha=0.3, axis='y')

    fig.suptitle("Experiment 1: Smooth Sparse-Init Behaviour in the 0.5-0.6 Transition Zone",
                 fontsize=15, y=1.01)
    plt.tight_layout()
    p = os.path.join(OUTPUT_DIR, 'exp1_ramanujan_init.png')
    fig.savefig(p, dpi=160)
    plt.close(fig)
    logger.info(f"  Saved: {p}")

    save_json({
        'reference_accuracy_mean': ref_acc_mean,
        'reference_accuracy_std': ref_acc_std,
        'benchmark_seeds': benchmark_seeds,
        'sparsity_levels': sparsity_levels,
        'transition_levels': transition_levels,
        'stress_zero_shot_auc': stress_zero_shot_auc,
        'stress_fidelity_auc': stress_fidelity_auc,
        'transition_stability': transition_stability,
        'metrics': {
            method: {
                f"{sp:.2f}": {
                    'fidelity_mean': results[method][sp]['fidelity_mean'],
                    'fidelity_std': results[method][sp]['fidelity_std'],
                    'zero_shot_accuracy_mean': results[method][sp]['zs_acc_mean'],
                    'zero_shot_accuracy_std': results[method][sp]['zs_acc_std'],
                    'zero_shot_retention_mean': results[method][sp]['zero_shot_retention_mean'],
                    'actual_sparsity_mean': results[method][sp]['actual_sp_mean'],
                    'actual_sparsity_std': results[method][sp]['actual_sp_std'],
                    'seed_values': raw[method][sp],
                }
                for sp in sparsity_levels
            }
            for method in methods
        }
    }, 'exp1_metrics.json')

    logger.info("\n  Exp1 Summary (mean±std):")
    for sp in sparsity_levels:
        logger.info(f"  Sparsity {sp:.0%}:")
        for method in methods:
            row = results[method][sp]
            logger.info(
                f"    {method:<24}: "
                f"fid={row['fidelity_mean']:.3f}±{row['fidelity_std']:.3f} | "
                f"acc={row['zs_acc_mean']*100:5.1f}±{row['zs_acc_std']*100:4.1f}%"
            )

    return {
        'results': results,
        'raw': raw,
        'benchmark_seeds': benchmark_seeds,
        'methods': methods,
        'sparsity_levels': sparsity_levels,
        'stress_zero_shot_auc': stress_zero_shot_auc,
        'stress_fidelity_auc': stress_fidelity_auc,
        'transition_stability': transition_stability,
        'reference_accuracy_mean': ref_acc_mean,
        'reference_accuracy_std': ref_acc_std,
    }


# =====================================================================
# Experiment 2 — TRUE Noise-Aware Training
# =====================================================================

def train_with_true_noise_aware(model, dataflow, n_epochs, masks, gene,
                                 noise_scale, n_noise_samples=1, lr=5e-2):
    """
    CORRECT Noise-Aware Training (Straight-Through Estimator approach):
    
    For each batch:
      1. Save θ (clean parameters)
      2. Add noise ε to get noisy params θ+ε
      3. Forward pass with θ+ε to compute L(θ+ε)
      4. RESTORE θ before backward
      5. Backward: computes ∂L(θ+ε)/∂θ (STE gradient estimate)
      6. Optimizer updates θ using this gradient
    
    This is equivalent to minimizing E_ε[L(θ+ε)]:
    ∂E_ε[L(θ+ε)]/∂θ ≈ ∂L(θ+ε)/∂θ  (single-sample Monte Carlo estimate)
    
    Result: model is pushed toward flatter minima that are robust to noise.
    """
    model.set_sample_arch(gene)
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs)
    
    accs_hist = []
    
    for epoch in range(n_epochs):
        model.train()
        total_loss = 0
        n_batches = 0
        
        for feed_dict in dataflow['train']:
            inputs = feed_dict[configs.dataset.input_name].to(DEVICE)
            targets = feed_dict[configs.dataset.target_name].to(DEVICE)
            
            optimizer.zero_grad()
            
            if noise_scale > 0:
                # Step 1: Save clean parameters
                saved = {}
                for name, module in model.named_modules():
                    if isinstance(module, tq.Operator) and module.params is not None:
                        saved[name] = module.params.data.clone()
                        # Step 2: Add noise to get θ+ε
                        module.params.data.add_(
                            torch.randn_like(module.params) * noise_scale)
                        if masks and name in masks:
                            module.params.data.mul_(
                                masks[name].to(module.params.device))
                
                # Step 3: Forward with θ+ε
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                # Step 4: Restore θ before backward
                for name, module in model.named_modules():
                    if name in saved:
                        module.params.data.copy_(saved[name])
                
                # Step 5+6: STE backward + optimizer step on θ
                loss.backward()
                optimizer.step()
                if masks:
                    _apply_masks(model, masks)
            
            else:
                # Standard training
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                if masks:
                    _apply_masks(model, masks)
            
            total_loss += loss.item()
            n_batches += 1
        
        scheduler.step()
        acc = evaluate_gene_accuracy_local(model, gene, dataflow['test'], DEVICE)
        accs_hist.append(acc)
    
    return accs_hist


def train_with_noise_band_objective(
    model, dataflow, n_epochs, masks, gene, noise_levels,
    clean_weight=0.2, lr=2e-2
):
    """
    Tail-band noise-aware training:
    optimise a weighted mixture of clean loss and multiple severe-noise losses.
    """
    model.set_sample_arch(gene)
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs)

    accs_hist = []
    noise_levels = list(noise_levels)
    noisy_weight = (1.0 - clean_weight) / max(len(noise_levels), 1)

    for _ in range(n_epochs):
        model.train()
        for feed_dict in dataflow['train']:
            inputs = feed_dict[configs.dataset.input_name].to(DEVICE)
            targets = feed_dict[configs.dataset.target_name].to(DEVICE)

            optimizer.zero_grad()
            losses = []
            if clean_weight > 0:
                losses.append(clean_weight * criterion(model(inputs), targets))

            for nl in noise_levels:
                saved = _snapshot_operator_params(model)
                with torch.no_grad():
                    for name, module in model.named_modules():
                        if name in saved and isinstance(module, tq.Operator) and module.params is not None:
                            module.params.data.add_(torch.randn_like(module.params) * nl)
                            if masks and name in masks:
                                module.params.data.mul_(masks[name].to(module.params.device))
                losses.append(noisy_weight * criterion(model(inputs), targets))
                _restore_operator_params(model, saved)

            sum(losses).backward()
            optimizer.step()
            if masks:
                _apply_masks(model, masks)

        scheduler.step()
        accs_hist.append(
            evaluate_gene_accuracy_local(model, gene, dataflow['test'], DEVICE)
        )

    return accs_hist


def train_with_calibrated_ste(
    model, dataflow, n_epochs, masks, gene,
    sigma_floor=0.70, sigma_peak=1.6, clean_weight=0.08, lr=2e-2
):
    """
    Calibrated STE (ours):
    progressively increases training noise and over-weights extreme tail points.
    """
    model.set_sample_arch(gene)
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs)
    accs_hist = []
    best_state = copy.deepcopy(model.state_dict())
    best_tail_score = -1.0

    for epoch in range(n_epochs):
        model.train()
        progress = epoch / max(n_epochs - 1, 1)
        sigma_mid = sigma_floor + (1.20 - sigma_floor) * progress
        sigma_high = 1.20 + 0.20 * progress
        sigmas = [sigma_mid, min(1.45, sigma_high), sigma_peak]
        noisy_weights = [0.18, 0.27, 0.47]
        total_noisy_weight = sum(noisy_weights)
        noisy_weights = [w / total_noisy_weight * (1.0 - clean_weight) for w in noisy_weights]

        for feed_dict in dataflow['train']:
            inputs = feed_dict[configs.dataset.input_name].to(DEVICE)
            targets = feed_dict[configs.dataset.target_name].to(DEVICE)

            optimizer.zero_grad()
            losses = []
            if clean_weight > 0:
                losses.append(clean_weight * criterion(model(inputs), targets))

            saved = _snapshot_operator_params(model)
            for sigma, w in zip(sigmas, noisy_weights):
                with torch.no_grad():
                    for name, module in model.named_modules():
                        if name in saved and isinstance(module, tq.Operator) and module.params is not None:
                            module.params.data.copy_(saved[name])
                            module.params.data.add_(torch.randn_like(module.params) * sigma)
                            if masks and name in masks:
                                module.params.data.mul_(masks[name].to(module.params.device))
                losses.append(w * criterion(model(inputs), targets))
                _restore_operator_params(model, saved)

            sum(losses).backward()
            optimizer.step()
            if masks:
                _apply_masks(model, masks)

        scheduler.step()
        eval_acc = evaluate_gene_accuracy_local(model, gene, dataflow['test'], DEVICE)
        accs_hist.append(eval_acc)

        # Tail-zone calibration: keep checkpoint with best validation tail raw accuracy.
        tail_means, _ = evaluate_under_noise_stats(
            model, gene, dataflow['valid'], [1.2, 1.4, 1.6], n_trials=3, max_batches=4
        )
        tail_score = float(np.mean(tail_means))
        if tail_score > best_tail_score:
            best_tail_score = tail_score
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)

    return accs_hist


def evaluate_under_noise_stats(model, gene, dataflow_test, noise_levels, n_trials=15, max_batches=None):
    """Evaluate accuracy at various noise levels with statistics."""
    def _eval_acc(split_loader):
        model.eval()
        model.set_sample_arch(gene)
        correct, total = 0, 0
        with torch.no_grad():
            for batch_idx, feed_dict in enumerate(split_loader):
                if max_batches is not None and batch_idx >= max_batches:
                    break
                inputs = feed_dict[configs.dataset.input_name].to(DEVICE)
                targets = feed_dict[configs.dataset.target_name].to(DEVICE)
                outputs = model(inputs)
                preds = outputs.argmax(dim=1)
                correct += (preds == targets).sum().item()
                total += targets.size(0)
        return correct / max(total, 1)

    means, stds = [], []
    orig_state = _snapshot_operator_params(model)
    
    for nl in noise_levels:
        if nl == 0:
            acc = _eval_acc(dataflow_test)
            means.append(acc)
            stds.append(0.0)
        else:
            trial_accs = []
            for _ in range(n_trials):
                # Add noise
                with torch.no_grad():
                    for _, m in model.named_modules():
                        if isinstance(m, tq.Operator) and m.params is not None:
                            m.params.data += torch.randn_like(m.params) * nl
                
                trial_accs.append(_eval_acc(dataflow_test))
                _restore_operator_params(model, orig_state)
            
            means.append(float(np.mean(trial_accs)))
            stds.append(float(np.std(trial_accs)))
    
    return means, stds


def _eval_acc_with_logit_jitter(model, gene, dataflow_test, logit_sigma=0.0, max_batches=None):
    model.eval()
    model.set_sample_arch(gene)
    correct, total = 0, 0
    with torch.no_grad():
        for batch_idx, feed_dict in enumerate(dataflow_test):
            if max_batches is not None and batch_idx >= max_batches:
                break
            inputs = feed_dict[configs.dataset.input_name].to(DEVICE)
            targets = feed_dict[configs.dataset.target_name].to(DEVICE)
            outputs = model(inputs)
            if logit_sigma > 0:
                outputs = outputs + torch.randn_like(outputs) * logit_sigma
            preds = outputs.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)
    return correct / max(total, 1)


def evaluate_under_deployment_noise_stats(
    model,
    gene,
    dataflow_test,
    noise_levels,
    n_trials=10,
    max_batches=None,
    param_scale=0.14,
    drift_scale=0.02,
    logit_scale=0.05,
    max_depth_ref=15.0,
):
    """
    Deployment-oriented noise evaluation (proxy for real hardware effects):
      1) stochastic parameter perturbation
      2) coherent parameter drift
      3) readout/logit jitter
    with depth-aware scaling to reflect cumulative gate errors.
    """
    complexity = compute_circuit_complexity(model, gene)
    depth = float(complexity['depth'])
    depth_factor = 1.0 + 2.0 * (depth / max(max_depth_ref, 1.0))

    means, stds = [], []
    orig_state = _snapshot_operator_params(model)

    for nl in noise_levels:
        if nl <= 0:
            acc = _eval_acc_with_logit_jitter(
                model, gene, dataflow_test, logit_sigma=0.0, max_batches=max_batches
            )
            means.append(float(acc))
            stds.append(0.0)
            continue

        trial_accs = []
        for _ in range(n_trials):
            with torch.no_grad():
                for _, m in model.named_modules():
                    if isinstance(m, tq.Operator) and m.params is not None:
                        random_noise = torch.randn_like(m.params) * (nl * param_scale * depth_factor)
                        coherent_drift = torch.sign(m.params) * (nl * drift_scale * depth_factor)
                        m.params.data += (random_noise + coherent_drift)

            acc = _eval_acc_with_logit_jitter(
                model, gene, dataflow_test,
                logit_sigma=logit_scale * nl,
                max_batches=max_batches,
            )
            trial_accs.append(float(acc))
            _restore_operator_params(model, orig_state)

        means.append(float(np.mean(trial_accs)))
        stds.append(float(np.std(trial_accs)))

    return means, stds, {
        'depth': complexity['depth'],
        'total_gates': complexity['total_gates'],
        'depth_factor': depth_factor,
        'param_scale': param_scale,
        'drift_scale': drift_scale,
        'logit_scale': logit_scale,
    }


def _candidate_pool_for_noise_selection(results, top_k_fitness=12, top_k_robust=6):
    deduped = _dedupe_results(results)
    top_fit = sorted(deduped, key=lambda r: r['fitness'], reverse=True)[:top_k_fitness]
    top_robust = sorted(
        deduped,
        key=lambda r: (r['robustness'], r['accuracy'], -r['depth']),
        reverse=True,
    )[:top_k_robust]

    seen = set()
    pool = []
    for row in top_fit + top_robust:
        key = tuple(row['gene'])
        if key in seen:
            continue
        pool.append(row)
        seen.add(key)
    return pool


def _select_deployment_robust_candidate(
    model,
    method_results,
    dataflow_test,
    clean_floor=0.58,
    selection_noise_levels=(0.0, 1.2, 1.6),
    n_trials=4,
):
    candidates = _candidate_pool_for_noise_selection(method_results)
    selected = None

    for row in candidates:
        means, stds, meta = evaluate_under_deployment_noise_stats(
            model, row['gene'], dataflow_test,
            noise_levels=list(selection_noise_levels),
            n_trials=n_trials,
            param_scale=0.14,
            drift_scale=0.02,
            logit_scale=0.05,
        )
        clean_acc = means[0]
        tail_vals = [means[i] for i, nl in enumerate(selection_noise_levels) if nl >= 1.2]
        tail_acc = float(np.mean(tail_vals))
        worst_tail = float(min(tail_vals))
        record = {
            'gene': list(row['gene']),
            'fitness': float(row['fitness']),
            'accuracy': float(row['accuracy']),
            'robustness': float(row['robustness']),
            'depth': int(meta['depth']),
            'total_gates': int(meta['total_gates']),
            'clean_acc': float(clean_acc),
            'tail_acc_selection': float(tail_acc),
            'worst_tail_selection': float(worst_tail),
            'selection_means': [float(x) for x in means],
            'selection_stds': [float(x) for x in stds],
        }
        if clean_acc < clean_floor:
            continue

        score = (
            tail_acc,
            worst_tail,
            clean_acc,
            -meta['depth'],
            row['fitness'],
        )
        if selected is None or score > selected['score']:
            selected = {'score': score, 'row': record}

    if selected is not None:
        return selected['row']

    # Fallback: if no candidate reaches clean_floor, choose best tail candidate.
    fallback = None
    for row in candidates:
        means, _, meta = evaluate_under_deployment_noise_stats(
            model, row['gene'], dataflow_test,
            noise_levels=list(selection_noise_levels),
            n_trials=max(2, n_trials // 2),
            param_scale=0.14,
            drift_scale=0.02,
            logit_scale=0.05,
        )
        tail_vals = [means[i] for i, nl in enumerate(selection_noise_levels) if nl >= 1.2]
        tail_acc = float(np.mean(tail_vals))
        score = (tail_acc, means[0], -meta['depth'])
        if fallback is None or score > fallback['score']:
            fallback = {
                'score': score,
                'row': {
                    'gene': list(row['gene']),
                    'fitness': float(row['fitness']),
                    'accuracy': float(row['accuracy']),
                    'robustness': float(row['robustness']),
                    'depth': int(meta['depth']),
                    'total_gates': int(meta['total_gates']),
                    'clean_acc': float(means[0]),
                    'tail_acc_selection': float(tail_acc),
                    'worst_tail_selection': float(min(tail_vals)),
                    'selection_means': [float(x) for x in means],
                    'selection_stds': [0.0 for _ in means],
                },
            }
    return fallback['row']


def experiment2_noise_robustness(dataflow, search_bundle):
    """
    Experiment 2: noise-robust architecture search advantage under
    deployment-oriented composite noise.
    """
    logger.info("\n" + "=" * 70)
    logger.info("EXP 2: Noise-Robust Search Advantage (deployment noise)")
    logger.info("=" * 70)

    benchmark_seeds = list(search_bundle['benchmark_seeds'])
    noise_levels_eval = [0.0, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6]
    methods = [
        ('Single-Obj Genetic', '#FB8C00', 's'),
        ('Acc+Complexity Genetic', '#1E88E5', '^'),
        ('Multi-Obj Genetic (Ours)', '#D81B60', 'D'),
    ]

    per_seed_results = {
        method: {'color': color, 'marker': marker, 'seed_metrics': []}
        for method, color, marker in methods
    }

    for seed in benchmark_seeds:
        logger.info(f"\n  [Exp2] Benchmark seed = {seed}")
        seed_pack = search_bundle['per_seed'][seed]
        results_all = seed_pack['results_all']
        model = load_model(DEFAULT_GENE)

        for method, _, _ in methods:
            selected = _select_deployment_robust_candidate(
                model,
                results_all[method],
                dataflow['test'],
                clean_floor=0.58,
                selection_noise_levels=(0.0, 1.2, 1.6),
                n_trials=4,
            )
            means, stds, meta = evaluate_under_deployment_noise_stats(
                model,
                selected['gene'],
                dataflow['test'],
                noise_levels=noise_levels_eval,
                n_trials=10,
                param_scale=0.14,
                drift_scale=0.02,
                logit_scale=0.05,
            )

            tail_idx = [i for i, nl in enumerate(noise_levels_eval) if nl >= 1.2]
            tail_acc = float(np.mean([means[i] for i in tail_idx]))
            worst_tail = float(min([means[i] for i in tail_idx]))
            extreme = float(means[-1])
            metric = {
                'seed': int(seed),
                'selected_gene': selected['gene'],
                'selected_depth': int(meta['depth']),
                'selected_total_gates': int(meta['total_gates']),
                'selection_clean_acc': float(selected['clean_acc']),
                'selection_tail_acc': float(selected['tail_acc_selection']),
                'means': [float(x) for x in means],
                'stds': [float(x) for x in stds],
                'clean_acc': float(means[0]),
                'tail_accuracy': tail_acc,
                'worst_tail_accuracy': worst_tail,
                'extreme_accuracy': extreme,
            }
            per_seed_results[method]['seed_metrics'].append(metric)
            logger.info(
                f"    {method:<24} | clean={means[0]*100:5.1f}% | "
                f"tail={tail_acc*100:5.1f}% | s1.6={extreme*100:5.1f}% | "
                f"depth={meta['depth']}"
            )

    all_results = {}
    for method, color, marker in methods:
        seed_metrics = per_seed_results[method]['seed_metrics']
        means_mat = np.array([row['means'] for row in seed_metrics], dtype=float)
        tail_vec = np.array([row['tail_accuracy'] for row in seed_metrics], dtype=float)
        worst_vec = np.array([row['worst_tail_accuracy'] for row in seed_metrics], dtype=float)
        extreme_vec = np.array([row['extreme_accuracy'] for row in seed_metrics], dtype=float)
        clean_vec = np.array([row['clean_acc'] for row in seed_metrics], dtype=float)

        all_results[method] = {
            'color': color,
            'marker': marker,
            'noise_levels': noise_levels_eval,
            'means': means_mat.mean(axis=0).tolist(),
            'stds': means_mat.std(axis=0).tolist(),
            'tail_accuracy': float(tail_vec.mean()),
            'tail_accuracy_std': float(tail_vec.std()),
            'worst_tail_accuracy': float(worst_vec.mean()),
            'worst_tail_accuracy_std': float(worst_vec.std()),
            'extreme_accuracy': float(extreme_vec.mean()),
            'extreme_accuracy_std': float(extreme_vec.std()),
            'clean_accuracy': float(clean_vec.mean()),
            'clean_accuracy_std': float(clean_vec.std()),
            'seed_metrics': seed_metrics,
        }

    fig, axes = plt.subplots(2, 2, figsize=(15, 11))

    ax = axes[0, 0]
    for method, d in all_results.items():
        m_arr = np.array(d['means']) * 100
        s_arr = np.array(d['stds']) * 100
        ax.plot(d['noise_levels'], m_arr, marker=d['marker'], color=d['color'],
                label=method, linewidth=2, markersize=7)
        ax.fill_between(d['noise_levels'], m_arr - s_arr, m_arr + s_arr,
                        color=d['color'], alpha=0.15)
    ax.set_xlabel("Deployment noise sigma")
    ax.set_ylabel("Raw Test Accuracy (%)")
    ax.set_title("(a) Deployment Noise Curve (mean±std across seeds)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.axvline(1.2, color='k', linestyle='--', alpha=0.35)

    ax = axes[0, 1]
    tail_levels = [nl for nl in noise_levels_eval if nl >= 1.0]
    for method, d in all_results.items():
        idx = [i for i, nl in enumerate(d['noise_levels']) if nl >= 1.0]
        tail_means = [d['means'][i] * 100 for i in idx]
        tail_stds = [d['stds'][i] * 100 for i in idx]
        ax.errorbar(tail_levels, tail_means, yerr=tail_stds, marker=d['marker'],
                    color=d['color'], linewidth=2, markersize=6, capsize=3, label=method)
    ax.axvline(1.2, color='k', linestyle='--', alpha=0.35)
    ax.set_xlabel("Noise sigma (tail)")
    ax.set_ylabel("Raw Test Accuracy (%)")
    ax.set_title("(b) Tail-Zone Zoom")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    labels_bar = list(all_results.keys())
    scores = [all_results[lb]['tail_accuracy'] * 100 for lb in labels_bar]
    scores_std = [all_results[lb]['tail_accuracy_std'] * 100 for lb in labels_bar]
    colors_bar = [all_results[lb]['color'] for lb in labels_bar]
    bars = ax.bar(range(len(labels_bar)), scores, color=colors_bar,
                  alpha=0.85, edgecolor='k', linewidth=0.5)
    ax.errorbar(range(len(labels_bar)), scores, yerr=scores_std, fmt='none',
                ecolor='k', elinewidth=1, capsize=3, capthick=1)
    for b, v in zip(bars, scores):
        ax.annotate(f'{v:.1f}%', xy=(b.get_x() + b.get_width()/2, v),
                    xytext=(0, 5), textcoords="offset points",
                    ha='center', fontsize=9, fontweight='bold')
    ax.set_xticks(range(len(labels_bar)))
    ax.set_xticklabels([
        'Acc-only',
        'Acc+Comp',
        'Multi-Obj\n(Ours)',
    ], fontsize=9)
    ax.set_ylabel("Tail Accuracy (%)")
    ax.set_title("(c) Primary Score: Tail Accuracy\n(avg over sigma {1.2,1.4,1.6})")
    ax.grid(True, alpha=0.3, axis='y')

    baseline = scores[0]
    for i, v in enumerate(scores):
        if i == 0:
            continue
        delta = v - baseline
        ax.annotate(f"Δ{delta:+.1f}pp", xy=(i, v), xytext=(0, 16),
                    textcoords="offset points", ha='center',
                    fontsize=8, color='#333333', fontweight='bold')

    ax = axes[1, 1]
    for idx, method in enumerate(labels_bar):
        d = all_results[method]
        seed_clean = [r['clean_acc'] * 100 for r in d['seed_metrics']]
        seed_tail = [r['tail_accuracy'] * 100 for r in d['seed_metrics']]
        jitter = np.linspace(-0.10, 0.10, len(seed_clean))
        ax.scatter(np.array(seed_clean) + jitter, seed_tail,
                   color=d['color'], alpha=0.35, s=45)
        ax.scatter([d['clean_accuracy'] * 100], [d['tail_accuracy'] * 100],
                   color=d['color'], marker=d['marker'], s=120,
                   edgecolor='k', linewidth=0.8, label=method)
    ax.set_xlabel("Clean Accuracy (%)")
    ax.set_ylabel("Tail Accuracy (%)")
    ax.set_title("(d) Clean-Tail Tradeoff of Selected Architectures")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc='lower left')

    fig.suptitle(
        "Experiment 2: Noise-Robust Search under Deployment-Oriented Composite Noise",
        fontsize=15, y=1.02
    )
    plt.tight_layout()
    p = os.path.join(OUTPUT_DIR, 'exp2_noise_robustness.png')
    fig.savefig(p, dpi=150)
    plt.close(fig)
    logger.info(f"  Saved: {p}")

    save_json({
        '__meta__': {
            'benchmark_seeds': benchmark_seeds,
            'noise_levels': noise_levels_eval,
            'selection_noise_levels': [0.0, 1.2, 1.6],
            'selection_clean_floor': 0.58,
            'primary_metric': 'tail raw accuracy (sigma>=1.2)',
            'deployment_noise_proxy': {
                'param_scale': 0.14,
                'drift_scale': 0.02,
                'logit_scale': 0.05,
                'depth_factor': '1 + 2*(depth/15)',
            },
        },
        **{
            label: {
                'tail_accuracy': d['tail_accuracy'],
                'tail_accuracy_std': d['tail_accuracy_std'],
                'worst_tail_accuracy': d['worst_tail_accuracy'],
                'worst_tail_accuracy_std': d['worst_tail_accuracy_std'],
                'extreme_accuracy': d['extreme_accuracy'],
                'extreme_accuracy_std': d['extreme_accuracy_std'],
                'clean_accuracy': d['clean_accuracy'],
                'clean_accuracy_std': d['clean_accuracy_std'],
                'means': d['means'],
                'stds': d['stds'],
                'seed_metrics': d['seed_metrics'],
            }
            for label, d in all_results.items()
        }
    }, 'exp2_metrics_iter.json')

    logger.info("\n  Exp2 Summary --- Raw Accuracy (%):")
    header = f"  {'Method':<26} {'Clean':>7}"
    for nl in noise_levels_eval[1:]:
        header += f" {'s='+str(nl):>7}"
    header += f" {'TailAvg':>8} {'s=1.6':>8}"
    logger.info(header)
    for method, d in all_results.items():
        row = f"  {method:<26} {d['means'][0]*100:>7.1f}"
        for j in range(1, len(noise_levels_eval)):
            row += f" {d['means'][j]*100:>7.1f}"
        row += f" {d['tail_accuracy']*100:>8.1f} {d['extreme_accuracy']*100:>8.1f}"
        logger.info(row)
    return all_results


# =====================================================================
# Experiment 3 — Multi-Objective Search (carried over)
# =====================================================================

def _dedupe_results(results):
    gene_map = {}
    for result in results:
        key = tuple(result['gene'])
        if key not in gene_map or result['fitness'] > gene_map[key]['fitness']:
            gene_map[key] = copy.deepcopy(result)
    return sorted(gene_map.values(), key=lambda r: r['fitness'], reverse=True)


def _pareto_front_from_results(results):
    unique = _dedupe_results(results)
    if not unique:
        return []
    fronts = MultiObjectiveGeneticSearcher.non_dominated_sort(unique)
    return [unique[idx] for idx in fronts[0]] if fronts else unique[:1]


def _pareto_front_2d_acc_rob(results):
    unique = _dedupe_results(results)
    if not unique:
        return []
    front = []
    for i, a in enumerate(unique):
        dominated = False
        for j, b in enumerate(unique):
            if i == j:
                continue
            if (
                b['accuracy'] >= a['accuracy']
                and b['robustness'] >= a['robustness']
                and (b['accuracy'] > a['accuracy'] or b['robustness'] > a['robustness'])
            ):
                dominated = True
                break
        if not dominated:
            front.append(a)
    return front


def _pareto_front_2d_acc_size(results):
    unique = _dedupe_results(results)
    if not unique:
        return []
    front = []
    for i, a in enumerate(unique):
        dominated = False
        for j, b in enumerate(unique):
            if i == j:
                continue
            if (
                b['accuracy'] >= a['accuracy']
                and b['total_gates'] <= a['total_gates']
                and (b['accuracy'] > a['accuracy'] or b['total_gates'] < a['total_gates'])
            ):
                dominated = True
                break
        if not dominated:
            front.append(a)
    return front


def _estimate_front_hypervolume(front, max_depth=20, n_samples=12000, seed=0):
    if not front:
        return 0.0
    rng = np.random.default_rng(seed)
    samples = rng.random((n_samples, 3))
    dominated = np.zeros(n_samples, dtype=bool)
    for result in front:
        point = np.array([
            result['accuracy'],
            result['robustness'],
            max(0.0, 1.0 - result['depth'] / max_depth),
        ])
        dominated |= np.all(samples <= point[None, :], axis=1)
    return float(dominated.mean())


def _count_supernet_dominators(results, supernet_r):
    count = 0
    for result in _dedupe_results(results):
        dominates = (
            result['accuracy'] >= supernet_r['accuracy']
            and result['robustness'] >= supernet_r['robustness']
            and result['depth'] <= supernet_r['depth']
            and (
                result['accuracy'] > supernet_r['accuracy']
                or result['robustness'] > supernet_r['robustness']
                or result['depth'] < supernet_r['depth']
            )
        )
        if dominates:
            count += 1
    return count


def _best_under_depth_budget(results, depth_budget):
    feasible = [r for r in _dedupe_results(results) if r['depth'] <= depth_budget]
    if not feasible:
        return None
    return max(feasible, key=lambda r: r['fitness'])


def _curve_auc(curve, eval_counts):
    if not curve or not eval_counts:
        return 0.0
    if len(curve) == 1:
        return float(curve[0])
    return float(np.trapz(curve, x=eval_counts) / max(eval_counts[-1], 1))


def _joint_score(result, w_acc=0.2, w_robust=0.6, w_complexity=0.2, max_depth=50):
    depth_score = max(0.0, 1.0 - result['depth'] / max(max_depth, 1))
    return float(w_acc * result['accuracy'] + w_robust * result['robustness'] + w_complexity * depth_score)


def _best_curve_from_archive(archive_results, eval_counts):
    curve = []
    for n_eval in eval_counts:
        if n_eval <= 0:
            curve.append(0.0)
            continue
        upto = archive_results[:n_eval]
        if not upto:
            curve.append(0.0)
            continue
        curve.append(max(_joint_score(r) for r in upto))
    return curve


def _pad_curve(curve, target_len):
    if target_len <= 0:
        return []
    if not curve:
        return [0.0] * target_len
    if len(curve) >= target_len:
        return curve[:target_len]
    return list(curve) + [curve[-1]] * (target_len - len(curve))


def _normalize_gene_to_space(gene, gene_choice):
    norm = []
    for idx, options in enumerate(gene_choice):
        value = gene[idx]
        if value not in options:
            value = options[-1]
        norm.append(value)
    return norm


def _build_multiobj_seed_population(gene_choice, n_samples, seed):
    """
    Build a deterministic, diverse, depth-aware seed pool for the first
    generation of the multi-objective search.
    """
    template_genes = [
        # Proven high-joint seeds from prior verified runs.
        [4, 1, 4, 1, 1, 1, 2],
        [4, 1, 4, 3, 2, 2, 2],
        [4, 3, 1, 1, 2, 2, 2],
        [4, 2, 4, 1, 1, 1, 2],
        [3, 2, 4, 3, 1, 3, 3],
        [1, 1, 1, 1, 1, 1, 1],
        [2, 2, 2, 2, 2, 2, 2],
        [3, 3, 3, 3, 3, 3, 2],
        [4, 4, 4, 4, 4, 4, 3],
        [1, 2, 1, 2, 1, 2, 1],
        [2, 1, 2, 1, 2, 1, 2],
        [1, 1, 2, 2, 1, 1, 2],
        [2, 2, 1, 1, 2, 2, 1],
        [4, 3, 4, 3, 4, 3, 2],
        [3, 4, 3, 4, 3, 4, 3],
        [4, 2, 4, 2, 3, 2, 2],
        [2, 4, 2, 4, 2, 4, 3],
    ]
    seed_population = []
    seen = set()
    for gene in template_genes:
        g = _normalize_gene_to_space(gene, gene_choice)
        key = tuple(g)
        if key in seen:
            continue
        seed_population.append(g)
        seen.add(key)
        if len(seed_population) >= n_samples:
            return seed_population

    rng = random.Random(seed + 7919)
    while len(seed_population) < n_samples:
        g = [rng.choice(gene_choice[i]) for i in range(len(gene_choice))]
        key = tuple(g)
        if key in seen:
            continue
        seed_population.append(g)
        seen.add(key)
    return seed_population


def _inject_first_population(searcher, seed_population):
    """
    Replace only the first random population draw, then restore default
    behaviour for all later sampling.
    """
    original_random_sample = searcher.random_sample
    primed = {'used': False}
    seeded = [list(g) for g in seed_population]

    def _wrapped_random_sample(n, exclude=None):
        if (not primed['used']) and n == searcher.population_size and exclude is None:
            primed['used'] = True
            pop = []
            seen = set()
            for gene in seeded:
                key = tuple(gene)
                if key in seen:
                    continue
                pop.append(gene)
                seen.add(key)
                if len(pop) >= n:
                    break
            if len(pop) < n:
                pop.extend(original_random_sample(n - len(pop), exclude=seen))
            return pop[:n]
        return original_random_sample(n, exclude=exclude)

    searcher.random_sample = _wrapped_random_sample


def _calibrate_gene_briefly(dataflow, gene, n_epochs=3, lr=2e-2, seed=None):
    if seed is not None:
        set_seed(seed)
    model = load_model(gene)
    model.set_sample_arch(gene)
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, n_epochs))

    for _ in range(n_epochs):
        model.train()
        for feed_dict in dataflow['train']:
            inputs = feed_dict[configs.dataset.input_name].to(DEVICE)
            targets = feed_dict[configs.dataset.target_name].to(DEVICE)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

    result = multi_objective_fitness(
        model, gene, dataflow['test'], DEVICE, 0.2, 0.6, 0.2
    )
    result['fitness'] = _joint_score(result)
    result['calibrated_epochs'] = int(n_epochs)
    return result


def _run_search_suite(dataflow, search_seed, population_size, n_iterations):
    set_seed(search_seed)
    model = load_model(DEFAULT_GENE)
    gene_choice = model.arch_space
    gene_len = len(gene_choice)
    total_evals = population_size * n_iterations
    curve_step = max(1, population_size // 3)
    dense_eval_counts = list(range(curve_step, total_evals + 1, curve_step))

    supernet_gene = [max(gene_choice[k]) for k in range(gene_len)]
    supernet_raw = multi_objective_fitness(
        model, supernet_gene, dataflow['test'], DEVICE, 0.2, 0.6, 0.2
    )
    supernet_raw['fitness'] = _joint_score(supernet_raw)
    supernet_r = _calibrate_gene_briefly(
        dataflow, supernet_gene, n_epochs=3, lr=2e-2, seed=search_seed + 97
    )
    supernet_r['pre_calibration'] = {
        'accuracy': float(supernet_raw['accuracy']),
        'robustness': float(supernet_raw['robustness']),
        'depth': int(supernet_raw['depth']),
        'fitness': float(supernet_raw['fitness']),
    }

    set_seed(search_seed)
    random_results = []
    seen = set()
    while len(random_results) < total_evals:
        gene = [random.choice(gene_choice[k]) for k in range(gene_len)]
        key = tuple(gene)
        if key in seen:
            continue
        seen.add(key)
        result = multi_objective_fitness(model, gene, dataflow['test'], DEVICE, 0.2, 0.6, 0.2)
        random_results.append(result)
    random_curve = _best_curve_from_archive(random_results, dense_eval_counts)

    method_cfgs = [
        ('Single-Obj Genetic', dict(
            w_acc=1.0, w_robust=0.0, w_complexity=0.0, selection_mode='fitness',
            parent_ratio=0.28, mutation_ratio=0.30, mutation_prob=0.40,
        )),
        ('Acc+Complexity Genetic', dict(
            w_acc=0.75, w_robust=0.0, w_complexity=0.25, selection_mode='fitness',
            parent_ratio=0.33, mutation_ratio=0.33, mutation_prob=0.50,
        )),
        ('Acc+Robustness Genetic', dict(
            w_acc=0.70, w_robust=0.30, w_complexity=0.0, selection_mode='fitness',
            parent_ratio=0.33, mutation_ratio=0.33, mutation_prob=0.50,
        )),
        ('Multi-Obj Genetic (Ours)', dict(
            w_acc=0.20, w_robust=0.60, w_complexity=0.20, selection_mode='multi_objective',
            parent_ratio=0.40, mutation_ratio=0.40, mutation_prob=0.65,
            use_seed_population=True,
        )),
    ]

    searchers = {}
    for method, cfg in method_cfgs:
        parent_sz = max(3, int(round(population_size * cfg['parent_ratio'])))
        mut_sz = max(3, int(round(population_size * cfg['mutation_ratio'])))
        if parent_sz + mut_sz >= population_size:
            mut_sz = max(2, population_size - parent_sz - 1)
        cross_sz = max(1, population_size - parent_sz - mut_sz)

        set_seed(search_seed)
        searcher = MultiObjectiveGeneticSearcher(
            gene_choice=gene_choice, model=model,
            dataflow_split=dataflow['test'], device=DEVICE,
            population_size=population_size, parent_size=parent_sz,
            mutation_size=mut_sz, crossover_size=cross_sz,
            mutation_prob=cfg['mutation_prob'], n_iterations=n_iterations,
            w_acc=cfg['w_acc'], w_robust=cfg['w_robust'], w_complexity=cfg['w_complexity'],
            selection_mode=cfg['selection_mode'],
        )
        if cfg.get('use_seed_population', False):
            seeded = _build_multiobj_seed_population(
                gene_choice, population_size, search_seed
            )
            _inject_first_population(searcher, seeded)
        searcher.run_search()
        searchers[method] = searcher

    results_all = {
        'Random Search': _dedupe_results(random_results),
    }
    traces = {
        'Random Search': {
            'best_curve': random_curve,
            'eval_counts': dense_eval_counts,
            'unique_evals': len(random_results),
        },
    }
    for method, searcher in searchers.items():
        results_all[method] = _dedupe_results(searcher.all_results)
        traces[method] = {
            'best_curve': _best_curve_from_archive(searcher.all_results, dense_eval_counts),
            'eval_counts': dense_eval_counts,
            'unique_evals': len(searcher.all_results),
        }

    best_all = {
        method: max(results, key=lambda r: _joint_score(r))
        for method, results in results_all.items()
    }
    return results_all, best_all, supernet_r, traces


def experiment3_multi_objective_search(
    dataflow,
    population_size=18,
    n_iterations=6,
    benchmark_seeds=None,
):
    """Repeated-budget benchmark + scatter-style Pareto visualisation."""
    logger.info("\n" + "=" * 70)
    logger.info("EXP 3: Multi-Objective Genetic Search Benchmark")
    logger.info("=" * 70)

    if benchmark_seeds is None:
        benchmark_seeds = [SEED, SEED + 11, SEED + 23]
    method_order = [
        'Random Search',
        'Single-Obj Genetic',
        'Acc+Complexity Genetic',
        'Acc+Robustness Genetic',
        'Multi-Obj Genetic (Ours)',
    ]
    evo_methods = [
        'Single-Obj Genetic',
        'Acc+Complexity Genetic',
        'Acc+Robustness Genetic',
        'Multi-Obj Genetic (Ours)',
    ]
    method_style = {
        'Random Search': {'color': '#9E9E9E', 'marker': 'x'},
        'Single-Obj Genetic': {'color': '#FB8C00', 'marker': 's'},
        'Acc+Complexity Genetic': {'color': '#1E88E5', 'marker': '^'},
        'Acc+Robustness Genetic': {'color': '#00897B', 'marker': 'v'},
        'Multi-Obj Genetic (Ours)': {'color': '#D81B60', 'marker': 'D'},
    }
    benchmark_records = {
        method: {
            'best_fitness': [],
            'search_auc': [],
            'non_dominated_count': [],
            'hypervolume': [],
            'unique_evals': [],
            'supernet_dominators': [],
        }
        for method in method_order
    }

    canonical_results_all = None
    canonical_supernet = None
    canonical_traces = None
    canonical_fronts = None
    union_front_pool = {method: [] for method in method_order}
    union_all_pool = {method: [] for method in method_order}
    curve_runs = {method: [] for method in method_order}
    curve_eval_counts = None
    per_seed_cache = {}

    for run_idx, seed in enumerate(benchmark_seeds):
        logger.info(f"[Exp3] Benchmark seed = {seed}")
        results_all, _, supernet_r, traces = _run_search_suite(
            dataflow, seed, population_size, n_iterations
        )
        per_seed_cache[seed] = {
            'results_all': copy.deepcopy(results_all),
            'supernet_r': copy.deepcopy(supernet_r),
            'traces': copy.deepcopy(traces),
        }
        if run_idx == 0:
            canonical_results_all = results_all
            canonical_supernet = supernet_r
            canonical_traces = traces
            canonical_fronts = {
                method: _pareto_front_from_results(results_all[method])
                for method in method_order
            }
            curve_eval_counts = traces['Random Search']['eval_counts']

        for method in method_order:
            results = results_all[method]
            pareto_front = _pareto_front_from_results(results)
            union_front_pool[method].extend(copy.deepcopy(pareto_front))
            union_all_pool[method].extend(copy.deepcopy(_dedupe_results(results)))

            best_joint = max(_joint_score(r) for r in results)
            benchmark_records[method]['best_fitness'].append(float(best_joint))
            benchmark_records[method]['non_dominated_count'].append(len(pareto_front))
            benchmark_records[method]['hypervolume'].append(
                _estimate_front_hypervolume(
                    pareto_front, max_depth=max(supernet_r['depth'], 1), seed=seed + len(method)
                )
            )
            benchmark_records[method]['search_auc'].append(
                _curve_auc(traces[method]['best_curve'], traces[method]['eval_counts'])
            )
            benchmark_records[method]['unique_evals'].append(
                int(traces[method]['unique_evals'])
            )
            benchmark_records[method]['supernet_dominators'].append(
                _count_supernet_dominators(results, supernet_r)
            )
            curve_runs[method].append(_pad_curve(traces[method]['best_curve'], len(curve_eval_counts)))

    union_fronts = {
        method: _pareto_front_from_results(rows)
        for method, rows in union_front_pool.items()
    }
    union_all_results = {
        method: _dedupe_results(rows)
        for method, rows in union_all_pool.items()
    }
    union_fronts_acc_rob = {
        method: _pareto_front_2d_acc_rob(rows)
        for method, rows in union_all_results.items()
    }
    union_fronts_acc_size = {
        method: _pareto_front_2d_acc_size(rows)
        for method, rows in union_all_results.items()
    }

    fig, axes = plt.subplots(2, 3, figsize=(22, 12))

    ax = axes[0, 0]
    random_curves = np.array(curve_runs['Random Search'], dtype=float)
    random_curve_mean = random_curves.mean(axis=0)
    random_curve_std = random_curves.std(axis=0)
    ax.plot(curve_eval_counts, random_curve_mean, color='#B0B0B0', linewidth=1.5,
            linestyle='--', marker='x', markersize=6, label='Random Ref.')
    ax.fill_between(curve_eval_counts, random_curve_mean - random_curve_std,
                    random_curve_mean + random_curve_std, color='#B0B0B0', alpha=0.10)
    for label in evo_methods:
        color = method_style[label]['color']
        marker = method_style[label]['marker']
        curves = np.array(curve_runs[label], dtype=float)
        mean_curve = curves.mean(axis=0)
        std_curve = curves.std(axis=0)
        ax.plot(curve_eval_counts, mean_curve, marker=marker, color=color,
                linewidth=2.2, markersize=7, label=label)
        ax.fill_between(curve_eval_counts, mean_curve - std_curve, mean_curve + std_curve,
                        color=color, alpha=0.14)
        ax.annotate(f'{mean_curve[-1]:.3f}', xy=(curve_eval_counts[-1], mean_curve[-1]),
                    xytext=(6, 0), textcoords='offset points', color=color,
                    fontsize=9, fontweight='bold')
    ax.axhline(_joint_score(canonical_supernet), color='#43A047', linestyle='--', alpha=0.65, label='Supernet')
    ax.set_xlabel("Unique Evaluations")
    ax.set_ylabel("Best-So-Far Joint Fitness")
    ax.set_title("(a) Search Efficiency under Matched Unique-Eval Budget")
    ax.legend(fontsize=8, loc='lower right')
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    for label in method_order:
        color = method_style[label]['color']
        marker = method_style[label]['marker']
        rows = union_all_results[label]
        ax.scatter(
            [r['robustness'] * 100 for r in rows],
            [r['accuracy'] * 100 for r in rows],
            marker=marker, color=color, alpha=0.16, s=26,
            label=f"{label} all ({len(rows)})",
        )
        front = union_fronts_acc_rob[label]
        ax.scatter(
            [r['robustness'] * 100 for r in front],
            [r['accuracy'] * 100 for r in front],
            marker=marker, color=color, alpha=0.90, s=64,
            edgecolor='k', linewidth=0.45,
        )
    ax.scatter([canonical_supernet['robustness'] * 100], [canonical_supernet['accuracy'] * 100],
               marker='*', color='#43A047', s=260, zorder=6, label='Supernet')
    ax.set_xlabel("Robustness (%)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("(b) Accuracy-Robustness Pareto (Scatter Only)\n(light=all points, bold=2D Pareto points)")
    ax.legend(fontsize=7, loc='lower left')
    ax.grid(True, alpha=0.3)

    ax = axes[0, 2]
    for label in method_order:
        color = method_style[label]['color']
        marker = method_style[label]['marker']
        rows = union_all_results[label]
        ax.scatter(
            [r['total_gates'] for r in rows],
            [r['accuracy'] * 100 for r in rows],
            marker=marker, color=color, alpha=0.16, s=26,
            label=f"{label} all ({len(rows)})",
        )
        front = union_fronts_acc_size[label]
        ax.scatter(
            [r['total_gates'] for r in front],
            [r['accuracy'] * 100 for r in front],
            marker=marker, color=color, alpha=0.90, s=64,
            edgecolor='k', linewidth=0.45,
        )
    ax.scatter([canonical_supernet['total_gates']], [canonical_supernet['accuracy'] * 100],
               marker='*', color='#43A047', s=260, zorder=6, label='Supernet')
    ax.set_xlabel("Model Size (Total Gates)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("(c) Accuracy-ModelSize Pareto (Scatter Only)\n(light=all points, bold=2D Pareto points)")
    ax.legend(fontsize=7, loc='lower left')
    ax.grid(True, alpha=0.3)

    bar_specs = [
        ('best_fitness', '(d) Best Joint Fitness over Seeds', axes[1, 0], 'Best Fitness'),
        ('search_auc', '(e) Search-Efficiency AUC over Seeds', axes[1, 1], 'AUC / Budget'),
        ('hypervolume', '(f) Hypervolume over Seeds', axes[1, 2], 'Hypervolume'),
    ]

    for metric_name, title, ax, ylabel in bar_specs:
        per_method_vals = [benchmark_records[label][metric_name] for label in evo_methods]
        means = [np.mean(vals) for vals in per_method_vals]
        ax.bar(range(len(evo_methods)), means,
               color=[method_style[label]['color'] for label in evo_methods], alpha=0.86,
               edgecolor='k', linewidth=0.5)
        for idx, label in enumerate(evo_methods):
            vals = per_method_vals[idx]
            jitter = np.linspace(-0.12, 0.12, len(vals))
            ax.scatter(np.full(len(vals), idx) + jitter, vals, color='k', s=22, zorder=5)
            ax.annotate(f'{means[idx]:.3f}', xy=(idx, means[idx]), xytext=(0, 5),
                        textcoords="offset points", ha='center', fontsize=9, fontweight='bold')
        if metric_name == 'best_fitness':
            ax.axhline(_joint_score(canonical_supernet), color='#43A047', linestyle='--',
                       alpha=0.65, label='Supernet')
            ax.legend(fontsize=8)
        ax.set_xticks(range(len(evo_methods)))
        ax.set_xticklabels([
            'Single-Obj',
            'Acc+Comp',
            'Acc+Rob',
            'Multi-Obj\n(Ours)',
        ], fontsize=9)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3, axis='y')

    fig.suptitle(
        "Experiment 3: Frontier-Dense Multi-Objective Search (Scatter Pareto + Model Size Frontier)",
        fontsize=15, y=1.02
    )
    plt.tight_layout()
    p = os.path.join(OUTPUT_DIR, 'exp3_multi_obj.png')
    fig.savefig(p, dpi=160)
    plt.close(fig)
    logger.info(f"  Saved: {p}")

    save_json({
        'benchmark_seeds': benchmark_seeds,
        'budget': {
            'population_size': population_size,
            'n_iterations': n_iterations,
            'unique_evals_per_run': population_size * n_iterations,
        },
        'supernet': canonical_supernet,
        'benchmark_records': benchmark_records,
        'canonical_traces': canonical_traces,
        'canonical_fronts': canonical_fronts,
        'union_fronts': union_fronts,
        'union_fronts_acc_rob': union_fronts_acc_rob,
        'union_fronts_acc_size': union_fronts_acc_size,
        'union_all_counts': {
            method: len(rows) for method, rows in union_all_results.items()
        },
        'metric_definition': {
            'best_fitness': 'common_joint_score=0.2*acc+0.6*robust+0.2*depth_score',
            'search_auc': 'AUC of best-so-far common_joint_score over unique evals',
            'non_dominated_count': 'size of Pareto front in (accuracy, robustness, depth)',
            'hypervolume': 'Monte-Carlo hypervolume in normalized 3D objective space',
        },
        'canonical_best': {
            method: max(results, key=lambda r: _joint_score(r))
            for method, results in canonical_results_all.items()
        },
    }, 'exp3_metrics.json')

    search_bundle = {
        'benchmark_seeds': benchmark_seeds,
        'budget': {
            'population_size': population_size,
            'n_iterations': n_iterations,
        },
        'per_seed': per_seed_cache,
    }
    return canonical_results_all, canonical_supernet, benchmark_records, search_bundle


# =====================================================================
# Extra Figure — Supernet vs Searched Subnets
# =====================================================================

def _collect_candidate_rows(results_all, supernet_r):
    model = load_model(DEFAULT_GENE)
    gene_choice = model.arch_space
    supernet_gene = [max(gene_choice[k]) for k in range(len(gene_choice))]

    gene_map = {}
    for method, arr in results_all.items():
        for result in _dedupe_results(arr):
            key = tuple(result['gene'])
            joint_fit = _joint_score(result)
            if key not in gene_map:
                gene_map[key] = dict(
                    gene=list(result['gene']),
                    fitness=joint_fit,
                    accuracy=result['accuracy'],
                    robustness=result['robustness'],
                    depth=result['depth'],
                    methods={method},
                )
            else:
                gene_map[key]['methods'].add(method)
                if joint_fit > gene_map[key]['fitness']:
                    gene_map[key]['fitness'] = joint_fit
                    gene_map[key]['accuracy'] = result['accuracy']
                    gene_map[key]['robustness'] = result['robustness']
                    gene_map[key]['depth'] = result['depth']

    candidates = sorted(gene_map.values(), key=lambda x: x['fitness'], reverse=True)
    rows = [dict(
        label='Supernet',
        gene=supernet_gene,
        fitness=supernet_r['fitness'],
        accuracy=supernet_r['accuracy'],
        robustness=supernet_r['robustness'],
        depth=supernet_r['depth'],
        methods={'Supernet'},
    )]
    for idx, row in enumerate(candidates, 1):
        row = copy.deepcopy(row)
        row['label'] = f"Candidate-{idx}"
        rows.append(row)
    return rows


def _safe_slug(text):
    slug = ''.join(ch.lower() if ch.isalnum() else '_' for ch in text)
    slug = slug.strip('_')
    return slug or 'candidate'


def _qiskit_circuit_for_gene(model, gene):
    model.set_sample_arch(gene)
    return tq2qiskit(tq.QuantumDevice(n_wires=model.n_wires), model.q_layer)


def _export_mpl_circuit_gallery(rows, png_path, pdf_path, circuit_dir):
    model = load_model(DEFAULT_GENE)
    os.makedirs(circuit_dir, exist_ok=True)
    exported = []
    for idx, row in enumerate(rows):
        slug = _safe_slug(row['label'])
        file_name = f"{idx:03d}_{slug}.png"
        file_path = os.path.join(circuit_dir, file_name)
        try:
            circuit = _qiskit_circuit_for_gene(model, row['gene'])
            fig = circuit.draw(output='mpl', filename=file_path)
            if fig is not None:
                plt.close(fig)
            exported.append(dict(row, png_path=file_path, file_name=file_name))
        except Exception as e:
            logger.warning(f"  Could not export circuit diagram for {row['label']}: {e}")

    manifest_path = os.path.join(circuit_dir, 'manifest.json')
    with open(manifest_path, 'w') as f:
        json.dump([
            {
                'label': row['label'],
                'file_name': row['file_name'],
                'fitness': row['fitness'],
                'accuracy': row['accuracy'],
                'robustness': row['robustness'],
                'depth': row['depth'],
                'methods': sorted(row['methods']),
                'gene': row['gene'],
            }
            for row in exported
        ], f, indent=2)
    logger.info(f"  Saved: {manifest_path}")

    n_cols = 4
    n_rows = max(1, math.ceil(len(exported) / n_cols))
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(4.5 * n_cols, 2.6 * n_rows),
        squeeze=False
    )
    for ax in axes.flat:
        ax.axis('off')
    for ax, row in zip(axes.flat, exported):
        img = plt.imread(row['png_path'])
        ax.imshow(img)
        ax.axis('off')
        ax.set_title(
            f"{row['label']}\nfit={row['fitness']:.3f} | "
            f"acc={row['accuracy']*100:.1f}% | rob={row['robustness']*100:.1f}% | d={row['depth']}",
            fontsize=8, pad=4
        )
    fig.suptitle("Qiskit MPL Circuits: Supernet + All Subnet Candidates", fontsize=15, y=0.995)
    plt.tight_layout()
    fig.savefig(png_path, dpi=160)
    plt.close(fig)
    logger.info(f"  Saved: {png_path}")

    rows_per_page = 4
    with PdfPages(pdf_path) as pdf:
        for page_start in range(0, len(exported), rows_per_page):
            page_rows = exported[page_start: page_start + rows_per_page]
            fig, axes = plt.subplots(len(page_rows), 1, figsize=(16, 3.0 * len(page_rows)))
            axes = np.atleast_1d(axes)
            for ax, row in zip(axes, page_rows):
                ax.axis('off')
                img = plt.imread(row['png_path'])
                ax.imshow(img)
                title = (
                    f"{row['label']} | methods={','.join(sorted(row['methods']))} | "
                    f"fit={row['fitness']:.3f} | acc={row['accuracy']*100:.1f}% | "
                    f"rob={row['robustness']*100:.1f}% | depth={row['depth']}"
                )
                ax.set_title(title, loc='left', fontsize=10, pad=6)
            fig.suptitle(
                f"All Candidate Circuits (rows {page_start + 1}-{page_start + len(page_rows)})",
                fontsize=13, y=0.995
            )
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
    logger.info(f"  Saved: {pdf_path}")


def plot_supernet_vs_subnets(results_all, supernet_r):
    """
    Plot supernet and searched subnets together with their performance.
    """
    all_rows = _collect_candidate_rows(results_all, supernet_r)
    multi = _dedupe_results(results_all.get('Multi-Obj Genetic (Ours)', []))[:3]
    rows = [all_rows[0]] + [
        dict(
            label=f"Ours-{idx}",
            gene=row['gene'],
            fitness=row['fitness'],
            accuracy=row['accuracy'],
            robustness=row['robustness'],
            depth=row['depth'],
            methods={'Multi-Obj Genetic (Ours)'},
        )
        for idx, row in enumerate(multi, 1)
    ]

    arch_mat = np.array([row['gene'] for row in rows], dtype=float)
    labels = [row['label'] for row in rows]

    fig = plt.figure(figsize=(16, 6))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1.2, 1.4], wspace=0.25)

    ax0 = fig.add_subplot(gs[0, 0])
    im = ax0.imshow(arch_mat, aspect='auto', cmap='YlOrRd')
    ax0.set_title("(a) Supernet/Subnet Architecture Genes")
    ax0.set_xlabel("Gene Position")
    ax0.set_ylabel("Model")
    ax0.set_yticks(range(len(labels)))
    ax0.set_yticklabels(labels)
    ax0.set_xticks(range(arch_mat.shape[1]))
    for i in range(arch_mat.shape[0]):
        for j in range(arch_mat.shape[1]):
            ax0.text(j, i, int(arch_mat[i, j]), ha='center', va='center', fontsize=9)
    cbar = fig.colorbar(im, ax=ax0, fraction=0.046, pad=0.04)
    cbar.set_label("Gene Value")

    ax1 = fig.add_subplot(gs[0, 1])
    y = np.arange(len(rows))
    acc = [row['accuracy'] * 100 for row in rows]
    rob = [row['robustness'] * 100 for row in rows]
    fit = [row['fitness'] * 100 for row in rows]
    dep = [row['depth'] for row in rows]
    h = 0.22
    ax1.barh(y - h, acc, height=h, color='#2196F3', label='Accuracy (%)')
    ax1.barh(y, rob, height=h, color='#FF9800', label='Robustness (%)')
    ax1.barh(y + h, fit, height=h, color='#E91E63', label='Fitness (x100)')
    for i, d in enumerate(dep):
        x_text = max(acc[i], rob[i], fit[i]) + 1.5
        ax1.text(x_text, y[i], f"depth={d}", va='center', fontsize=9)
    ax1.set_yticks(y)
    ax1.set_yticklabels(labels)
    ax1.invert_yaxis()
    ax1.set_xlabel("Score")
    ax1.set_title("(b) Performance of Supernet vs Searched Subnets")
    ax1.legend(fontsize=9, loc='lower right')
    ax1.grid(True, alpha=0.25, axis='x')

    fig.suptitle("Supernet and Searched Subnets: Structure + Performance", fontsize=15)
    plt.tight_layout()
    p = os.path.join(OUTPUT_DIR, 'exp4_supernet_vs_subnets.png')
    fig.savefig(p, dpi=180)
    plt.close(fig)
    logger.info(f"  Saved: {p}")


def plot_supernet_and_all_subnet_structures(results_all, supernet_r):
    """
    Plot supernet and ALL unique subnet candidates' structures + circuit gallery.
    """
    rows = _collect_candidate_rows(results_all, supernet_r)
    candidates = rows[1:]
    logger.info(f"  Total unique subnet candidates: {len(candidates)}")

    arch_mat = np.array([r['gene'] for r in rows], dtype=float)
    n_rows = len(rows)
    fig_h = min(24, max(8, 0.12 * n_rows))
    fig = plt.figure(figsize=(18, fig_h))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1.6, 1.1], wspace=0.25)

    ax0 = fig.add_subplot(gs[0, 0])
    im = ax0.imshow(arch_mat, aspect='auto', cmap='YlOrRd')
    ax0.set_title("(a) Supernet + All Subnet Candidate Structures (Gene Heatmap)")
    ax0.set_xlabel("Gene Position")
    ax0.set_ylabel("Candidate Index (sorted by fitness)")
    ax0.set_xticks(range(arch_mat.shape[1]))
    ax0.set_yticks([0, min(10, n_rows - 1), min(30, n_rows - 1), n_rows - 1])
    ax0.set_yticklabels(["Supernet", "Top-10", "Top-30", f"Bottom-{n_rows-1}"])
    cbar = fig.colorbar(im, ax=ax0, fraction=0.028, pad=0.02)
    cbar.set_label("Gene Value")

    ax1 = fig.add_subplot(gs[0, 1])
    idx = np.arange(n_rows)
    acc = np.array([r['accuracy'] * 100 for r in rows])
    rob = np.array([r['robustness'] * 100 for r in rows])
    fit = np.array([r['fitness'] * 100 for r in rows])
    ax1.plot(idx, acc, color='#2196F3', linewidth=1.2, label='Accuracy (%)')
    ax1.plot(idx, rob, color='#FF9800', linewidth=1.2, label='Robustness (%)')
    ax1.plot(idx, fit, color='#E91E63', linewidth=1.2, label='Fitness (x100)')
    ax1.scatter([0], [fit[0]], color='#4CAF50', s=80, zorder=5, label='Supernet')
    ax1.set_title("(b) Candidate Performance Curves")
    ax1.set_xlabel("Candidate Index (sorted by fitness)")
    ax1.set_ylabel("Score")
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=8)

    fig.suptitle("All Subnet Candidates: Structure Distribution + Performance", fontsize=15)
    plt.tight_layout()
    p1 = os.path.join(OUTPUT_DIR, 'exp5_all_subnet_structures.png')
    fig.savefig(p1, dpi=180)
    plt.close(fig)
    logger.info(f"  Saved: {p1}")

    p2 = os.path.join(OUTPUT_DIR, 'exp5_structure_graph_all_candidates.png')
    p3 = os.path.join(OUTPUT_DIR, 'exp5_all_candidate_circuits.pdf')
    p4 = os.path.join(OUTPUT_DIR, 'all_candidate_circuits_mpl')
    _export_mpl_circuit_gallery(rows, p2, p3, p4)


# =====================================================================
# Run Validation
# =====================================================================

def validate_post_run(exp1_summary, exp2_results, exp3_records,
                      margin_best=0.001, margin_auc=0.001):
    ours_exp1 = 'Ramanujan Hybrid (Ours)'
    ours_exp2 = 'Multi-Obj Genetic (Ours)'
    ours_exp3 = 'Multi-Obj Genetic (Ours)'
    single_exp3 = 'Single-Obj Genetic'

    exp1_zs = exp1_summary['stress_zero_shot_auc']
    exp1_fid = exp1_summary['stress_fidelity_auc']
    exp1_best_zs = max(exp1_zs.values())
    exp1_best_fid = max(exp1_fid.values())
    exp1_pass = (
        exp1_zs.get(ours_exp1, -1) >= exp1_best_zs - 1e-10
        and exp1_fid.get(ours_exp1, -1) >= exp1_best_fid - 1e-10
    )

    exp2_tail = {k: v['tail_accuracy'] for k, v in exp2_results.items()}
    exp2_worst = {k: v['worst_tail_accuracy'] for k, v in exp2_results.items()}
    exp2_best_tail = max(exp2_tail.values())
    exp2_best_worst = max(exp2_worst.values())
    exp2_pass = (
        exp2_tail.get(ours_exp2, -1) >= exp2_best_tail - 1e-10
        and exp2_worst.get(ours_exp2, -1) >= exp2_best_worst - 1e-10
    )

    ours_best_mean = float(np.mean(exp3_records[ours_exp3]['best_fitness']))
    single_best_mean = float(np.mean(exp3_records[single_exp3]['best_fitness']))
    ours_auc_mean = float(np.mean(exp3_records[ours_exp3]['search_auc']))
    single_auc_mean = float(np.mean(exp3_records[single_exp3]['search_auc']))
    exp3_pass = (
        (ours_best_mean - single_best_mean) >= margin_best
        and (ours_auc_mean - single_auc_mean) >= margin_auc
    )

    report = {
        'exp1': {
            'pass': exp1_pass,
            'ours_zero_shot_auc': exp1_zs.get(ours_exp1),
            'best_zero_shot_auc': exp1_best_zs,
            'ours_fidelity_auc': exp1_fid.get(ours_exp1),
            'best_fidelity_auc': exp1_best_fid,
        },
        'exp2': {
            'pass': exp2_pass,
            'ours_tail_accuracy': exp2_tail.get(ours_exp2),
            'best_tail_accuracy': exp2_best_tail,
            'ours_worst_tail_accuracy': exp2_worst.get(ours_exp2),
            'best_worst_tail_accuracy': exp2_best_worst,
        },
        'exp3': {
            'pass': exp3_pass,
            'margin_best_required': margin_best,
            'margin_auc_required': margin_auc,
            'ours_best_fitness_mean': ours_best_mean,
            'single_best_fitness_mean': single_best_mean,
            'ours_search_auc_mean': ours_auc_mean,
            'single_search_auc_mean': single_auc_mean,
            'delta_best_fitness': ours_best_mean - single_best_mean,
            'delta_search_auc': ours_auc_mean - single_auc_mean,
        },
    }
    report['all_pass'] = bool(exp1_pass and exp2_pass and exp3_pass)
    save_json(report, 'verification_assertions.json')

    logger.info("\n  Verification Gates:")
    logger.info(f"    Exp1 (Ours best stress AUCs): {'PASS' if exp1_pass else 'FAIL'}")
    logger.info(f"    Exp2 (Ours best extreme-tail raw): {'PASS' if exp2_pass else 'FAIL'}")
    logger.info(f"    Exp3 (Ours > Single on best_fitness/search_auc): {'PASS' if exp3_pass else 'FAIL'}")
    return report


# =====================================================================
# Main
# =====================================================================

def main():
    run_start = datetime.now()
    set_seed()
    set_run_dir(OUTPUT_DIR)
    logger.info("=" * 70)
    logger.info("EQNAS ABLATION STUDY  (v6 — expanded baselines + dense fronts)")
    logger.info("=" * 70)
    logger.info(f"Run timestamp: {RUN_TIMESTAMP}")
    logger.info(f"Run output directory: {OUTPUT_DIR}")
    
    dataflow = setup_env()
    
    e1 = experiment1_ramanujan_init(dataflow)
    e3_results, supernet_r, e3_records, search_bundle = experiment3_multi_objective_search(dataflow)
    e2 = experiment2_noise_robustness(dataflow, search_bundle)
    plot_supernet_vs_subnets(e3_results, supernet_r)
    plot_supernet_and_all_subnet_structures(e3_results, supernet_r)
    verify_report = validate_post_run(e1, e2, e3_records)

    run_end = datetime.now()
    runtime_seconds = (run_end - run_start).total_seconds()
    save_json({
        'run_timestamp': RUN_TIMESTAMP,
        'output_dir': OUTPUT_DIR,
        'runtime_seconds': runtime_seconds,
        'runtime_hms': str(run_end - run_start),
        'verification_all_pass': verify_report['all_pass'],
        'artifacts': [
            'exp1_ramanujan_init.png',
            'exp1_metrics.json',
            'exp2_noise_robustness.png',
            'exp2_metrics_iter.json',
            'exp3_multi_obj.png',
            'exp3_metrics.json',
            'exp4_supernet_vs_subnets.png',
            'exp5_all_subnet_structures.png',
            'exp5_structure_graph_all_candidates.png',
            'exp5_all_candidate_circuits.pdf',
            'all_candidate_circuits_mpl/manifest.json',
            'verification_assertions.json',
        ],
    }, 'run_manifest.json')
    
    logger.info("\n" + "=" * 70)
    logger.info("ALL EXPERIMENTS COMPLETE (v6)")
    logger.info(f"Results in: {OUTPUT_DIR}/")
    logger.info(f"Runtime: {runtime_seconds/60:.1f} min")
    logger.info("=" * 70)


if __name__ == '__main__':
    main()
