"""
EQNAS Ablation Study & Verification  (v2 — fixed)
====================================================
Three controlled experiments verify each innovation point.

  Experiment 1 (Lottery-Ticket Sparse Init):
    Maintain pruning mask during training so sparsity is persistent.
    Compare dense | random-mask+train | LT-mask+train at 50% sparsity.

  Experiment 2 (Noise-Aware Pruning):
    Fix noise injection to be temporary (save -> noise -> forward -> restore).
    All models pruned to same sparsity first (shared mask), then fine-tuned
    with / without noise injection.  Evaluate absolute accuracy under noise.

  Experiment 3 (Multi-Objective Genetic Search):
    Use parameter-perturbation robustness (sigma=0.1) so architectures truly
    differ in robustness.  Larger population and more iterations.
    Compare random search, single-obj genetic, multi-obj genetic (ours).
"""

import copy
import os
import sys
import random
from typing import Dict, List, Tuple

# Fix path shadowing: remove CWD entries whose torchquantum/ has no __init__.py
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

# Restore removed paths so local modules (eqnas) can be found
for _p in _removed_paths:
    if _p not in sys.path:
        sys.path.append(_p)

from torchpack.utils.config import configs
from torchpack.utils.logging import logger
from torchpack.environ import set_run_dir

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

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

# ============================================================================
# Global
# ============================================================================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CHECKPOINT_PATH = 'torchquantum/max-acc-valid.pt'
OUTPUT_DIR = 'eqnas_verification'
SEED = 42
DEFAULT_GENE = [4, 4, 4, 4, 4, 4, 3]

os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.rcParams.update({
    'font.size': 12, 'axes.labelsize': 13, 'axes.titlesize': 14,
    'legend.fontsize': 10, 'figure.dpi': 150,
    'savefig.bbox': 'tight', 'savefig.pad_inches': 0.1,
})


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
        sampler = torch.utils.data.RandomSampler(dataset[split])
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
    # Do NOT attach QiskitProcessor — forces perturbation-based robustness
    return model


# ============================================================================
# Helpers: mask-aware training
# ============================================================================

def _get_operator_params(model):
    return [(m, 'params') for _, m in model.named_modules()
            if isinstance(m, tq.Operator) and m.params is not None]


def _build_mask_dict(model):
    """Build {module_name: bool-like mask} from current zero pattern."""
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


def train_with_mask(model, dataflow, n_epochs, masks=None, lr=5e-2,
                    gene=None, noise_scale=0.0):
    """
    Training loop that re-applies masks after every optimizer.step(),
    maintaining sparsity.  Optional *temporary* noise injection.
    """
    if gene is not None:
        model.set_sample_arch(gene)
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs)

    epoch_losses, epoch_accs = [], []

    for epoch in range(n_epochs):
        model.train()
        total_loss, n_batches = 0, 0
        for feed_dict in dataflow['train']:
            inputs = feed_dict[configs.dataset.input_name].to(DEVICE)
            targets = feed_dict[configs.dataset.target_name].to(DEVICE)

            # ---- Temporary noise injection ----
            backups = {}
            if noise_scale > 0:
                for mname, module in model.named_modules():
                    if isinstance(module, tq.Operator) and module.params is not None:
                        backups[mname] = module.params.data.clone()
                        noise = torch.randn_like(module.params) * noise_scale
                        module.params.data += noise
                        if masks is not None and mname in masks:
                            module.params.data *= masks[mname].to(module.params.device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # ---- Restore clean params before backward ----
            if backups:
                for mname, module in model.named_modules():
                    if mname in backups:
                        module.params.data.copy_(backups[mname])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Re-apply mask so pruned params stay zero
            if masks is not None:
                _apply_masks(model, masks)

            total_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = total_loss / max(n_batches, 1)
        acc = evaluate_gene_accuracy(
            model, gene or DEFAULT_GENE, dataflow['test'], DEVICE)
        epoch_losses.append(avg_loss)
        epoch_accs.append(acc)

    return epoch_losses, epoch_accs


def evaluate_under_noise(model, gene, dataflow_test, noise_levels,
                         n_trials=5):
    """Absolute accuracy at each noise level (mean over trials)."""
    accs = []
    for nl in noise_levels:
        if nl == 0:
            accs.append(evaluate_gene_accuracy(
                model, gene, dataflow_test, DEVICE))
        else:
            trial_accs = []
            orig = copy.deepcopy(model.state_dict())
            for _ in range(n_trials):
                with torch.no_grad():
                    for _, m in model.named_modules():
                        if isinstance(m, tq.Operator) and m.params is not None:
                            m.params.data += torch.randn_like(m.params) * nl
                trial_accs.append(evaluate_gene_accuracy(
                    model, gene, dataflow_test, DEVICE))
                model.load_state_dict(orig)
            accs.append(float(np.mean(trial_accs)))
    return accs


# ============================================================================
# Experiment 1 — Lottery-Ticket Sparse Initialisation
# ============================================================================

def experiment1_lottery_ticket(dataflow):
    """
    A) Dense  (no prune, upper-bound baseline)
    B) Random-mask + train with mask maintained  (50 % sparsity)
    C) LT-mask + train with mask maintained      (50 % sparsity)
    """
    logger.info("\n" + "=" * 70)
    logger.info("EXP 1: Lottery-Ticket Sparse Initialisation  (v2)")
    logger.info("=" * 70)

    N_EPOCHS = 20
    SPARSITY = 0.5
    gene = DEFAULT_GENE
    results = {}

    # ----- A) Dense -----
    set_seed()
    logger.info("[Exp1-A] Dense baseline …")
    m_dense = load_model(gene)
    l_d, a_d = train_with_mask(m_dense, dataflow, N_EPOCHS, masks=None, gene=gene)
    results['Dense (no prune)'] = dict(losses=l_d, accs=a_d,
                                       sparsity=get_sparsity(m_dense))

    # ----- B) Random mask + masked training -----
    set_seed()
    logger.info("[Exp1-B] Random mask + masked training …")
    m_rand = load_model(gene)
    ptp = _get_operator_params(m_rand)
    if ptp:
        nn.utils.prune.global_unstructured(
            ptp, pruning_method=nn.utils.prune.RandomUnstructured,
            amount=SPARSITY)
        for m, n in ptp:
            nn.utils.prune.remove(m, n)
    masks_rand = _build_mask_dict(m_rand)
    # Reload original weights + apply mask (rewind to init, keep random mask)
    m_rand2 = load_model(gene)
    _apply_masks(m_rand2, masks_rand)
    l_r, a_r = train_with_mask(m_rand2, dataflow, N_EPOCHS,
                                masks=masks_rand, gene=gene)
    results['Random Mask'] = dict(losses=l_r, accs=a_r,
                                   sparsity=get_sparsity(m_rand2))

    # ----- C) Lottery-ticket mask + masked training -----
    set_seed()
    logger.info("[Exp1-C] LT mask + masked training …")
    m_lt = load_model(gene)
    criterion = nn.NLLLoss()
    m_lt, _ = lottery_ticket_sparse_init(
        model=m_lt, init_sparsity=SPARSITY, rewinding_epoch=3,
        dataflow=dataflow, criterion=criterion, optimizer_cls=optim.Adam,
        lr=configs.optimizer.lr, device=DEVICE, gene=gene,
    )
    lt_masks = _build_mask_dict(m_lt)
    l_lt, a_lt = train_with_mask(m_lt, dataflow, N_EPOCHS,
                                  masks=lt_masks, gene=gene)
    results['LT-Sparse Init (Ours)'] = dict(losses=l_lt, accs=a_lt,
                                              sparsity=get_sparsity(m_lt))

    # ----- Plot -----
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    clr = {'Dense (no prune)': '#2196F3', 'Random Mask': '#FF9800',
           'LT-Sparse Init (Ours)': '#E91E63'}
    mkr = {'Dense (no prune)': 'o', 'Random Mask': 's',
           'LT-Sparse Init (Ours)': 'D'}

    for nm, d in results.items():
        ep = list(range(1, N_EPOCHS + 1))
        sp_lbl = f" (sp={d['sparsity']:.0%})"
        axes[0].plot(ep, d['losses'], marker=mkr[nm], color=clr[nm],
                     label=nm + sp_lbl, markersize=4, linewidth=2)
        axes[1].plot(ep, [a * 100 for a in d['accs']], marker=mkr[nm],
                     color=clr[nm], label=nm + sp_lbl,
                     markersize=4, linewidth=2)

    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Training Loss")
    axes[0].set_title("(a) Training Loss"); axes[0].legend(); axes[0].grid(True, alpha=.3)
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Test Accuracy (%)")
    axes[1].set_title("(b) Test Accuracy"); axes[1].legend(); axes[1].grid(True, alpha=.3)

    fig.suptitle("Experiment 1: Lottery-Ticket Sparse Init", fontsize=15, y=1.02)
    plt.tight_layout()
    p = os.path.join(OUTPUT_DIR, 'exp1_lottery_ticket.png')
    fig.savefig(p, dpi=150); plt.close(fig)
    logger.info(f"  Saved: {p}")

    logger.info("\n  Summary:")
    logger.info(f"  {'Method':<25} {'Sparsity':>10} {'Final Acc':>12} {'Final Loss':>12}")
    for nm, d in results.items():
        logger.info(f"  {nm:<25} {d['sparsity']:>10.1%} "
                     f"{d['accs'][-1]:>12.4f} {d['losses'][-1]:>12.4f}")
    return results


# ============================================================================
# Experiment 2 — Noise-Aware Pruning
# ============================================================================

def experiment2_noise_aware_pruning(dataflow):
    """
    Shared mask (LT, 50 % sparsity) applied to all models.
    Then fine-tune with mask maintenance — vary only noise_scale.
    Evaluate absolute accuracy under a sweep of noise levels.
    """
    logger.info("\n" + "=" * 70)
    logger.info("EXP 2: Noise-Aware Pruning  (v2)")
    logger.info("=" * 70)

    N_EPOCHS = 20
    SPARSITY = 0.5
    gene = DEFAULT_GENE
    noise_scales_train = [0.0, 0.03, 0.08]
    noise_levels_eval = [0.0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3]

    # Build shared mask
    set_seed()
    m_mask = load_model(gene)
    crit = nn.NLLLoss()
    m_mask, _ = lottery_ticket_sparse_init(
        model=m_mask, init_sparsity=SPARSITY, rewinding_epoch=3,
        dataflow=dataflow, criterion=crit, optimizer_cls=optim.Adam,
        lr=configs.optimizer.lr, device=DEVICE, gene=gene,
    )
    shared_mask = _build_mask_dict(m_mask)
    logger.info(f"  Shared mask sparsity: {get_sparsity(m_mask):.1%}")

    results = {}
    for ns in noise_scales_train:
        set_seed()
        label = f"Noise-Aware (σ={ns})" if ns > 0 else "Standard (σ=0)"
        logger.info(f"\n[Exp2] Fine-tune with noise_scale={ns} …")

        model = load_model(gene)
        _apply_masks(model, shared_mask)

        losses, accs = train_with_mask(
            model, dataflow, N_EPOCHS,
            masks=shared_mask, gene=gene, noise_scale=ns)

        logger.info(f"  clean acc={accs[-1]:.4f}, sparsity={get_sparsity(model):.1%}")

        accs_at_noise = evaluate_under_noise(
            model, gene, dataflow['test'], noise_levels_eval, n_trials=5)
        results[label] = dict(
            noise_levels=noise_levels_eval, accs=accs_at_noise,
            train_accs=accs, sparsity=get_sparsity(model))

    # ----- Plot -----
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    colors = ['#2196F3', '#E91E63', '#4CAF50']
    markers = ['o', 'D', 's']

    for idx, (label, d) in enumerate(results.items()):
        axes[0].plot(d['noise_levels'], [a * 100 for a in d['accs']],
                     marker=markers[idx], color=colors[idx], label=label,
                     markersize=6, linewidth=2)
    axes[0].set_xlabel("Evaluation Noise σ"); axes[0].set_ylabel("Test Accuracy (%)")
    axes[0].set_title("(a) Accuracy vs Noise Level"); axes[0].legend()
    axes[0].grid(True, alpha=.3)

    # Bar: clean vs high-noise
    bar_labels = list(results.keys())
    clean = [results[l]['accs'][0] * 100 for l in bar_labels]
    noisy = [results[l]['accs'][-1] * 100 for l in bar_labels]
    x = np.arange(len(bar_labels)); w = 0.3
    axes[1].bar(x - w / 2, clean, w, label='Clean (σ=0)',
                color='#4CAF50', alpha=.85)
    axes[1].bar(x + w / 2, noisy, w,
                label=f'Noisy (σ={noise_levels_eval[-1]})',
                color='#F44336', alpha=.85)
    for i, (c, n) in enumerate(zip(clean, noisy)):
        axes[1].annotate(f'{c:.1f}%', xy=(i - w / 2, c), xytext=(0, 4),
                         textcoords="offset points", ha='center', fontsize=9,
                         color='green')
        axes[1].annotate(f'{n:.1f}%', xy=(i + w / 2, n), xytext=(0, 4),
                         textcoords="offset points", ha='center', fontsize=9,
                         color='red')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(['Std\n(σ=0)',
                             f'NA\n(σ={noise_scales_train[1]})',
                             f'NA\n(σ={noise_scales_train[2]})'], fontsize=10)
    axes[1].set_ylabel("Test Accuracy (%)")
    axes[1].set_title(f"(b) Clean vs Noisy (σ={noise_levels_eval[-1]})")
    axes[1].legend(); axes[1].grid(True, alpha=.3, axis='y')

    fig.suptitle("Experiment 2: Noise-Aware Pruning", fontsize=15, y=1.02)
    plt.tight_layout()
    p = os.path.join(OUTPUT_DIR, 'exp2_noise_aware_pruning.png')
    fig.savefig(p, dpi=150); plt.close(fig)
    logger.info(f"  Saved: {p}")

    logger.info("\n  Summary:")
    logger.info(f"  {'Method':<25} {'Clean':>10} {'σ=0.1':>10} {'σ=0.3':>10}")
    for label, d in results.items():
        nls = d['noise_levels']; av = d['accs']
        i01 = nls.index(0.1) if 0.1 in nls else -1
        logger.info(f"  {label:<25} {av[0]:>10.4f} {av[i01]:>10.4f} {av[-1]:>10.4f}")
    return results


# ============================================================================
# Experiment 3 — Multi-Objective Genetic Search
# ============================================================================

def experiment3_multi_objective_search(dataflow):
    """
    Use parameter-perturbation robustness (sigma=0.1) so different
    architectures have genuinely different robustness.  Larger budget.
    """
    logger.info("\n" + "=" * 70)
    logger.info("EXP 3: Multi-Objective Genetic Search  (v2)")
    logger.info("=" * 70)

    model = load_model(DEFAULT_GENE)
    N_ITERS = 6
    POP = 20
    gene_choice = model.arch_space
    gene_len = len(gene_choice)
    total_evals = POP * N_ITERS

    # --- A) Random Search ---
    set_seed()
    logger.info("[Exp3-A] Random Search …")
    random_results = []
    for _ in tqdm.tqdm(range(total_evals), desc="Random Search"):
        g = [random.choice(gene_choice[k]) for k in range(gene_len)]
        r = multi_objective_fitness(model, g, dataflow['test'], DEVICE,
                                    w_acc=0.5, w_robust=0.3, w_complexity=0.2)
        random_results.append(r)
    results_all = {'Random Search': random_results}

    # --- B) Single-objective genetic (accuracy only) ---
    set_seed()
    logger.info("[Exp3-B] Single-obj Genetic (accuracy only) …")
    s_single = MultiObjectiveGeneticSearcher(
        gene_choice=gene_choice, model=model,
        dataflow_split=dataflow['test'], device=DEVICE,
        population_size=POP,
        parent_size=max(POP // 4, 2),
        mutation_size=max(POP // 3, 2),
        crossover_size=max(POP - POP // 4 - POP // 3, 2),
        mutation_prob=0.5, n_iterations=N_ITERS,
        w_acc=1.0, w_robust=0.0, w_complexity=0.0,
    )
    _, pareto_s = s_single.run_search()
    # Re-evaluate pareto with multi-obj weights for fair fitness comparison
    single_results = []
    if pareto_s:
        for r in pareto_s:
            r2 = multi_objective_fitness(model, r['gene'], dataflow['test'],
                                         DEVICE, 0.5, 0.3, 0.2)
            single_results.append(r2)
    # Fill remaining budget with random genes evaluated under multi-obj
    set_seed(SEED + 100)
    while len(single_results) < total_evals:
        g = [random.choice(gene_choice[k]) for k in range(gene_len)]
        r = multi_objective_fitness(model, g, dataflow['test'], DEVICE,
                                    0.5, 0.3, 0.2)
        single_results.append(r)
    results_all['Single-Obj Genetic'] = single_results[:total_evals]

    # --- C) Multi-objective genetic (ours) ---
    set_seed()
    logger.info("[Exp3-C] Multi-obj Genetic Search (ours) …")
    s_multi = MultiObjectiveGeneticSearcher(
        gene_choice=gene_choice, model=model,
        dataflow_split=dataflow['test'], device=DEVICE,
        population_size=POP,
        parent_size=max(POP // 4, 2),
        mutation_size=max(POP // 3, 2),
        crossover_size=max(POP - POP // 4 - POP // 3, 2),
        mutation_prob=0.5, n_iterations=N_ITERS,
        w_acc=0.5, w_robust=0.3, w_complexity=0.2,
    )
    best_m, pareto_m = s_multi.run_search()
    multi_results = list(pareto_m) if pareto_m else [best_m]
    # Fill with extra random evaluations for scatter comparison
    seen = {str(r['gene']) for r in multi_results}
    set_seed(SEED + 200)
    while len(multi_results) < total_evals:
        g = [random.choice(gene_choice[k]) for k in range(gene_len)]
        if str(g) not in seen:
            r = multi_objective_fitness(model, g, dataflow['test'], DEVICE,
                                        0.5, 0.3, 0.2)
            multi_results.append(r)
            seen.add(str(g))
    results_all['Multi-Obj Genetic (Ours)'] = multi_results[:total_evals]

    # ----- Plot -----
    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.30)

    plot_cfgs = [
        ('Random Search', '#9E9E9E', 'x', 0.5),
        ('Single-Obj Genetic', '#FF9800', 's', 0.7),
        ('Multi-Obj Genetic (Ours)', '#E91E63', 'D', 0.9),
    ]

    # (a) Accuracy vs Depth
    ax1 = fig.add_subplot(gs[0, 0])
    for lb, co, mk, al in plot_cfgs:
        data = results_all[lb]
        ax1.scatter([r['depth'] for r in data],
                    [r['accuracy'] * 100 for r in data],
                    marker=mk, color=co, label=lb, alpha=al, s=50,
                    edgecolors='k', linewidth=0.3)
    ax1.set_xlabel("Circuit Depth"); ax1.set_ylabel("Accuracy (%)")
    ax1.set_title("(a) Accuracy vs Depth"); ax1.legend(fontsize=9)
    ax1.grid(True, alpha=.3)

    # (b) Accuracy vs Robustness
    ax2 = fig.add_subplot(gs[0, 1])
    for lb, co, mk, al in plot_cfgs:
        data = results_all[lb]
        ax2.scatter([r['robustness'] for r in data],
                    [r['accuracy'] * 100 for r in data],
                    marker=mk, color=co, label=lb, alpha=al, s=50,
                    edgecolors='k', linewidth=0.3)
    ax2.set_xlabel("Noise Robustness"); ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("(b) Accuracy vs Robustness"); ax2.legend(fontsize=9)
    ax2.grid(True, alpha=.3)

    # (c) Pareto front comparison
    ax3 = fig.add_subplot(gs[1, 0])

    def pareto_2d(data, kx='depth', ky='accuracy'):
        s = sorted(data, key=lambda r: r[kx])
        front, best_y = [], -1e9
        for r in s:
            if r[ky] > best_y:
                best_y = r[ky]; front.append(r)
        return sorted(front, key=lambda r: r[kx])

    for lb, co, mk in [('Random Search', '#9E9E9E', 'x'),
                        ('Single-Obj Genetic', '#FF9800', 's'),
                        ('Multi-Obj Genetic (Ours)', '#E91E63', 'D')]:
        front = pareto_2d(results_all[lb])
        ax3.plot([r['depth'] for r in front],
                 [r['accuracy'] * 100 for r in front],
                 marker=mk, color=co, label=lb, linewidth=2, markersize=8)
    ax3.set_xlabel("Circuit Depth"); ax3.set_ylabel("Accuracy (%)")
    ax3.set_title("(c) Pareto Front: Acc vs Depth"); ax3.legend(fontsize=9)
    ax3.grid(True, alpha=.3)

    # (d) Fitness bar
    ax4 = fig.add_subplot(gs[1, 1])
    methods = list(results_all.keys())
    mx = [max(r['fitness'] for r in results_all[m]) for m in methods]
    av = [float(np.mean([r['fitness'] for r in results_all[m]])) for m in methods]
    x = np.arange(len(methods)); w = 0.3
    b1 = ax4.bar(x - w / 2, mx, w, label='Best', color='#E91E63', alpha=.85)
    b2 = ax4.bar(x + w / 2, av, w, label='Avg', color='#2196F3', alpha=.85)
    for bg in [b1, b2]:
        for bar in bg:
            h = bar.get_height()
            ax4.annotate(f'{h:.3f}',
                         xy=(bar.get_x() + bar.get_width() / 2, h),
                         xytext=(0, 3), textcoords="offset points",
                         ha='center', fontsize=8)
    ax4.set_xticks(x)
    ax4.set_xticklabels(['Random', 'Single-Obj\nGenetic',
                          'Multi-Obj\nGenetic\n(Ours)'], fontsize=9)
    ax4.set_ylabel("Multi-Obj Fitness"); ax4.set_title("(d) Fitness")
    ax4.legend(); ax4.grid(True, alpha=.3, axis='y')

    fig.suptitle("Experiment 3: Multi-Objective Genetic Search", fontsize=15, y=1.01)
    p = os.path.join(OUTPUT_DIR, 'exp3_multi_objective_search.png')
    fig.savefig(p, dpi=150); plt.close(fig)
    logger.info(f"  Saved: {p}")

    logger.info("\n  Best solutions:")
    logger.info(f"  {'Method':<25} {'Fitness':>10} {'Acc':>10} {'Robust':>10} {'Depth':>8}")
    for m in methods:
        best = max(results_all[m], key=lambda r: r['fitness'])
        logger.info(f"  {m:<25} {best['fitness']:>10.4f} "
                     f"{best['accuracy']:>10.4f} "
                     f"{best['robustness']:>10.4f} {best['depth']:>8}")
    return results_all


# ============================================================================
# Combined Summary
# ============================================================================

def create_summary_figure(e1, e2, e3):
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    # Panel 1
    ax = axes[0]
    nms = list(e1.keys())
    fa = [e1[n]['accs'][-1] * 100 for n in nms]
    cs = ['#2196F3', '#FF9800', '#E91E63']
    bars = ax.bar(range(len(nms)), fa, color=cs, alpha=.85,
                  edgecolor='k', linewidth=.5)
    for b, v in zip(bars, fa):
        ax.annotate(f'{v:.1f}%', xy=(b.get_x() + b.get_width() / 2, v),
                    xytext=(0, 5), textcoords="offset points",
                    ha='center', fontsize=10, fontweight='bold')
    ax.set_xticks(range(len(nms)))
    ax.set_xticklabels(['Dense', 'Random\nMask', 'LT-Sparse\nInit (Ours)'], fontsize=10)
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title("Innovation 1:\nLottery-Ticket Sparse Init",
                 fontsize=12, fontweight='bold')
    ax.grid(True, alpha=.3, axis='y')

    # Panel 2
    ax = axes[1]
    cs2 = ['#2196F3', '#E91E63', '#4CAF50']
    for idx, (lb, d) in enumerate(e2.items()):
        ax.plot(d['noise_levels'], [a * 100 for a in d['accs']],
                marker='o', color=cs2[idx], label=lb, linewidth=2, markersize=5)
    ax.set_xlabel("Noise σ"); ax.set_ylabel("Test Accuracy (%)")
    ax.set_title("Innovation 2:\nNoise-Aware Pruning",
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=8, loc='lower left'); ax.grid(True, alpha=.3)

    # Panel 3
    ax = axes[2]
    ms = list(e3.keys())
    bf = [max(r['fitness'] for r in e3[m]) for m in ms]
    cs3 = ['#9E9E9E', '#FF9800', '#E91E63']
    bars = ax.bar(range(3), bf, color=cs3, alpha=.85,
                  edgecolor='k', linewidth=.5)
    for b, v in zip(bars, bf):
        ax.annotate(f'{v:.3f}', xy=(b.get_x() + b.get_width() / 2, v),
                    xytext=(0, 5), textcoords="offset points",
                    ha='center', fontsize=10, fontweight='bold')
    ax.set_xticks(range(3))
    ax.set_xticklabels(['Random', 'Single-Obj\nGenetic',
                         'Multi-Obj\nGenetic\n(Ours)'], fontsize=9)
    ax.set_ylabel("Multi-Obj Fitness")
    ax.set_title("Innovation 3:\nMulti-Obj Genetic Search",
                 fontsize=12, fontweight='bold')
    ax.grid(True, alpha=.3, axis='y')

    fig.suptitle("EQNAS: Verification of Three Innovation Points (v2)",
                 fontsize=16, fontweight='bold', y=1.04)
    plt.tight_layout()
    p = os.path.join(OUTPUT_DIR, 'eqnas_summary.png')
    fig.savefig(p, dpi=200); plt.close(fig)
    logger.info(f"\n  Summary figure saved: {p}")


# ============================================================================
# Main
# ============================================================================

def main():
    set_seed()
    set_run_dir(os.path.join(OUTPUT_DIR, 'runs'))
    logger.info("=" * 70)
    logger.info("EQNAS ABLATION STUDY  (v2 — fixed)")
    logger.info("=" * 70)

    dataflow = setup_env()
    e1 = experiment1_lottery_ticket(dataflow)
    e2 = experiment2_noise_aware_pruning(dataflow)
    e3 = experiment3_multi_objective_search(dataflow)
    create_summary_figure(e1, e2, e3)

    logger.info("\n" + "=" * 70)
    logger.info("ALL EXPERIMENTS COMPLETE")
    logger.info(f"Results in: {OUTPUT_DIR}/")
    logger.info("=" * 70)


if __name__ == '__main__':
    main()
