"""
Quick test of Experiments 1 and 2 redesigned logic.
"""

import copy, os, sys, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.prune
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

# Path fix
_cwd = os.path.abspath(os.getcwd())
_removed = []
for _p in list(sys.path):
    _a = os.path.abspath(_p) if _p else _cwd
    _c = os.path.join(_a, 'torchquantum')
    if os.path.isdir(_c) and not os.path.isfile(os.path.join(_c, '__init__.py')):
        sys.path.remove(_p); _removed.append(_p)

import torchquantum as tq
from torchquantum.prune_utils import PhaseL1UnstructuredPruningMethod
from torchquantum.datasets import MNIST

from torchpack.utils.config import configs
from torchpack.utils.logging import logger
from torchpack.environ import set_run_dir

for _p in _removed:
    if _p not in sys.path:
        sys.path.append(_p)

from eqnas import SuperQFCModel, DEFAULT_CONFIG

DEVICE = torch.device('cpu')
CHECKPOINT = 'torchquantum/max-acc-valid.pt'
GENE = [4, 4, 4, 4, 4, 4, 3]


def setup():
    os.makedirs('eqnas_verification', exist_ok=True)
    config_path = 'eqnas_verification/configs.yml'
    with open(config_path, 'w') as f:
        f.write(DEFAULT_CONFIG)
    configs.load(config_path)
    if isinstance(configs.optimizer.lr, str):
        configs.optimizer.lr = eval(configs.optimizer.lr)
    if isinstance(configs.optimizer.weight_decay, str):
        configs.optimizer.weight_decay = eval(configs.optimizer.weight_decay)
    dataset = MNIST(
        root='./mnist_data', train_valid_split_ratio=[0.9, 0.1],
        digits_of_interest=[0, 1, 2, 3],
        n_test_samples=200, n_train_samples=2000, n_valid_samples=500,
    )
    dataflow = {}
    for split in dataset:
        sampler = torch.utils.data.RandomSampler(dataset[split])
        dataflow[split] = torch.utils.data.DataLoader(
            dataset[split], batch_size=configs.run.bsz,
            sampler=sampler, num_workers=0, pin_memory=False,
        )
    return dataflow


def load_model(gene=None):
    import __main__
    __main__.SuperQFCModel0 = SuperQFCModel
    model = SuperQFCModel(configs.model.arch)
    ckpt = torch.load(CHECKPOINT, map_location='cpu', weights_only=False)
    if isinstance(ckpt, dict) and 'model' in ckpt:
        model.load_state_dict(ckpt['model'], strict=False)
    elif isinstance(ckpt, nn.Module):
        model.load_state_dict(ckpt.state_dict(), strict=False)
    else:
        model.load_state_dict(ckpt, strict=False)
    if gene is not None:
        model.set_sample_arch(gene)
    return model


def get_ptp(model):
    return [(m, 'params') for _, m in model.named_modules()
            if isinstance(m, tq.Operator) and m.params is not None]


def build_mask(model):
    return {name: (mod.params.data != 0).float().clone()
            for name, mod in model.named_modules()
            if isinstance(mod, tq.Operator) and mod.params is not None}


def apply_mask(model, masks):
    with torch.no_grad():
        for name, mod in model.named_modules():
            if name in masks and hasattr(mod, 'params') and mod.params is not None:
                mod.params.data *= masks[name]


def eval_acc(model, gene, loader):
    model.eval()
    model.set_sample_arch(gene)
    ok = tot = 0
    with torch.no_grad():
        for fd in loader:
            x = fd[configs.dataset.input_name]
            y = fd[configs.dataset.target_name]
            ok += (model(x).argmax(1) == y).sum().item()
            tot += y.size(0)
    return ok / max(tot, 1)


def output_cosine(m1, m2, loader, gene, n=3):
    m1.eval(); m2.eval()
    m1.set_sample_arch(gene); m2.set_sample_arch(gene)
    sims = []
    with torch.no_grad():
        for i, fd in enumerate(loader):
            if i >= n:
                break
            x = fd[configs.dataset.input_name]
            sims.extend(F.cosine_similarity(m1(x), m2(x), dim=1).tolist())
    return float(np.mean(sims))


def train_quick(model, gene, dataflow, masks, noise_scale, n_epochs, lr=5e-2):
    """
    Correct noise-aware training (STE pattern):
      Standard (noise_scale=0): normal SGD
      Noise-aware: save θ → add ε ~ N(0,σ²) → forward(θ+ε) → restore θ → backward → step on clean θ
    """
    model.set_sample_arch(gene)
    crit = nn.NLLLoss()
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = CosineAnnealingLR(opt, T_max=n_epochs)
    hist = []

    for _ in range(n_epochs):
        model.train()
        for fd in dataflow['train']:
            x = fd[configs.dataset.input_name]
            y = fd[configs.dataset.target_name]
            opt.zero_grad()

            if noise_scale > 0:
                # Save clean params
                saved = {}
                for name, mod in model.named_modules():
                    if isinstance(mod, tq.Operator) and mod.params is not None:
                        saved[name] = mod.params.data.clone()
                        mod.params.data.add_(
                            torch.randn_like(mod.params) * noise_scale
                        )
                        if masks and name in masks:
                            mod.params.data.mul_(masks[name])
                # Forward on noisy params
                out = model(x)
                loss = crit(out, y)
                # Restore clean params BEFORE backward (STE)
                for name, mod in model.named_modules():
                    if name in saved:
                        mod.params.data.copy_(saved[name])
                # Backward + step operates on clean θ
                loss.backward()
                opt.step()
            else:
                out = model(x)
                loss = crit(out, y)
                loss.backward()
                opt.step()

            apply_mask(model, masks)

        sched.step()
        hist.append(eval_acc(model, gene, dataflow['test']))

    return hist


def eval_noise_stats(model, gene, loader, noise_levels, n_trial=15):
    orig = copy.deepcopy(model.state_dict())
    means, stds = [], []
    for nl in noise_levels:
        if nl == 0:
            means.append(eval_acc(model, gene, loader))
            stds.append(0.)
        else:
            accs = []
            for _ in range(n_trial):
                with torch.no_grad():
                    for _, m in model.named_modules():
                        if isinstance(m, tq.Operator) and m.params is not None:
                            m.params.data += torch.randn_like(m.params) * nl
                accs.append(eval_acc(model, gene, loader))
                model.load_state_dict(orig)
            means.append(float(np.mean(accs)))
            stds.append(float(np.std(accs)))
    return means, stds


# ============================================================
# EXP 1 Quick Test
# ============================================================
print("\n" + "=" * 60)
print("QUICK TEST: EXP 1 — Phase-Aware Pruning")
print("=" * 60)

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
set_run_dir('eqnas_verification/runs')
dataflow = setup()

# Reference model
ref = load_model(GENE)
ref_acc = eval_acc(ref, GENE, dataflow['test'])
print(f"Reference acc: {ref_acc*100:.1f}%")

# Compute gradient importance
grad_accum = {}
for name, mod in ref.named_modules():
    if isinstance(mod, tq.Operator) and mod.params is not None:
        grad_accum[name] = torch.zeros_like(mod.params.data)

ref.train()
ref.set_sample_arch(GENE)
crit_ref = nn.NLLLoss()
for i, fd in enumerate(dataflow['train']):
    if i >= 3:
        break
    x = fd[configs.dataset.input_name]
    y = fd[configs.dataset.target_name]
    ref.zero_grad()
    crit_ref(ref(x), y).backward()
    for name, mod in ref.named_modules():
        if name in grad_accum and mod.params is not None and mod.params.grad is not None:
            grad_accum[name] += mod.params.grad.abs()

N_EPOCHS_E1 = 10
sparsity_levels = [0.3, 0.5, 0.7]
methods = ['Random', 'L1', 'PhaseL1', 'Gradient', 'Combined']
table = {m: {} for m in methods}

for sp in sparsity_levels:
    print(f"\n  Sparsity {sp:.0%}:")
    for method in methods:
        torch.manual_seed(42)
        model = load_model(GENE)
        ptp = get_ptp(model)

        if method == 'Random':
            if ptp:
                nn.utils.prune.global_unstructured(
                    ptp, pruning_method=nn.utils.prune.RandomUnstructured, amount=sp)
                for m, n in ptp:
                    nn.utils.prune.remove(m, n)

        elif method == 'L1':
            if ptp:
                nn.utils.prune.global_unstructured(
                    ptp, pruning_method=nn.utils.prune.L1Unstructured, amount=sp)
                for m, n in ptp:
                    nn.utils.prune.remove(m, n)

        elif method == 'PhaseL1':
            if ptp:
                nn.utils.prune.global_unstructured(
                    ptp, pruning_method=PhaseL1UnstructuredPruningMethod, amount=sp)
                for m, n in ptp:
                    nn.utils.prune.remove(m, n)

        elif method == 'Gradient':
            all_g = torch.cat([v.view(-1) for v in grad_accum.values()])
            n_p = int(sp * all_g.numel())
            thr = torch.topk(all_g, k=n_p, largest=False).values[-1]
            for name, mod in model.named_modules():
                if isinstance(mod, tq.Operator) and mod.params is not None and name in grad_accum:
                    mod.params.data *= (grad_accum[name] > thr).float()

        elif method == 'Combined':
            all_scores = []
            score_map = {}
            for name, mod in model.named_modules():
                if isinstance(mod, tq.Operator) and mod.params is not None:
                    theta = mod.params.data
                    phase = theta % (2 * np.pi)
                    phase[phase > np.pi] -= 2 * np.pi
                    phase_s = torch.abs(phase)
                    grad_s = grad_accum.get(name, torch.ones_like(theta)) + 1e-8
                    score = phase_s * torch.sqrt(grad_s)
                    score_map[name] = score
                    all_scores.append(score.view(-1))
            if all_scores:
                all_s = torch.cat(all_scores)
                n_p = int(sp * all_s.numel())
                thr = torch.topk(all_s, k=n_p, largest=False).values[-1]
                for name, mod in model.named_modules():
                    if name in score_map:
                        mod.params.data *= (score_map[name] > thr).float()

        ref_fresh = load_model(GENE)
        fidelity = output_cosine(ref_fresh, model, dataflow['test'], GENE)
        zs_acc = eval_acc(model, GENE, dataflow['test'])
        masks = build_mask(model)

        m_train = load_model(GENE)
        apply_mask(m_train, masks)
        hist = train_quick(m_train, GENE, dataflow, masks, 0.0, N_EPOCHS_E1)

        table[method][sp] = {'fidelity': fidelity, 'zs_acc': zs_acc, 'final': hist[-1]}
        print(f"    {method:<12}: fid={fidelity:.3f}  zs={zs_acc*100:5.1f}%  final={hist[-1]*100:5.1f}%")

print("\nExp1 Summary:")
print(f"  {'Method':<12}  sparsity=30%           sparsity=50%           sparsity=70%")
print(f"  {'':12}  fid   zs    final   fid   zs    final   fid   zs    final")
for m in methods:
    row = f"  {m:<12}"
    for sp in sparsity_levels:
        r = table[m][sp]
        row += f"  {r['fidelity']:.3f} {r['zs_acc']*100:5.1f}% {r['final']*100:5.1f}%"
    print(row)


# ============================================================
# EXP 2 Quick Test — noise-aware training with STE, 50% sparsity
# ============================================================
print("\n" + "=" * 60)
print("QUICK TEST: EXP 2 — Noise-Aware Training (STE, 50% sparsity)")
print("=" * 60)

sp_e2 = 0.5
noise_levels_eval = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
N_EPOCHS_E2 = 30
N_TRIAL = 15

# Shared mask: PhaseL1 at 50% sparsity
torch.manual_seed(42)
m_mask = load_model(GENE)
ptp = get_ptp(m_mask)
if ptp:
    nn.utils.prune.global_unstructured(
        ptp, pruning_method=PhaseL1UnstructuredPruningMethod, amount=sp_e2)
    for m, n in ptp:
        nn.utils.prune.remove(m, n)
shared_mask = build_mask(m_mask)
actual_sp = (
    sum((v == 0).sum().item() for v in shared_mask.values()) /
    max(sum(v.numel() for v in shared_mask.values()), 1)
)
print(f"Mask sparsity: {actual_sp:.1%}")

conditions = [
    ('Standard (σ=0)',      0.0),
    ('Noise-Aware (σ=0.3)', 0.3),
    ('Noise-Aware (σ=0.5)', 0.5),
    ('Noise-Aware (σ=0.7)', 0.7),
]

exp2_results = {}
for label, ns in conditions:
    torch.manual_seed(42)
    print(f"\n  Training: {label} for {N_EPOCHS_E2} epochs...")
    model = load_model(GENE)
    apply_mask(model, shared_mask)
    hist = train_quick(model, GENE, dataflow, shared_mask, ns, N_EPOCHS_E2, lr=5e-2)
    means, stds = eval_noise_stats(model, GENE, dataflow['test'], noise_levels_eval, N_TRIAL)

    clean_acc = means[0]
    auc = float(np.trapz(means, noise_levels_eval))
    high_noise_score = float(np.mean([
        means[i] for i, nl in enumerate(noise_levels_eval) if nl >= 0.4
    ]))
    retentions = [m / max(clean_acc, 1e-8) * 100 for m in means]
    avg_retention = float(np.mean([
        retentions[i] for i, nl in enumerate(noise_levels_eval) if nl >= 0.3
    ]))

    exp2_results[label] = {
        'means': means, 'stds': stds, 'auc': auc, 'clean': clean_acc,
        'robust': high_noise_score, 'retentions': retentions, 'avg_ret': avg_retention,
    }
    print(f"    Clean={clean_acc*100:.1f}%  AUC={auc:.4f}  "
          f"Robust(σ≥0.4)={high_noise_score*100:.1f}%  "
          f"Retention(σ≥0.3)={avg_retention:.1f}%")

print("\nExp2 Summary — Raw accuracy (%):")
header = f"  {'Condition':<26}"
for nl in noise_levels_eval:
    header += f" {str(nl):>5}"
header += f"  {'Robust':>7}  {'Retain%':>8}"
print(header)
for label, d in exp2_results.items():
    row = f"  {label:<26}"
    for v in d['means']:
        row += f" {v*100:>5.1f}"
    row += f"  {d['robust']*100:>7.1f}  {d['avg_ret']:>8.1f}"
    print(row)

print("\nExp2 Summary — Retention ratio (noisy/clean × 100%):")
ret_header = f"  {'Condition':<26}"
for nl in noise_levels_eval:
    ret_header += f" {str(nl):>5}"
print(ret_header)
for label, d in exp2_results.items():
    row = f"  {label:<26}"
    for r in d['retentions']:
        row += f" {r:>5.1f}"
    print(row)
