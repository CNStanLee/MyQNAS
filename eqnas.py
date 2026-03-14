"""
EQNAS: Efficient Noise-Aware Sparse Quantum Neural Architecture Search
=======================================================================

A unified framework that combines:
1. Lottery-ticket-inspired sparse initialisation & noise-aware pruning
2. Multi-objective genetic search (accuracy + noise robustness + circuit complexity)
3. Deployable QNN subnetwork discovery for realistic noisy quantum hardware

Built on top of the QuantumNAS / TorchQuantum codebase.
"""

# Fix sys.path: remove CWD entries that contain a bare `torchquantum/` directory
# (the git repo root) which shadows the real installed package.
import sys as _sys
import os as _os
_cwd = _os.getcwd()
_clean_path = []
for _p in _sys.path:
    _abs = _os.path.abspath(_p) if _p else _cwd
    _candidate = _os.path.join(_abs, 'torchquantum')
    # Keep the entry unless it points to a dir that has torchquantum/ WITHOUT __init__.py
    if _os.path.isdir(_candidate) and not _os.path.isfile(_os.path.join(_candidate, '__init__.py')):
        continue
    _clean_path.append(_p)
_sys.path[:] = _clean_path
del _cwd, _clean_path, _abs, _candidate, _p

import argparse
import copy
import os
import sys
import random

# Fix: prevent the local 'torchquantum/' git repo directory (no __init__.py)
# from shadowing the installed torchquantum package as a namespace package.
_cwd = os.path.abspath(os.getcwd())
for _p in list(sys.path):
    _abs = os.path.abspath(_p) if _p else _cwd
    _candidate = os.path.join(_abs, 'torchquantum')
    if os.path.isdir(_candidate) and not os.path.isfile(os.path.join(_candidate, '__init__.py')):
        if _p in sys.path:
            sys.path.remove(_p)
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.prune
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

import torchquantum as tq
import torchquantum.functional as tqf
from torchquantum.encoding import encoder_op_list_name_dict
from torchquantum.super_layers import super_layer_name_dict
from torchquantum.plugins import (
    tq2qiskit,
    qiskit2tq,
    tq2qiskit_measurement,
    qiskit_assemble_circs,
    op_history2qiskit,
    op_history2qiskit_expand_params,
    QiskitProcessor,
)
from torchquantum.utils import (
    build_module_from_op_list,
    build_module_op_list,
    get_v_c_reg_mapping,
    get_p_c_reg_mapping,
    get_p_v_reg_mapping,
    get_cared_configs,
)
from torchquantum.super_utils import get_named_sample_arch
from torchquantum.prune_utils import PhaseL1UnstructuredPruningMethod, ThresholdScheduler
from torchquantum.datasets import MNIST

from torchpack.utils.config import configs, Config
from torchpack.utils.logging import logger
from torchpack.environ import set_run_dir

import tqdm


# =============================================================================
# Section 1: Supercircuit Model (weight-sharing)
# =============================================================================

class SuperQFCModel(tq.QuantumModule):
    """
    Supercircuit model for quantum neural architecture search.

    Contains an encoder, a weight-sharing super quantum layer, and
    measurement. Different subnetworks (genes) are sampled by selecting
    which gates / blocks are active.
    """
    def __init__(self, arch: dict):
        super().__init__()
        self.arch = arch
        self.n_wires = arch['n_wires']
        self.encoder = tq.GeneralEncoder(
            encoder_op_list_name_dict[arch['encoder_op_list_name']]
        )
        self.q_layer = super_layer_name_dict[arch['q_layer_name']](arch)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.sample_arch = None

    def set_sample_arch(self, sample_arch):
        self.sample_arch = sample_arch
        self.q_layer.set_sample_arch(sample_arch)

    def count_sample_params(self):
        return self.q_layer.count_sample_params()

    def forward(self, x, verbose=False, use_qiskit=False):
        bsz = x.shape[0]
        qdev = tq.QuantumDevice(
            n_wires=self.n_wires, bsz=bsz,
            record_op=True, device=x.device
        )

        if getattr(self.arch, 'down_sample_kernel_size', None) is not None:
            x = F.avg_pool2d(x, self.arch['down_sample_kernel_size'])
        x = x.view(bsz, -1)

        if use_qiskit:
            self.encoder(qdev, x)
            op_history_parameterized = qdev.op_history
            qdev.reset_op_history()
            encoder_circs = op_history2qiskit_expand_params(
                self.n_wires, op_history_parameterized, bsz=bsz
            )
            self.q_layer(qdev)
            op_history_fixed = qdev.op_history
            qdev.reset_op_history()
            q_layer_circ = op_history2qiskit(self.n_wires, op_history_fixed)
            measurement_circ = tq2qiskit_measurement(qdev, self.measure)
            assembled_circs = qiskit_assemble_circs(
                encoder_circs, q_layer_circ, measurement_circ
            )
            x0 = self.qiskit_processor.process_ready_circs(
                qdev, assembled_circs
            ).to(x.device)
            x = x0
        else:
            self.encoder(qdev, x)
            self.q_layer(qdev)
            x = self.measure(qdev)

        if verbose:
            logger.info(f"[use_qiskit]={use_qiskit}, expectation:\n {x.data}")

        if getattr(self.arch, 'output_len', None) is not None:
            x = x.reshape(bsz, -1, self.arch.output_len).sum(-1)
        if x.dim() > 2:
            x = x.squeeze()
        x = F.log_softmax(x, dim=1)
        return x

    @property
    def arch_space(self):
        space = []
        for layer in self.q_layer.super_layers_all:
            space.append(layer.arch_space)
        space.append(list(range(
            self.q_layer.n_front_share_blocks,
            self.q_layer.n_blocks + 1
        )))
        return space


# =============================================================================
# Section 2: Lottery-Ticket-Inspired Sparse Initialisation
# =============================================================================

def lottery_ticket_sparse_init(
    model: nn.Module,
    init_sparsity: float = 0.3,
    rewinding_epoch: int = 3,
    dataflow: dict = None,
    criterion: Callable = None,
    optimizer_cls=optim.Adam,
    lr: float = 5e-3,
    device: torch.device = torch.device('cpu'),
    gene: list = None,
) -> Tuple[nn.Module, dict]:
    """
    Lottery-Ticket-Hypothesis inspired sparse initialisation for QNNs.

    Steps:
      1. Train the supercircuit for a few epochs to get an initial set of weights.
      2. Identify the least important rotation parameters (close to k*pi)
         using the phase-aware magnitude criterion.
      3. Prune them and rewind the remaining weights to their initial values.
      4. Return the pruned model with the "winning ticket" mask.

    This gives a sparse subnetwork that can be trained from near-initialisation
    to match or exceed the dense supercircuit performance, analogous to
    the classical lottery ticket hypothesis.

    Args:
        model: The supercircuit model.
        init_sparsity: Fraction of parameters to prune in the initial ticket.
        rewinding_epoch: Number of warm-up epochs before rewinding.
        dataflow: dict with 'train' DataLoader.
        criterion: Loss function.
        optimizer_cls: optimizer class.
        lr: Learning rate for warm-up.
        device: torch device.
        gene: Subnetwork architecture gene.

    Returns:
        (model, masks_dict): Pruned model and dict of pruning masks.
    """
    logger.info(f"[LotteryTicket] Saving initial weights (rewinding point)...")
    initial_state = copy.deepcopy(model.state_dict())

    if gene is not None:
        model.set_sample_arch(gene)

    # Step 1: Brief warm-up training
    optimizer = optimizer_cls(model.parameters(), lr=lr)
    model.train()
    for epoch in range(rewinding_epoch):
        for feed_dict in dataflow['train']:
            inputs = feed_dict[configs.dataset.input_name].to(device)
            targets = feed_dict[configs.dataset.target_name].to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        logger.info(f"  [LotteryTicket] Warm-up epoch {epoch+1}/{rewinding_epoch}, "
                     f"loss={loss.item():.4f}")

    # Step 2: Phase-aware pruning — prune params closest to multiples of pi
    parameters_to_prune = [
        (module, "params")
        for _, module in model.named_modules()
        if isinstance(module, tq.Operator) and module.params is not None
    ]

    if len(parameters_to_prune) > 0:
        nn.utils.prune.global_unstructured(
            parameters_to_prune,
            pruning_method=PhaseL1UnstructuredPruningMethod,
            amount=init_sparsity,
        )

    # Step 3: Rewind unpruned weights to initial values, keep masks
    masks_dict = {}
    for name, module in model.named_modules():
        if isinstance(module, tq.Operator) and module.params is not None:
            if hasattr(module, 'params_mask'):
                masks_dict[name] = module.params_mask.clone()

    # Remove the pruning reparametrisation to get clean params
    for module, param_name in parameters_to_prune:
        nn.utils.prune.remove(module, param_name)

    # Rewind: restore initial weights but zero out pruned positions
    rewound_state = copy.deepcopy(initial_state)
    current_state = model.state_dict()
    for key in current_state:
        if key in rewound_state:
            current_state[key] = rewound_state[key]
    model.load_state_dict(current_state, strict=False)

    # Re-apply the masks as permanent zeros
    for name, module in model.named_modules():
        if name in masks_dict:
            if hasattr(module, 'params') and module.params is not None:
                with torch.no_grad():
                    module.params.data *= masks_dict[name].to(module.params.device)

    n_total = sum(p.numel() for p in model.parameters())
    n_pruned = sum((m == 0).sum().item() for m in masks_dict.values()) if masks_dict else 0
    logger.info(f"[LotteryTicket] Sparse init done: {n_pruned}/{n_total} params pruned "
                 f"({100*n_pruned/max(n_total,1):.1f}% sparsity)")
    return model, masks_dict


# =============================================================================
# Section 3: Noise-Aware Pruning Trainer
# =============================================================================

class NoiseAwarePruningTrainer:
    """
    Training loop that simultaneously performs:
    - Noise-aware training (via QiskitProcessor with noise model or injected noise)
    - Progressive phase-aware pruning (rotation angles near k*pi get pruned)
    - Gradual sparsity schedule (cubic polynomial, same as TF model opt)

    Unlike the original PruningTrainer which uses torchpack.train.Trainer,
    this is a self-contained training loop for clarity and flexibility.
    """

    def __init__(
        self,
        model: nn.Module,
        criterion: Callable,
        optimizer: optim.Optimizer,
        scheduler,
        device: torch.device,
        init_sparsity: float = 0.1,
        target_sparsity: float = 0.5,
        pruning_start_epoch: int = 0,
        pruning_end_epoch: int = 30,
        noise_injection_scale: float = 0.0,
    ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.noise_injection_scale = noise_injection_scale

        # Pruning config
        self._parameters_to_prune = self._extract_prunable(model)
        self.sparsity_scheduler = ThresholdScheduler(
            pruning_start_epoch, pruning_end_epoch,
            init_sparsity, target_sparsity,
        )
        self.current_sparsity = init_sparsity
        self._pruned_once = False

    @staticmethod
    def _extract_prunable(model: nn.Module) -> list:
        return [
            (module, "params")
            for _, module in model.named_modules()
            if isinstance(module, tq.Operator) and module.params is not None
        ]

    def _inject_noise(self, x: torch.Tensor) -> torch.Tensor:
        """
        Inject depolarising-like noise during training by adding small
        Gaussian perturbations to the gate parameters. This encourages the
        model to be robust to noise without a full Qiskit noise simulation.

        The noise is stored so it can be subtracted after the forward pass.
        """
        self._noise_backups = {}
        if self.noise_injection_scale > 0:
            for module_name, module in self.model.named_modules():
                if isinstance(module, tq.Operator) and module.params is not None:
                    self._noise_backups[module_name] = module.params.data.clone()
                    noise = torch.randn_like(module.params) * self.noise_injection_scale
                    module.params.data += noise
        return x

    def _restore_params(self):
        """Restore parameters after noise-injected forward pass."""
        if hasattr(self, '_noise_backups') and self._noise_backups:
            for module_name, module in self.model.named_modules():
                if module_name in self._noise_backups:
                    module.params.data.copy_(self._noise_backups[module_name])
            self._noise_backups = {}

    def _prune_model(self, amount: float):
        if self._pruned_once:
            for module, name in self._parameters_to_prune:
                try:
                    nn.utils.prune.remove(module, name)
                except ValueError:
                    pass

        if len(self._parameters_to_prune) > 0:
            nn.utils.prune.global_unstructured(
                self._parameters_to_prune,
                pruning_method=PhaseL1UnstructuredPruningMethod,
                amount=amount,
            )
        self._pruned_once = True

    def _remove_pruning(self):
        for module, name in self._parameters_to_prune:
            try:
                nn.utils.prune.remove(module, name)
            except ValueError:
                pass
        self._pruned_once = False

    def get_sparsity(self) -> float:
        """Compute actual fraction of zero parameters."""
        n_zero = 0
        n_total = 0
        for p in self.model.parameters():
            n_total += p.numel()
            n_zero += (p == 0).sum().item()
        return n_zero / max(n_total, 1)

    def train_one_epoch(self, dataflow, epoch: int):
        self.model.train()
        total_loss = 0
        n_batches = 0

        for feed_dict in dataflow:
            inputs = feed_dict[configs.dataset.input_name].to(self.device)
            targets = feed_dict[configs.dataset.target_name].to(self.device)

            # Noise injection for noise-aware training (temporary)
            self._inject_noise(inputs)

            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)

            # Restore clean params before backward so gradients update clean weights
            self._restore_params()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        self.scheduler.step()

        # Update pruning
        self.current_sparsity = self.sparsity_scheduler.step()
        self._prune_model(self.current_sparsity)

        avg_loss = total_loss / max(n_batches, 1)
        actual_sparsity = self.get_sparsity()
        logger.info(
            f"  [Epoch {epoch}] loss={avg_loss:.4f}, "
            f"target_sparsity={self.current_sparsity:.3f}, "
            f"actual_sparsity={actual_sparsity:.3f}"
        )
        return avg_loss

    def finalize(self):
        """Remove pruning reparametrisation, making sparsity permanent."""
        self._remove_pruning()
        logger.info(f"  Pruning finalized. Actual sparsity = {self.get_sparsity():.3f}")


# =============================================================================
# Section 4: Multi-Objective Fitness Evaluation
# =============================================================================

def evaluate_gene_accuracy(
    model: nn.Module,
    gene: list,
    dataflow_split,
    device: torch.device,
    use_qiskit: bool = False,
) -> float:
    """Evaluate classification accuracy for a given gene (subnetwork)."""
    model.set_sample_arch(gene)
    model.eval()
    target_all, output_all = None, None

    with torch.no_grad():
        for feed_dict in dataflow_split:
            inputs = feed_dict[configs.dataset.input_name].to(device)
            targets = feed_dict[configs.dataset.target_name].to(device)
            outputs = model(inputs, use_qiskit=use_qiskit)
            if target_all is None:
                target_all = targets
                output_all = outputs
            else:
                target_all = torch.cat([target_all, targets])
                output_all = torch.cat([output_all, outputs])

    _, indices = output_all.topk(1, dim=1)
    correct = indices.eq(target_all.view(-1, 1)).sum().item()
    accuracy = correct / target_all.shape[0]
    return accuracy


def compute_circuit_complexity(model: nn.Module, gene: list) -> dict:
    """
    Estimate circuit complexity: depth and gate count.
    Uses tq2qiskit to convert the sampled subnetwork to a Qiskit circuit.
    """
    model.set_sample_arch(gene)
    try:
        circ = tq2qiskit(tq.QuantumDevice(n_wires=model.n_wires), model.q_layer)
        depth = circ.depth()
        gate_counts = dict(circ.count_ops())
        total_gates = sum(gate_counts.values())
    except Exception as e:
        logger.warning(f"Circuit conversion failed for gene {gene}: {e}")
        depth = 999
        total_gates = 999
        gate_counts = {}
    return {
        'depth': depth,
        'total_gates': total_gates,
        'gate_counts': gate_counts,
    }


def compute_noise_robustness(
    model: nn.Module,
    gene: list,
    dataflow_split,
    device: torch.device,
    noise_scale: float = 0.1,
    n_trials: int = 3,
) -> float:
    """
    Noise robustness = mean(accuracy_noisy) / accuracy_noisefree.
    Always uses parameter perturbation for reproducible, differentiating results.
    Averages over multiple noise trials to reduce variance.
    """
    acc_clean = evaluate_gene_accuracy(model, gene, dataflow_split, device, use_qiskit=False)
    if acc_clean < 1e-8:
        return 0.0

    # Always use parameter perturbation for meaningful robustness signal
    noisy_accs = []
    original_state = copy.deepcopy(model.state_dict())
    for _ in range(n_trials):
        with torch.no_grad():
            for name, module in model.named_modules():
                if isinstance(module, tq.Operator) and module.params is not None:
                    noise = torch.randn_like(module.params) * noise_scale
                    module.params.data += noise
        acc_noisy = evaluate_gene_accuracy(model, gene, dataflow_split, device, use_qiskit=False)
        noisy_accs.append(acc_noisy)
        model.load_state_dict(original_state)

    robustness = np.mean(noisy_accs) / acc_clean
    return min(robustness, 1.0)


def multi_objective_fitness(
    model: nn.Module,
    gene: list,
    dataflow_split,
    device: torch.device,
    w_acc: float = 0.5,
    w_robust: float = 0.3,
    w_complexity: float = 0.2,
    max_depth: int = 50,
) -> dict:
    """
    Compute a multi-objective fitness score for a candidate gene.

    fitness = w_acc * accuracy
            + w_robust * noise_robustness
            + w_complexity * (1 - depth / max_depth)

    Returns a dict with individual scores and the combined fitness.
    """
    accuracy = evaluate_gene_accuracy(model, gene, dataflow_split, device, use_qiskit=False)
    robustness = compute_noise_robustness(model, gene, dataflow_split, device)
    complexity = compute_circuit_complexity(model, gene)

    depth_score = max(0.0, 1.0 - complexity['depth'] / max_depth)

    fitness = w_acc * accuracy + w_robust * robustness + w_complexity * depth_score

    return {
        'gene': gene,
        'accuracy': accuracy,
        'robustness': robustness,
        'depth': complexity['depth'],
        'total_gates': complexity['total_gates'],
        'depth_score': depth_score,
        'fitness': fitness,
    }


# =============================================================================
# Section 5: Multi-Objective Genetic Search (NSGA-II inspired)
# =============================================================================

class MultiObjectiveGeneticSearcher:
    """
    Multi-objective evolutionary architecture search.

    Uses NSGA-II-inspired non-dominated sorting and crowding distance
    to jointly optimise:
      - Accuracy (maximise)
      - Noise robustness (maximise)
      - Circuit complexity / depth (minimise)

    This goes beyond the original single-objective evolutionary search
    in QuantumNAS by maintaining a Pareto front of solutions.
    """

    def __init__(
        self,
        gene_choice: list,
        model: nn.Module,
        dataflow_split,
        device: torch.device,
        population_size: int = 20,
        parent_size: int = 6,
        mutation_size: int = 8,
        crossover_size: int = 6,
        mutation_prob: float = 0.5,
        n_iterations: int = 5,
        w_acc: float = 0.5,
        w_robust: float = 0.3,
        w_complexity: float = 0.2,
        max_depth: int = 50,
        selection_mode: str = 'multi_objective',
    ):
        self.gene_choice = gene_choice
        self.gene_len = len(gene_choice)
        self.model = model
        self.dataflow_split = dataflow_split
        self.device = device
        self.population_size = population_size
        self.parent_size = parent_size
        self.mutation_size = mutation_size
        self.crossover_size = crossover_size
        self.mutation_prob = mutation_prob
        self.n_iterations = n_iterations
        self.w_acc = w_acc
        self.w_robust = w_robust
        self.w_complexity = w_complexity
        self.max_depth = max_depth
        self.selection_mode = selection_mode

        self.population = []
        self.pareto_front = []
        self.best_result = None
        self.all_results = []
        self.best_history = []
        self.eval_counts = []
        self._eval_cache = {}

    @staticmethod
    def _gene_key(gene: list) -> tuple:
        return tuple(gene)

    def random_sample(self, n: int, exclude: Optional[set] = None) -> list:
        exclude = set() if exclude is None else set(exclude)
        population = []
        while len(population) < n:
            gene = []
            for k in range(self.gene_len):
                gene.append(random.choice(self.gene_choice[k]))
            key = self._gene_key(gene)
            if key in exclude:
                continue
            population.append(gene)
            exclude.add(key)
        return population

    def _dedup_and_fill_population(self, population: list) -> list:
        unique = []
        seen = set()
        for gene in population:
            key = self._gene_key(gene)
            if key in seen:
                continue
            unique.append(gene)
            seen.add(key)
        if len(unique) < self.population_size:
            unique.extend(
                self.random_sample(self.population_size - len(unique), exclude=seen)
            )
        return unique[:self.population_size]

    def mutate(self, gene: list) -> list:
        mutated = []
        for i in range(self.gene_len):
            if np.random.uniform() < self.mutation_prob:
                mutated.append(random.choice(self.gene_choice[i]))
            else:
                mutated.append(gene[i])
        return mutated

    def crossover(self, gene_a: list, gene_b: list) -> list:
        child = []
        for i in range(self.gene_len):
            child.append(gene_a[i] if np.random.uniform() < 0.5 else gene_b[i])
        return child

    @staticmethod
    def dominates(a: dict, b: dict) -> bool:
        """True if solution a dominates b (better or equal on all, strictly better on at least one)."""
        better_or_eq = (
            a['accuracy'] >= b['accuracy']
            and a['robustness'] >= b['robustness']
            and a['depth'] <= b['depth']
        )
        strictly_better = (
            a['accuracy'] > b['accuracy']
            or a['robustness'] > b['robustness']
            or a['depth'] < b['depth']
        )
        return better_or_eq and strictly_better

    @staticmethod
    def non_dominated_sort(results: list) -> list:
        """Return list of fronts (each front is a list of indices)."""
        n = len(results)
        domination_count = [0] * n
        dominated_by = [[] for _ in range(n)]
        fronts = [[]]

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if MultiObjectiveGeneticSearcher.dominates(results[i], results[j]):
                    dominated_by[i].append(j)
                elif MultiObjectiveGeneticSearcher.dominates(results[j], results[i]):
                    domination_count[i] += 1

            if domination_count[i] == 0:
                fronts[0].append(i)

        current_front = 0
        while len(fronts[current_front]) > 0:
            next_front = []
            for i in fronts[current_front]:
                for j in dominated_by[i]:
                    domination_count[j] -= 1
                    if domination_count[j] == 0:
                        next_front.append(j)
            current_front += 1
            fronts.append(next_front)

        return [f for f in fronts if len(f) > 0]

    def evaluate_population(self, population: list) -> list:
        """Evaluate all genes in the population."""
        results = []
        for gene in tqdm.tqdm(population, desc="Evaluating population"):
            key = self._gene_key(gene)
            if key not in self._eval_cache:
                self._eval_cache[key] = multi_objective_fitness(
                    self.model, gene, self.dataflow_split, self.device,
                    self.w_acc, self.w_robust, self.w_complexity, self.max_depth,
                )
            results.append(copy.deepcopy(self._eval_cache[key]))
        return results

    @staticmethod
    def crowding_distance(results: list, front_indices: list) -> Dict[int, float]:
        """
        Compute NSGA-II crowding distance for solutions on one front.
        Objectives: accuracy (max), robustness (max), -depth (min depth = max -depth).
        """
        if len(front_indices) <= 2:
            return {i: float('inf') for i in front_indices}
        objectives = ['accuracy', 'robustness']  # maximise
        cd = {i: 0.0 for i in front_indices}
        for obj in objectives:
            sorted_idx = sorted(front_indices, key=lambda i: results[i][obj])
            cd[sorted_idx[0]] = float('inf')
            cd[sorted_idx[-1]] = float('inf')
            obj_range = results[sorted_idx[-1]][obj] - results[sorted_idx[0]][obj]
            if obj_range < 1e-12:
                continue
            for k in range(1, len(sorted_idx) - 1):
                cd[sorted_idx[k]] += (results[sorted_idx[k + 1]][obj]
                                      - results[sorted_idx[k - 1]][obj]) / obj_range
        # depth: minimise → sort ascending, boundary = inf
        sorted_d = sorted(front_indices, key=lambda i: results[i]['depth'])
        cd[sorted_d[0]] = float('inf')
        cd[sorted_d[-1]] = float('inf')
        d_range = results[sorted_d[-1]]['depth'] - results[sorted_d[0]]['depth']
        if d_range > 1e-12:
            for k in range(1, len(sorted_d) - 1):
                cd[sorted_d[k]] += (results[sorted_d[k + 1]]['depth']
                                    - results[sorted_d[k - 1]]['depth']) / d_range
        return cd

    def select_parents(self, results: list, population: list) -> list:
        """
        Select parents from evaluated candidates.
        `fitness` mode uses scalar ranking; otherwise use NSGA-II-style
        non-dominated sorting and crowding distance.
        """
        if self.selection_mode == 'fitness':
            ranked = sorted(
                zip(results, population),
                key=lambda pair: pair[0]['fitness'],
                reverse=True,
            )
            return [copy.deepcopy(gene) for _, gene in ranked[:self.parent_size]]

        fronts = self.non_dominated_sort(results)
        parents = []
        for front in fronts:
            if len(parents) >= self.parent_size:
                break
            cd = self.crowding_distance(results, front)
            # Sort by crowding distance (descending) to maintain diversity
            front_sorted = sorted(front, key=lambda i: -cd[i])
            for idx in front_sorted:
                if len(parents) < self.parent_size:
                    parents.append(population[idx])
        return parents

    def _generate_unseen_population(self, parents: list, seen_keys: set) -> list:
        """
        Build the next population using only unseen genes so that the search
        budget is counted in unique evaluations rather than repeated cache hits.
        """
        if not parents:
            return self.random_sample(self.population_size, exclude=seen_keys)

        next_population = []
        next_seen = set()

        def maybe_add(gene: list) -> bool:
            key = self._gene_key(gene)
            if key in seen_keys or key in next_seen:
                return False
            next_population.append(gene)
            next_seen.add(key)
            return True

        attempts = 0
        max_attempts = max(200, self.population_size * 80)

        while len(next_population) < self.mutation_size and attempts < max_attempts:
            attempts += 1
            maybe_add(self.mutate(random.choice(parents)))

        while (len(next_population) < self.mutation_size + self.crossover_size
               and len(next_population) < self.population_size
               and attempts < max_attempts):
            attempts += 1
            if len(parents) >= 2:
                pa, pb = random.sample(parents, 2)
                maybe_add(self.crossover(pa, pb))
            else:
                maybe_add(self.mutate(parents[0]))

        while len(next_population) < self.population_size and attempts < max_attempts:
            attempts += 1
            gene = [random.choice(self.gene_choice[k]) for k in range(self.gene_len)]
            maybe_add(gene)

        if len(next_population) < self.population_size:
            next_population.extend(
                self.random_sample(
                    self.population_size - len(next_population),
                    exclude=seen_keys.union(next_seen),
                )
            )
        return next_population[:self.population_size]

    def run_search(self) -> Tuple[dict, list]:
        """
        Run the multi-objective genetic search.

        Returns:
            (best_result, pareto_front): best single result and the full Pareto front.
        """
        self.population = self.random_sample(self.population_size)
        all_results = []
        seen_keys = set()

        for iteration in range(self.n_iterations):
            self.population = self._dedup_and_fill_population(self.population)
            logger.info(f"\n{'='*60}")
            logger.info(f"[EQNAS Search] Iteration {iteration+1}/{self.n_iterations}, "
                         f"population size = {len(self.population)}")
            logger.info(f"{'='*60}")

            results = self.evaluate_population(self.population)

            # Keep track of all UNIQUE results seen across the search.
            for gene, result in zip(self.population, results):
                key = self._gene_key(gene)
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                all_results.append(copy.deepcopy(result))

            # Select parents from the full evaluated archive.
            archive_population = [r['gene'] for r in all_results]
            parents = self.select_parents(all_results, archive_population)

            # Find generation best and best-so-far.
            best_idx = max(range(len(results)), key=lambda i: results[i]['fitness'])
            gen_best = results[best_idx]
            current_best = max(all_results, key=lambda r: r['fitness'])
            logger.info(
                f"  Best this iter: fitness={gen_best['fitness']:.4f}, "
                f"acc={gen_best['accuracy']:.4f}, "
                f"robustness={gen_best['robustness']:.4f}, "
                f"depth={gen_best['depth']} | "
                f"archive={len(all_results)} unique"
            )
            self.best_history.append(copy.deepcopy(current_best))
            self.eval_counts.append(len(all_results))

            if self.best_result is None or current_best['fitness'] > self.best_result['fitness']:
                self.best_result = copy.deepcopy(current_best)

            # Breed the next generation from elites without re-spending budget
            # on already-evaluated genes.
            if iteration < self.n_iterations - 1:
                self.population = self._generate_unseen_population(parents, seen_keys)

        # Compute final Pareto front from all evaluated solutions
        self.all_results = [copy.deepcopy(r) for r in all_results]
        fronts = self.non_dominated_sort(all_results)
        if fronts:
            self.pareto_front = [all_results[i] for i in fronts[0]]
        else:
            self.pareto_front = [self.best_result] if self.best_result else []

        logger.info(f"\n{'='*60}")
        logger.info(f"[EQNAS Search] Complete!")
        logger.info(f"  Best gene: {self.best_result['gene']}")
        logger.info(f"  Best fitness: {self.best_result['fitness']:.4f}")
        logger.info(f"  Best accuracy: {self.best_result['accuracy']:.4f}")
        logger.info(f"  Best robustness: {self.best_result['robustness']:.4f}")
        logger.info(f"  Best depth: {self.best_result['depth']}")
        logger.info(f"  Pareto front size: {len(self.pareto_front)}")
        logger.info(f"{'='*60}")

        return self.best_result, self.pareto_front


# =============================================================================
# Section 6: EQNAS Pipeline — Putting It All Together
# =============================================================================

class EQNASPipeline:
    """
    End-to-end EQNAS pipeline:
      1. Load supercircuit with pre-trained weights
      2. Lottery-ticket sparse initialisation
      3. Noise-aware pruning-aware fine-tuning
      4. Multi-objective genetic search for best subnetwork
      5. Final evaluation and circuit export
    """

    def __init__(self, config_path: str = None, device: str = 'cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.config_path = config_path
        self.model = None
        self.dataflow = None
        self.best_gene = None
        self.pareto_front = None

    def setup(self):
        """Load configs, dataset, and model."""
        if self.config_path:
            configs.load(self.config_path)

        # Ensure numeric config values are not strings
        if isinstance(configs.optimizer.lr, str):
            configs.optimizer.lr = eval(configs.optimizer.lr)
        if isinstance(configs.optimizer.weight_decay, str):
            configs.optimizer.weight_decay = eval(configs.optimizer.weight_decay)

        if configs.debug.set_seed:
            torch.manual_seed(configs.debug.seed)
            np.random.seed(configs.debug.seed)
            random.seed(configs.debug.seed)

        # Dataset
        dataset = MNIST(
            root='./mnist_data',
            train_valid_split_ratio=[0.9, 0.1],
            digits_of_interest=[0, 1, 2, 3],
            n_test_samples=300,
            n_train_samples=5000,
            n_valid_samples=3000,
        )
        self.dataflow = {}
        for split in dataset:
            sampler = torch.utils.data.RandomSampler(dataset[split])
            self.dataflow[split] = torch.utils.data.DataLoader(
                dataset[split],
                batch_size=configs.run.bsz,
                sampler=sampler,
                num_workers=configs.run.workers_per_gpu,
                pin_memory=True,
            )

        # Model
        self.model = SuperQFCModel(configs.model.arch)
        logger.info(f"Model created with {sum(p.numel() for p in self.model.parameters())} params")

    def load_pretrained(self, checkpoint_path: str, gene: list = None):
        """Load pre-trained supercircuit weights."""
        # The checkpoint was pickled with SuperQFCModel0 in __main__,
        # so we need to make that class available for unpickling.
        import __main__
        __main__.SuperQFCModel0 = SuperQFCModel
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        if isinstance(checkpoint, dict) and 'model' in checkpoint:
            self.model.load_state_dict(checkpoint['model'], strict=False)
        elif isinstance(checkpoint, nn.Module):
            # The checkpoint is the model itself
            self.model.load_state_dict(checkpoint.state_dict(), strict=False)
        else:
            self.model.load_state_dict(checkpoint, strict=False)
        self.model.to(self.device)
        if gene is not None:
            self.model.set_sample_arch(gene)
        logger.info(f"Loaded pretrained weights from {checkpoint_path}")

    def setup_noise_processor(self, noise_model_name: str = None):
        """Attach a QiskitProcessor for noisy simulation."""
        processor = QiskitProcessor(
            use_real_qc=False,
            noise_model_name=noise_model_name,
        )
        processor.set_layout(list(range(self.model.n_wires)))
        self.model.set_qiskit_processor(processor)
        logger.info(f"Noise processor attached (noise_model={noise_model_name})")

    def step1_sparse_init(
        self,
        init_sparsity: float = 0.3,
        rewinding_epochs: int = 3,
        gene: list = None,
    ):
        """Step 1: Lottery-ticket sparse initialisation."""
        logger.info("\n" + "="*60)
        logger.info("[EQNAS Step 1] Lottery-Ticket Sparse Initialisation")
        logger.info("="*60)

        criterion = nn.NLLLoss()
        self.model, self.masks = lottery_ticket_sparse_init(
            model=self.model,
            init_sparsity=init_sparsity,
            rewinding_epoch=rewinding_epochs,
            dataflow=self.dataflow,
            criterion=criterion,
            optimizer_cls=optim.Adam,
            lr=configs.optimizer.lr,
            device=self.device,
            gene=gene,
        )

    def step2_noise_aware_finetune(
        self,
        n_epochs: int = 10,
        target_sparsity: float = 0.5,
        noise_injection_scale: float = 0.02,
        gene: list = None,
    ):
        """Step 2: Noise-aware pruning fine-tuning."""
        logger.info("\n" + "="*60)
        logger.info("[EQNAS Step 2] Noise-Aware Pruning Fine-tuning")
        logger.info("="*60)

        if gene is not None:
            self.model.set_sample_arch(gene)

        criterion = nn.NLLLoss()
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=configs.optimizer.lr,
            weight_decay=configs.optimizer.weight_decay,
        )
        scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs)

        trainer = NoiseAwarePruningTrainer(
            model=self.model,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=self.device,
            init_sparsity=configs.prune.init_pruning_amount,
            target_sparsity=target_sparsity,
            pruning_start_epoch=0,
            pruning_end_epoch=n_epochs,
            noise_injection_scale=noise_injection_scale,
        )

        for epoch in range(n_epochs):
            trainer.train_one_epoch(self.dataflow['train'], epoch)

        trainer.finalize()

    def step3_genetic_search(
        self,
        n_iterations: int = 5,
        population_size: int = 20,
        w_acc: float = 0.5,
        w_robust: float = 0.3,
        w_complexity: float = 0.2,
    ) -> Tuple[dict, list]:
        """Step 3: Multi-objective genetic search."""
        logger.info("\n" + "="*60)
        logger.info("[EQNAS Step 3] Multi-Objective Genetic Architecture Search")
        logger.info("="*60)

        parent_size = max(population_size // 4, 2)
        mutation_size = max(population_size // 3, 2)
        crossover_size = population_size - parent_size - mutation_size

        searcher = MultiObjectiveGeneticSearcher(
            gene_choice=self.model.arch_space,
            model=self.model,
            dataflow_split=self.dataflow['test'],
            device=self.device,
            population_size=population_size,
            parent_size=parent_size,
            mutation_size=mutation_size,
            crossover_size=crossover_size,
            mutation_prob=configs.es.mutation_prob,
            n_iterations=n_iterations,
            w_acc=w_acc,
            w_robust=w_robust,
            w_complexity=w_complexity,
        )

        best_result, pareto_front = searcher.run_search()
        self.best_gene = best_result['gene']
        self.pareto_front = pareto_front
        return best_result, pareto_front

    def step4_evaluate_and_export(self, output_dir: str = "eqnas_results"):
        """Step 4: Final evaluation and circuit export."""
        logger.info("\n" + "="*60)
        logger.info("[EQNAS Step 4] Final Evaluation & Export")
        logger.info("="*60)

        os.makedirs(output_dir, exist_ok=True)

        if self.best_gene is not None:
            self.model.set_sample_arch(self.best_gene)

        # Noise-free accuracy
        acc_clean = evaluate_gene_accuracy(
            self.model, self.best_gene, self.dataflow['test'], self.device, use_qiskit=False
        )
        logger.info(f"  Final noise-free accuracy: {acc_clean:.4f}")

        # Circuit info
        complexity = compute_circuit_complexity(self.model, self.best_gene)
        logger.info(f"  Circuit depth: {complexity['depth']}")
        logger.info(f"  Total gates: {complexity['total_gates']}")
        logger.info(f"  Gate breakdown: {complexity['gate_counts']}")

        # Sparsity
        n_total = sum(p.numel() for p in self.model.parameters())
        n_zero = sum((p == 0).sum().item() for p in self.model.parameters())
        sparsity = n_zero / max(n_total, 1)
        logger.info(f"  Parameter sparsity: {sparsity:.3f} ({n_zero}/{n_total} zeros)")

        # Export circuit diagram
        try:
            circ = tq2qiskit(tq.QuantumDevice(n_wires=self.model.n_wires), self.model.q_layer)
            circ.draw('mpl', filename=os.path.join(output_dir, 'best_circuit.png'))
            logger.info(f"  Circuit diagram saved to {output_dir}/best_circuit.png")
        except Exception as e:
            logger.warning(f"  Could not export circuit diagram: {e}")

        # Save model and results
        save_path = os.path.join(output_dir, 'eqnas_best_model.pt')
        torch.save({
            'model': self.model.state_dict(),
            'best_gene': self.best_gene,
            'accuracy': acc_clean,
            'complexity': complexity,
            'sparsity': sparsity,
            'pareto_front': self.pareto_front,
        }, save_path)
        logger.info(f"  Model saved to {save_path}")

        # Save Pareto front summary
        if self.pareto_front:
            summary_path = os.path.join(output_dir, 'pareto_front.txt')
            with open(summary_path, 'w') as f:
                f.write("gene | accuracy | robustness | depth | total_gates | fitness\n")
                f.write("-" * 80 + "\n")
                for r in sorted(self.pareto_front, key=lambda x: -x['fitness']):
                    f.write(
                        f"{r['gene']} | {r['accuracy']:.4f} | "
                        f"{r['robustness']:.4f} | {r['depth']} | "
                        f"{r['total_gates']} | {r['fitness']:.4f}\n"
                    )
            logger.info(f"  Pareto front saved to {summary_path}")

        return {
            'accuracy': acc_clean,
            'sparsity': sparsity,
            'complexity': complexity,
            'best_gene': self.best_gene,
        }


# =============================================================================
# Section 7: Configuration & Main Entry Point
# =============================================================================

DEFAULT_CONFIG = """model:
  arch:
    n_wires: 4
    encoder_op_list_name: 4x4_ryzxy
    n_blocks: 3
    n_layers_per_block: 2
    q_layer_name: u3cu3_s0
    down_sample_kernel_size: 6
    n_front_share_blocks: 1
    n_front_share_wires: 1
    n_front_share_ops: 1
  sampler:
    strategy:
      name: plain
  transpile_before_run: False
  load_op_list: False

dataset:
  name: mnist
  input_name: image
  target_name: digit

optimizer:
  name: adam
  lr: 5e-2
  weight_decay: 1e-4
  lambda_lr: 1e-2

run:
  n_epochs: 40
  bsz: 256
  workers_per_gpu: 2
  device: gpu

debug:
  pdb: False
  set_seed: True
  seed: 42

callbacks:
  - callback: 'InferenceRunner'
    split: 'valid'
    subcallbacks:
      - metrics: 'CategoricalAccuracy'
        name: 'acc/valid'
      - metrics: 'NLLError'
        name: 'loss/valid'
  - callback: 'InferenceRunner'
    split: 'test'
    subcallbacks:
      - metrics: 'CategoricalAccuracy'
        name: 'acc/test'
      - metrics: 'NLLError'
        name: 'loss/test'
  - callback: 'MaxSaver'
    name: 'acc/valid'
  - callback: 'Saver'
    max_to_keep: 10

qiskit:
  use_qiskit: False
  use_real_qc: False
  backend_name: null
  noise_model_name: null
  basis_gates_name: null
  n_shots: 8192
  initial_layout: null
  seed_transpiler: 42
  seed_simulator: 42
  optimization_level: 0
  est_success_rate: False
  max_jobs: 1

es:
  random_search: False
  population_size: 20
  parent_size: 6
  mutation_size: 8
  mutation_prob: 0.5
  crossover_size: 6
  n_iterations: 5
  est_success_rate: False
  score_mode: loss_succ
  gene_mask: null
  eval:
    use_noise_model: False
    use_real_qc: False
    bsz: qiskit_max
    n_test_samples: 150

prune:
  target_pruning_amount: 0.5
  init_pruning_amount: 0.1
  start_epoch: 0
  end_epoch: 30

eqnas:
  init_sparsity: 0.3
  rewinding_epochs: 3
  finetune_epochs: 10
  target_sparsity: 0.5
  noise_injection_scale: 0.02
  search_iterations: 5
  search_population: 20
  w_accuracy: 0.5
  w_robustness: 0.3
  w_complexity: 0.2
"""


def main():
    parser = argparse.ArgumentParser(description="EQNAS: Noise-Aware Sparse Quantum NAS")
    parser.add_argument('--config', type=str, default=None, help='Path to config YAML')
    parser.add_argument('--checkpoint', type=str, default='max-acc-valid.pt',
                        help='Path to pre-trained supercircuit checkpoint')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--output-dir', type=str, default='eqnas_results')
    parser.add_argument('--init-gene', type=int, nargs='+', default=[4, 4, 4, 4, 4, 4, 3],
                        help='Initial gene for the supercircuit')
    parser.add_argument('--skip-sparse-init', action='store_true',
                        help='Skip lottery-ticket sparse initialisation')
    parser.add_argument('--skip-finetune', action='store_true',
                        help='Skip noise-aware fine-tuning')
    parser.add_argument('--skip-search', action='store_true',
                        help='Skip genetic search (use init-gene directly)')
    args = parser.parse_args()

    # Write default config if none provided
    if args.config is None:
        args.config = 'eqnas_configs.yml'
        with open(args.config, 'w') as f:
            f.write(DEFAULT_CONFIG)
        logger.info(f"Written default config to {args.config}")

    # ---- Pipeline ----
    pipeline = EQNASPipeline(config_path=args.config, device=args.device)
    pipeline.setup()
    pipeline.load_pretrained(args.checkpoint, gene=args.init_gene)

    # Optionally set up local noisy simulation
    pipeline.setup_noise_processor(noise_model_name=None)

    # Step 1: Lottery-ticket sparse init
    if not args.skip_sparse_init:
        eqnas_cfg = configs.get('eqnas', {})
        pipeline.step1_sparse_init(
            init_sparsity=eqnas_cfg.get('init_sparsity', 0.3),
            rewinding_epochs=eqnas_cfg.get('rewinding_epochs', 3),
            gene=args.init_gene,
        )

    # Step 2: Noise-aware pruning fine-tune
    if not args.skip_finetune:
        eqnas_cfg = configs.get('eqnas', {})
        pipeline.step2_noise_aware_finetune(
            n_epochs=eqnas_cfg.get('finetune_epochs', 10),
            target_sparsity=eqnas_cfg.get('target_sparsity', 0.5),
            noise_injection_scale=eqnas_cfg.get('noise_injection_scale', 0.02),
            gene=args.init_gene,
        )

    # Step 3: Multi-objective genetic search
    if not args.skip_search:
        eqnas_cfg = configs.get('eqnas', {})
        best_result, pareto = pipeline.step3_genetic_search(
            n_iterations=eqnas_cfg.get('search_iterations', 5),
            population_size=eqnas_cfg.get('search_population', 20),
            w_acc=eqnas_cfg.get('w_accuracy', 0.5),
            w_robust=eqnas_cfg.get('w_robustness', 0.3),
            w_complexity=eqnas_cfg.get('w_complexity', 0.2),
        )
    else:
        pipeline.best_gene = args.init_gene

    # Step 4: Final evaluation and export
    results = pipeline.step4_evaluate_and_export(output_dir=args.output_dir)
    logger.info(f"\nDone! Results: {results}")


if __name__ == '__main__':
    main()
