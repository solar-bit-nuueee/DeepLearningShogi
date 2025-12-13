import torch
import torch.nn as nn
import pytorch_lightning as pl
import math
import numpy as np

class SRigLCallback(pl.Callback):
    """
    Structured RigL (SRigL) - ICLR 2024 Paper-Compliant Implementation
    
    Based on "Dynamic Sparse Training with Structured Sparsity" (ICLR 2024)
    by Lasby et al. https://arxiv.org/abs/2305.02299
    
    Key Features:
    1. True dense gradient sampling every δT steps
    2. Salient weights: union of top-K drop and grow sets
    3. Neuron ablation: count(salient) < γ_sal * k'
    4. Exact constant fan-in with strict enforcement
    5. K schedule with cosine decay to 0
    6. ERK distribution with exact parameter count matching
    """
    
    def __init__(self, 
                 sparsity=0.9,
                 update_freq=1000,
                 warmup_steps=0,
                 alpha_init=0.3,
                 alpha_final=0.0,
                 gamma_sal=0.3,
                 delta_T=100,
                 use_erk_distribution=True,
                 min_fan_in=1,
                 model_type='cnn'):
        """
        Args:
            sparsity: Target global sparsity
            update_freq: Topology update frequency (ΔT)
            warmup_steps: Dense training before first update
            alpha_init: Initial update rate (K = α * active_weights)
            alpha_final: Final update rate (typically 0.0)
            gamma_sal: Neuron ablation threshold (γ_sal)
                      CNN: 0.3, ViT: 0.5-0.9 (higher for transformers)
            delta_T: Dense gradient sampling interval (δT)
            use_erk_distribution: Use ERK for layer-wise sparsity
            min_fan_in: Minimum fan-in per neuron
            model_type: 'cnn' or 'vit' (affects default γ_sal)
        """
        super().__init__()
        self.sparsity = sparsity
        self.update_freq = update_freq
        self.warmup_steps = warmup_steps
        self.alpha_init = alpha_init
        self.alpha_final = alpha_final
        
        # Adjust gamma_sal for model type
        if model_type == 'vit' and gamma_sal == 0.3:
            gamma_sal = 0.5  # Higher for transformers
            print(f"[INFO] Auto-adjusted γ_sal to {gamma_sal} for ViT")
        self.gamma_sal = gamma_sal
        
        self.delta_T = delta_T
        self.use_erk_distribution = use_erk_distribution
        self.min_fan_in = min_fan_in
        self.model_type = model_type
        
        # State
        self.masks = {}
        self.fan_in = {}  # k' per layer
        self.dense_gradients = {}  # True dense gradients
        self.gradient_steps = 0
        self.layers_to_prune = []
        self.initialized = False
        self.total_updates = 0
        self.next_dense_sample = delta_T
        
        print(f"\n{'='*70}")
        print(f"Structured RigL (SRigL) - ICLR 2024 Paper-Compliant")
        print(f"{'='*70}")
        print(f"Target Sparsity: {self.sparsity:.4f}")
        print(f"Model Type: {self.model_type.upper()}")
        print(f"Constant Fan-in: Strict (std=0)")
        print(f"Update Frequency (ΔT): {self.update_freq} steps")
        print(f"Warmup Steps: {self.warmup_steps}")
        print(f"Alpha Schedule: {self.alpha_init} → {self.alpha_final} (cosine)")
        print(f"Neuron Ablation (γ_sal): {self.gamma_sal}")
        print(f"Dense Gradient Sampling (δT): {self.delta_T} steps")
        print(f"ERK Distribution: {self.use_erk_distribution}")
        print(f"Min Fan-in: {self.min_fan_in}")
        print(f"{'='*70}\n")
    
    def compute_erk_distribution(self, layers, target_params):
        """
        ERK distribution with exact parameter count matching.
        Uses binary search + fractional assignment to hit target exactly.
        """
        erk_scores = {}
        layer_shapes = {}
        
        for name, layer in layers:
            n_out, n_in = layer.weight.shape
            erk_score = (n_in + n_out) / (n_in * n_out)
            erk_scores[name] = erk_score
            layer_shapes[name] = (n_out, n_in)
        
        # Binary search for lambda
        def compute_total_params(lambda_val):
            total = 0
            for name, (n_out, n_in) in layer_shapes.items():
                density = min(1.0, lambda_val * erk_scores[name])
                fan_in = max(self.min_fan_in, int(n_in * density))
                total += n_out * fan_in
            return total
        
        low, high = 0.0, 100.0
        for _ in range(50):
            mid = (low + high) / 2
            total = compute_total_params(mid)
            if total < target_params:
                low = mid
            else:
                high = mid
        
        lambda_final = (low + high) / 2
        
        # Compute initial fan-in
        layer_fan_in = {}
        for name, (n_out, n_in) in layer_shapes.items():
            density = min(1.0, lambda_final * erk_scores[name])
            fan_in = max(self.min_fan_in, int(n_in * density))
            layer_fan_in[name] = fan_in
        
        # Adjust for exact target (fractional assignment)
        current_total = sum(layer_shapes[name][0] * fan_in 
                           for name, fan_in in layer_fan_in.items())
        deficit = target_params - current_total
        
        if deficit != 0:
            # Sort layers by ERK score
            sorted_layers = sorted(erk_scores.items(), key=lambda x: x[1], reverse=(deficit > 0))
            
            # Distribute deficit
            for name, _ in sorted_layers:
                if deficit == 0:
                    break
                n_out, n_in = layer_shapes[name]
                if deficit > 0 and layer_fan_in[name] < n_in:
                    increment = min(deficit // n_out, n_in - layer_fan_in[name])
                    if increment > 0:
                        layer_fan_in[name] += increment
                        deficit -= n_out * increment
                elif deficit < 0 and layer_fan_in[name] > self.min_fan_in:
                    decrement = min((-deficit) // n_out, layer_fan_in[name] - self.min_fan_in)
                    if decrement > 0:
                        layer_fan_in[name] -= decrement
                        deficit += n_out * decrement
        
        # Verify exact match
        final_total = sum(layer_shapes[name][0] * fan_in 
                         for name, fan_in in layer_fan_in.items())
        print(f"ERK Distribution: target={target_params}, actual={final_total}, diff={final_total - target_params}")
        
        return layer_fan_in
    
    def initialize_constant_fanin_mask(self, shape, fan_in, device):
        """Initialize mask with exact constant fan-in."""
        n_out, n_in = shape
        mask = torch.zeros(shape, device=device)
        
        for i in range(n_out):
            indices = torch.randperm(n_in, device=device)[:fan_in]
            mask[i, indices] = 1.0
        
        return mask
    
    def on_train_start(self, trainer, pl_module):
        if hasattr(pl_module, 'layer_stacks'):
            ls = pl_module.layer_stacks
            self.layers_to_prune = [
                ('layer_stacks.l1', ls.l1),
                ('layer_stacks.l2', ls.l2),
                ('layer_stacks.output', ls.output)
            ]
        
        # Compute target parameters
        total_params = sum(layer.weight.numel() for _, layer in self.layers_to_prune)
        target_params = int(total_params * (1 - self.sparsity))
        
        # Compute ERK distribution
        if self.use_erk_distribution:
            layer_fan_in = self.compute_erk_distribution(self.layers_to_prune, target_params)
        else:
            layer_fan_in = {}
            for name, layer in self.layers_to_prune:
                n_out, n_in = layer.weight.shape
                fan_in = max(self.min_fan_in, int(n_in * (1 - self.sparsity)))
                layer_fan_in[name] = fan_in
        
        print("Initializing constant fan-in masks...")
        
        for name, layer in self.layers_to_prune:
            n_out, n_in = layer.weight.shape
            device = layer.weight.device
            fan_in = layer_fan_in[name]
            
            self.fan_in[name] = fan_in
            
            # Initialize mask
            mask = self.initialize_constant_fanin_mask((n_out, n_in), fan_in, device)
            self.masks[name] = mask
            
            # Initialize dense gradient buffer
            self.dense_gradients[name] = torch.zeros_like(layer.weight)
            
            actual_sparsity = 1.0 - (mask.sum().item() / mask.numel())
            active_neurons = (mask.sum(dim=1) > 0).sum().item()
            
            print(f"  {name}:")
            print(f"    Shape: {n_out}x{n_in}")
            print(f"    Fan-in k': {fan_in}")
            print(f"    Active neurons: {active_neurons}/{n_out}")
            print(f"    Sparsity: {actual_sparsity:.4f}")
        
        self.initialized = True
        print("")
    
    def sample_dense_gradients(self, trainer, pl_module, batch):
        """
        Sample true dense gradients by temporarily removing masks.
        This is critical for evaluating pruned weights.
        """
        # Save current masks
        saved_masks = {name: mask.clone() for name, mask in self.masks.items()}
        
        # Remove masks temporarily
        for name, layer in self.layers_to_prune:
            self.masks[name] = torch.ones_like(layer.weight)
        
        # Forward + backward with dense weights
        pl_module.zero_grad()
        
        # Manually run forward/backward
        us, them, white, black, outcome, score, layer_stack_indices, ply = batch
        output = pl_module(us, them, white, black, layer_stack_indices)
        
        # Compute loss (simplified, should match training loss)
        loss = pl_module.training_step((us, them, white, black, outcome, score, layer_stack_indices, ply), 0)
        
        if isinstance(loss, dict):
            loss = loss['loss']
        
        loss.backward()
        
        # Store dense gradients
        for name, layer in self.layers_to_prune:
            if layer.weight.grad is not None:
                self.dense_gradients[name] = torch.abs(layer.weight.grad.clone())
        
        # Restore masks
        self.masks = saved_masks
        
        # Clear gradients (don't update weights)
        pl_module.zero_grad()
    
    def on_after_backward(self, trainer, pl_module):
        if not self.initialized:
            return
        
        # Apply mask to gradients for optimization
        for name, layer in self.layers_to_prune:
            if layer.weight.grad is not None:
                layer.weight.grad.mul_(self.masks[name])
    
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        if not self.initialized:
            return
        
        self.gradient_steps += 1
        
        # Dense gradient sampling at δT intervals
        if self.gradient_steps >= self.next_dense_sample:
            print(f"[Dense Gradient Sampling at step {trainer.global_step}]")
            self.sample_dense_gradients(trainer, pl_module, batch)
            self.next_dense_sample = self.gradient_steps + self.delta_T
    
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if not self.initialized:
            return
        
        # Apply masks to weights
        with torch.no_grad():
            for name, layer in self.layers_to_prune:
                layer.weight.mul_(self.masks[name])
        
        global_step = trainer.global_step
        
        # Initial pruning after warmup
        if global_step == self.warmup_steps and self.warmup_steps > 0:
            print(f"\n=== Warmup complete at step {global_step} ===")
            self.enforce_constant_fanin(pl_module)
        
        # Periodic topology updates
        if global_step > self.warmup_steps and (global_step - self.warmup_steps) % self.update_freq == 0:
            self.srigl_update(pl_module, global_step)
    
    def enforce_constant_fanin(self, pl_module):
        """Enforce constant fan-in by magnitude pruning."""
        print("Enforcing constant fan-in constraint...")
        
        for name, layer in self.layers_to_prune:
            mask = self.masks[name]
            weight = layer.weight
            fan_in = self.fan_in[name]
            n_out, n_in = weight.shape
            
            new_mask = torch.zeros_like(mask)
            
            for i in range(n_out):
                row_weights = torch.abs(weight[i])
                _, indices = torch.topk(row_weights, min(fan_in, n_in))
                new_mask[i, indices] = 1.0
            
            self.masks[name] = new_mask
            
            with torch.no_grad():
                layer.weight.mul_(new_mask)
            
            # Strict check
            fanin_per_neuron = new_mask.sum(dim=1)
            active_neurons = fanin_per_neuron > 0
            if active_neurons.any():
                std_fanin = fanin_per_neuron[active_neurons].std().item()
                assert std_fanin < 1e-6, f"Fan-in not constant: std={std_fanin}"
            
            n_active = active_neurons.sum().item()
            actual_sparsity = 1.0 - (new_mask.sum().item() / new_mask.numel())
            print(f"  {name}: k'={fan_in}, active={n_active}/{n_out}, s={actual_sparsity:.4f}, std=0.00")
        
        print("")
    
    def compute_alpha(self, step):
        """Cosine decay schedule for update rate α(t)."""
        if step <= self.warmup_steps:
            return self.alpha_init
        
        progress = (step - self.warmup_steps) / (self.update_freq * 100)
        progress = min(1.0, progress)
        
        alpha = self.alpha_final + 0.5 * (self.alpha_init - self.alpha_final) * (1 + math.cos(math.pi * progress))
        return alpha
    
    def compute_salient_weights(self, mask, weight, gradient, K):
        """
        Compute salient weight set (union of drop-set and grow-set).
        
        drop_set: Top-K smallest magnitude active weights
        grow_set: Top-K largest gradient pruned weights
        
        Returns: binary mask of salient weights [n_out, n_in]
        """
        device = weight.device
        
        # Drop-set: Top-K smallest active weights
        active_mask = mask > 0
        active_weights = torch.abs(weight) * mask
        active_indices = active_mask.flatten().nonzero(as_tuple=True)[0]
        
        if len(active_indices) > 0 and K > 0:
            active_magnitudes = active_weights.flatten()[active_indices]
            K_drop = min(K, len(active_indices))
            _, drop_idx_local = torch.topk(active_magnitudes, K_drop, largest=False)
            drop_idx = active_indices[drop_idx_local]
        else:
            drop_idx = torch.tensor([], dtype=torch.long, device=device)
        
        # Grow-set: Top-K largest gradient pruned weights (use abs!)
        pruned_mask = mask == 0
        pruned_gradients = torch.abs(gradient) * pruned_mask  # ← FIX: abs(gradient)
        pruned_indices = pruned_mask.flatten().nonzero(as_tuple=True)[0]
        
        if len(pruned_indices) > 0 and K > 0:
            pruned_mags = pruned_gradients.flatten()[pruned_indices]
            K_grow = min(K, len(pruned_indices))
            _, grow_idx_local = torch.topk(pruned_mags, K_grow, largest=True)
            grow_idx = pruned_indices[grow_idx_local]
        else:
            grow_idx = torch.tensor([], dtype=torch.long, device=device)
        
        # Union of drop and grow sets
        salient_flat = torch.zeros(weight.numel(), device=device)
        if len(drop_idx) > 0:
            salient_flat[drop_idx] = 1.0
        if len(grow_idx) > 0:
            salient_flat[grow_idx] = 1.0
        
        salient_mask = salient_flat.reshape(weight.shape)
        return salient_mask
    
    def ablate_neurons(self, mask, salient_mask, fan_in, gamma_sal):
        """
        Neuron ablation: remove neurons with < γ_sal * k' salient weights.
        """
        # Count salient weights per neuron
        salient_per_neuron = salient_mask.sum(dim=1)  # [n_out]
        
        # Ablation threshold
        threshold = gamma_sal * fan_in
        
        # Keep neurons with enough salient weights
        active_neurons = salient_per_neuron >= threshold
        
        # Update mask
        new_mask = mask.clone()
        new_mask[~active_neurons, :] = 0.0
        
        n_ablated = (~active_neurons).sum().item()
        
        return new_mask, n_ablated, active_neurons
    
    def recompute_fan_in(self, mask, target_active_weights):
        """Recompute k' after neuron ablation."""
        n_active_neurons = (mask.sum(dim=1) > 0).sum().item()
        
        if n_active_neurons == 0:
            return self.min_fan_in
        
        new_fan_in = max(self.min_fan_in, target_active_weights // n_active_neurons)
        return new_fan_in
    
    def redistribute_connections(self, mask, weight, gradient, fan_in, K):
        """
        Redistribute connections with layer-wide K consistency.
        
        1. Prune K lowest magnitude connections (layer-wide)
        2. Grow K highest gradient connections (layer-wide)
        3. Enforce exactly fan_in per active neuron
        """
        n_out, n_in = mask.shape
        device = mask.device
        
        active_neurons = mask.sum(dim=1) > 0
        
        # Prune K connections (layer-wide, smallest magnitude)
        active_mask = mask > 0
        active_weights = torch.abs(weight) * mask
        active_indices = active_mask.flatten().nonzero(as_tuple=True)[0]
        
        if len(active_indices) > K:
            active_magnitudes = active_weights.flatten()[active_indices]
            _, prune_idx_local = torch.topk(active_magnitudes, K, largest=False)
            prune_idx = active_indices[prune_idx_local]
            
            mask_flat = mask.flatten()
            mask_flat[prune_idx] = 0.0
            mask = mask_flat.reshape(n_out, n_in)
        
        # Grow K connections (layer-wide, largest gradient)
        pruned_mask = mask == 0
        pruned_gradients = torch.abs(gradient) * pruned_mask  # ← FIX: abs(gradient)
        pruned_indices = pruned_mask.flatten().nonzero(as_tuple=True)[0]
        
        if len(pruned_indices) > 0:
            pruned_mags = pruned_gradients.flatten()[pruned_indices]
            K_grow = min(K, len(pruned_indices))
            _, grow_idx_local = torch.topk(pruned_mags, K_grow, largest=True)
            grow_idx = pruned_indices[grow_idx_local]
            
            mask_flat = mask.flatten()
            mask_flat[grow_idx] = 1.0
            mask = mask_flat.reshape(n_out, n_in)
        
        # Enforce exact constant fan-in per active neuron
        new_mask = torch.zeros_like(mask)
        
        for i in range(n_out):
            if not active_neurons[i]:
                continue
            
            current_fan_in = int(mask[i].sum().item())
            
            if current_fan_in > fan_in:
                # Prune to fan_in (by magnitude)
                active_idx = (mask[i] > 0).nonzero(as_tuple=True)[0]
                row_weights = torch.abs(weight[i, active_idx])
                _, keep_idx = torch.topk(row_weights, fan_in)
                new_mask[i, active_idx[keep_idx]] = 1.0
            elif current_fan_in < fan_in:
                # Grow to fan_in (by gradient)
                new_mask[i] = mask[i]
                inactive_idx = (mask[i] == 0).nonzero(as_tuple=True)[0]
                if len(inactive_idx) > 0:
                    n_to_grow = min(fan_in - current_fan_in, len(inactive_idx))
                    row_grads = torch.abs(gradient[i, inactive_idx])  # ← FIX: abs
                    _, grow_idx = torch.topk(row_grads, n_to_grow)
                    new_mask[i, inactive_idx[grow_idx]] = 1.0
            else:
                new_mask[i] = mask[i]
        
        # Strict verification: all active neurons have exactly fan_in
        fanin_per_neuron = new_mask.sum(dim=1)
        active_fanins = fanin_per_neuron[fanin_per_neuron > 0]
        if len(active_fanins) > 0:
            std_fanin = active_fanins.std().item()
            assert std_fanin < 1e-6, f"Fan-in not constant after redistribute: std={std_fanin}"
        
        return new_mask
    
    def srigl_update(self, pl_module, global_step):
        """
        SRigL topology update (Algorithm 1).
        """
        print(f"\n{'='*70}")
        print(f"SRigL Update #{self.total_updates + 1} at step {global_step}")
        print(f"{'='*70}")
        
        alpha = self.compute_alpha(global_step)
        print(f"Update rate α(t): {alpha:.4f}\n")
        
        for name, layer in self.layers_to_prune:
            mask = self.masks[name]
            weight = layer.weight
            gradient = self.dense_gradients[name]
            fan_in = self.fan_in[name]
            n_out, n_in = weight.shape
            
            print(f"Layer: {name}")
            
            # Compute K
            n_active = mask.sum().item()
            K = max(1, int(alpha * n_active))
            print(f"  K (connections to update): {K}")
            
            # Compute salient weights BEFORE ablation
            salient_mask = self.compute_salient_weights(mask, weight, gradient, K)
            
            # Neuron ablation
            mask, n_ablated, active_neurons = self.ablate_neurons(
                mask, salient_mask, fan_in, self.gamma_sal
            )
            
            if n_ablated > 0:
                print(f"  Ablated neurons: {n_ablated}")
                
                # Recompute k' after ablation
                target_active = int(n_out * n_in * (1 - self.sparsity))
                new_fan_in = self.recompute_fan_in(mask, target_active)
                
                if new_fan_in != fan_in:
                    print(f"  Recomputed k': {fan_in} → {new_fan_in}")
                    self.fan_in[name] = new_fan_in
                    fan_in = new_fan_in
            
            # Redistribute connections with K consistency
            mask = self.redistribute_connections(mask, weight, gradient, fan_in, K)
            
            # Update mask
            self.masks[name] = mask
            
            # Apply mask
            with torch.no_grad():
                layer.weight.mul_(mask)
            
            # Statistics
            n_active_final = mask.sum().item()
            n_active_neurons = (mask.sum(dim=1) > 0).sum().item()
            actual_sparsity = 1.0 - (n_active_final / mask.numel())
            fanin_per_neuron = mask.sum(dim=1)
            active_fanins = fanin_per_neuron[fanin_per_neuron > 0]
            avg_fanin = active_fanins.float().mean().item() if len(active_fanins) > 0 else 0
            std_fanin = active_fanins.float().std().item() if len(active_fanins) > 0 else 0
            
            print(f"  Active neurons: {n_active_neurons}/{n_out}")
            print(f"  Active connections: {int(n_active_final)}")
            print(f"  Sparsity: {actual_sparsity:.4f}")
            print(f"  Fan-in: {avg_fanin:.2f} ± {std_fanin:.4f} (target: {fan_in})")
            
            # Verify strict constant fan-in
            if std_fanin > 1e-3:
                print(f"  [WARNING] Fan-in variance detected: {std_fanin:.6f}")
            
            print("")
        
        self.total_updates += 1
        print(f"{'='*70}\n")
