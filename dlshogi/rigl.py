import torch
import torch.nn as nn
import pytorch_lightning as pl
import math

class StructuredRigLCallback(pl.Callback):
    """
    Structured Rigging the Lottery (RigL) with Neuron-level Sparsity.
    
    RigL is a dynamic sparse training method that:
    1. Starts from random sparse initialization
    2. Periodically removes low-magnitude connections
    3. Regrows connections based on gradient magnitude
    
    Structured variant operates on entire neurons (rows/columns) for hardware efficiency.
    
    Reference:
    Evci et al. "Rigging the Lottery: Making All Tickets Winners" (2020)
    """
    
    def __init__(self, 
                 sparsity=0.9,
                 update_freq=1000,
                 warmup_steps=0,
                 prune_rate=0.2,
                 use_random_init=True,
                 structure='neuron',  # 'neuron', 'row', 'column', or 'unstructured'
                 momentum_based_regrowth=False,
                 momentum_beta=0.9):
        """
        Args:
            sparsity: Target sparsity (0.9 = 90% zeros)
            update_freq: Topology update frequency in steps
            warmup_steps: Dense training steps before first sparse update
            prune_rate: Fraction of active weights to prune each update
            use_random_init: Random sparse initialization (RigL default)
            structure: Sparsity structure ('neuron', 'row', 'column', 'unstructured')
            momentum_based_regrowth: Use gradient momentum for regrowth
            momentum_beta: Momentum coefficient for gradient tracking
        """
        super().__init__()
        self.sparsity = sparsity
        self.update_freq = update_freq
        self.warmup_steps = warmup_steps
        self.prune_rate = prune_rate
        self.use_random_init = use_random_init
        self.structure = structure
        self.momentum_based_regrowth = momentum_based_regrowth
        self.momentum_beta = momentum_beta
        
        self.masks = {}
        self.gradient_momentum = {}
        self.layers_to_prune = []
        self.initialized = False
        
        print(f"\n{'='*60}")
        print(f"Structured RigL Dynamic Sparse Training")
        print(f"{'='*60}")
        print(f"Target Sparsity: {self.sparsity}")
        print(f"Structure: {self.structure}")
        print(f"Update Frequency: {self.update_freq} steps")
        print(f"Warmup Steps: {self.warmup_steps}")
        print(f"Prune Rate: {self.prune_rate}")
        print(f"Random Init: {self.use_random_init}")
        print(f"Momentum Regrowth: {self.momentum_based_regrowth}")
        if self.momentum_based_regrowth:
            print(f"Momentum Beta: {self.momentum_beta}")
        print(f"{'='*60}\n")
    
    def initialize_random_sparse_mask(self, shape, sparsity, device):
        """
        Initialize random sparse mask.
        RigL paper uses uniform random sparsity.
        """
        total_params = shape[0] * shape[1]
        n_active = int(total_params * (1 - sparsity))
        
        mask = torch.zeros(total_params, device=device)
        indices = torch.randperm(total_params, device=device)[:n_active]
        mask[indices] = 1.0
        
        return mask.reshape(shape)
    
    def get_neuron_importance(self, weight, mask, gradient=None):
        """
        Compute neuron-level importance scores.
        
        For output neurons (rows): sum of |weight| or |gradient| across inputs
        For input neurons (columns): sum of |weight| or |gradient| across outputs
        """
        if gradient is not None:
            # Use gradient magnitude for importance
            scores = torch.abs(gradient * mask)
        else:
            # Use weight magnitude
            scores = torch.abs(weight * mask)
        
        if self.structure == 'neuron' or self.structure == 'row':
            # Output neuron importance
            return scores.sum(dim=1)
        elif self.structure == 'column':
            # Input neuron importance
            return scores.sum(dim=0)
        else:
            # Unstructured
            return scores.flatten()
    
    def prune_structured(self, mask, weight, n_to_prune):
        """
        Structured pruning: remove entire neurons.
        """
        device = weight.device
        
        if self.structure == 'neuron' or self.structure == 'row':
            # Prune output neurons (rows)
            active_neurons = (mask.sum(dim=1) > 0).float()
            neuron_importance = self.get_neuron_importance(weight, mask)
            
            # Only consider active neurons
            neuron_importance = neuron_importance * active_neurons
            
            n_active = int(active_neurons.sum().item())
            n_to_prune_clamped = min(n_to_prune, n_active)
            
            if n_to_prune_clamped <= 0:
                return mask
            
            # Select least important neurons
            _, indices = torch.topk(neuron_importance, n_active - n_to_prune_clamped, largest=True)
            
            new_mask = torch.zeros_like(mask)
            new_mask[indices, :] = mask[indices, :]
            
            return new_mask
            
        elif self.structure == 'column':
            # Prune input neurons (columns)
            active_neurons = (mask.sum(dim=0) > 0).float()
            neuron_importance = self.get_neuron_importance(weight, mask)
            
            neuron_importance = neuron_importance * active_neurons
            
            n_active = int(active_neurons.sum().item())
            n_to_prune_clamped = min(n_to_prune, n_active)
            
            if n_to_prune_clamped <= 0:
                return mask
            
            _, indices = torch.topk(neuron_importance, n_active - n_to_prune_clamped, largest=True)
            
            new_mask = torch.zeros_like(mask)
            new_mask[:, indices] = mask[:, indices]
            
            return new_mask
        
        else:
            # Unstructured pruning (standard RigL)
            importance = torch.abs(weight * mask).flatten()
            n_active = int(mask.sum().item())
            n_to_keep = n_active - n_to_prune
            
            if n_to_keep <= 0:
                return torch.zeros_like(mask)
            
            _, indices = torch.topk(importance, n_to_keep)
            new_mask = torch.zeros(mask.numel(), device=device)
            new_mask[indices] = 1.0
            
            return new_mask.reshape(mask.shape)
    
    def regrow_structured(self, mask, weight, gradient, n_to_add):
        """
        Structured regrowth: add entire neurons based on gradient magnitude.
        RigL uses gradient magnitude to identify promising connections.
        """
        device = weight.device
        
        if n_to_add <= 0:
            return mask
        
        if self.structure == 'neuron' or self.structure == 'row':
            # Regrow output neurons
            inactive_neurons = (mask.sum(dim=1) == 0).float()
            
            if self.momentum_based_regrowth and gradient is not None:
                # Use gradient momentum for regrowth scores
                neuron_scores = self.get_neuron_importance(weight, 1 - mask, gradient)
            else:
                # Use instantaneous gradient
                if gradient is not None:
                    neuron_scores = torch.abs(gradient * (1 - mask)).sum(dim=1)
                else:
                    # Fallback to random
                    neuron_scores = torch.rand(mask.shape[0], device=device)
            
            neuron_scores = neuron_scores * inactive_neurons
            
            n_inactive = int(inactive_neurons.sum().item())
            n_to_add_clamped = min(n_to_add, n_inactive)
            
            if n_to_add_clamped <= 0:
                return mask
            
            # Select neurons with highest gradient magnitude
            _, indices = torch.topk(neuron_scores, n_to_add_clamped)
            
            new_mask = mask.clone()
            # Reactivate all connections in selected neurons
            new_mask[indices, :] = 1.0
            
            return new_mask
            
        elif self.structure == 'column':
            # Regrow input neurons
            inactive_neurons = (mask.sum(dim=0) == 0).float()
            
            if self.momentum_based_regrowth and gradient is not None:
                neuron_scores = self.get_neuron_importance(weight, 1 - mask, gradient)
            else:
                if gradient is not None:
                    neuron_scores = torch.abs(gradient * (1 - mask)).sum(dim=0)
                else:
                    neuron_scores = torch.rand(mask.shape[1], device=device)
            
            neuron_scores = neuron_scores * inactive_neurons
            
            n_inactive = int(inactive_neurons.sum().item())
            n_to_add_clamped = min(n_to_add, n_inactive)
            
            if n_to_add_clamped <= 0:
                return mask
            
            _, indices = torch.topk(neuron_scores, n_to_add_clamped)
            
            new_mask = mask.clone()
            new_mask[:, indices] = 1.0
            
            return new_mask
        
        else:
            # Unstructured regrowth
            if gradient is not None:
                scores = torch.abs(gradient * (1 - mask)).flatten()
            else:
                scores = torch.rand(mask.numel(), device=device)
            
            n_inactive = int((1 - mask).sum().item())
            n_to_add_clamped = min(n_to_add, n_inactive)
            
            if n_to_add_clamped <= 0:
                return mask
            
            _, indices = torch.topk(scores, n_to_add_clamped)
            
            new_mask = mask.clone().flatten()
            new_mask[indices] = 1.0
            
            return new_mask.reshape(mask.shape)
    
    def on_train_start(self, trainer, pl_module):
        if hasattr(pl_module, 'layer_stacks'):
            ls = pl_module.layer_stacks
            self.layers_to_prune = [
                ('layer_stacks.l1', ls.l1),
                ('layer_stacks.l2', ls.l2),
                ('layer_stacks.output', ls.output)
            ]
        
        # Initialize masks
        for name, layer in self.layers_to_prune:
            if name not in self.masks:
                device = layer.weight.device
                
                if self.use_random_init:
                    # Random sparse initialization (RigL)
                    mask = self.initialize_random_sparse_mask(
                        layer.weight.shape, 
                        self.sparsity, 
                        device
                    )
                    print(f"  {name}: Random sparse init, sparsity={self.sparsity:.4f}")
                else:
                    # Dense initialization
                    mask = torch.ones_like(layer.weight, device=device)
                    print(f"  {name}: Dense initialization")
                
                self.masks[name] = mask
                
                # Initialize gradient momentum buffers
                if self.momentum_based_regrowth:
                    self.gradient_momentum[name] = torch.zeros_like(layer.weight, device=device)
        
        self.initialized = True
    
    def on_after_backward(self, trainer, pl_module):
        if not self.initialized:
            return
        
        # Apply masks to gradients
        for name, layer in self.layers_to_prune:
            if name in self.masks:
                mask = self.masks[name]
                if layer.weight.grad is not None:
                    layer.weight.grad.mul_(mask)
                    
                    # Update gradient momentum
                    if self.momentum_based_regrowth:
                        self.gradient_momentum[name] = (
                            self.momentum_beta * self.gradient_momentum[name] + 
                            (1 - self.momentum_beta) * torch.abs(layer.weight.grad)
                        )
    
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if not self.initialized:
            return
        
        # Apply masks to weights
        with torch.no_grad():
            for name, layer in self.layers_to_prune:
                if name in self.masks:
                    layer.weight.mul_(self.masks[name])
        
        global_step = trainer.global_step
        
        # Initial pruning after warmup
        if global_step == self.warmup_steps and self.warmup_steps > 0:
            print(f"\n=== Warmup complete at step {global_step} ===")
            print(f"Performing initial pruning to sparsity {self.sparsity:.4f}")
            
            with torch.no_grad():
                for name, layer in self.layers_to_prune:
                    if name in self.masks:
                        mask = self.masks[name]
                        weight = layer.weight
                        
                        # Simple magnitude pruning
                        flat_weights = torch.abs(weight).flatten()
                        n_to_keep = int(flat_weights.numel() * (1 - self.sparsity))
                        _, indices = torch.topk(flat_weights, n_to_keep)
                        
                        new_mask = torch.zeros_like(flat_weights)
                        new_mask[indices] = 1.0
                        self.masks[name] = new_mask.reshape(mask.shape)
                        
                        layer.weight.mul_(self.masks[name])
                        
                        active = self.masks[name].sum().item()
                        total = self.masks[name].numel()
                        actual_sparsity = 1.0 - (active / total)
                        print(f"  {name}: {active}/{total} active (sparsity: {actual_sparsity:.4f})")
        
        # Periodic topology updates (RigL)
        if global_step > self.warmup_steps and global_step % self.update_freq == 0:
            self.rigl_update(pl_module, global_step)
    
    def rigl_update(self, pl_module, global_step):
        """
        RigL topology update:
        1. Prune low-magnitude connections
        2. Regrow based on gradient magnitude
        """
        print(f"\n=== RigL Update at step {global_step} ===")
        
        for name, layer in self.layers_to_prune:
            if name in self.masks:
                mask = self.masks[name]
                weight = layer.weight
                device = weight.device
                
                if self.structure in ['neuron', 'row', 'column']:
                    # Structured pruning/regrowth
                    if self.structure == 'neuron' or self.structure == 'row':
                        n_active_neurons = (mask.sum(dim=1) > 0).sum().item()
                        n_prune = int(n_active_neurons * self.prune_rate)
                    elif self.structure == 'column':
                        n_active_neurons = (mask.sum(dim=0) > 0).sum().item()
                        n_prune = int(n_active_neurons * self.prune_rate)
                    
                    print(f"  Layer {name} ({self.structure}):")
                    print(f"    Before: {n_active_neurons} active neurons")
                else:
                    # Unstructured
                    n_active = mask.sum().item()
                    n_prune = int(n_active * self.prune_rate)
                    print(f"  Layer {name} (unstructured):")
                    print(f"    Before: {n_active}/{mask.numel()} active connections")
                
                # Prune
                old_mask = mask.clone()
                mask = self.prune_structured(mask, weight, n_prune)
                
                if self.structure in ['neuron', 'row', 'column']:
                    if self.structure == 'neuron' or self.structure == 'row':
                        n_after_prune = (mask.sum(dim=1) > 0).sum().item()
                    else:
                        n_after_prune = (mask.sum(dim=0) > 0).sum().item()
                    print(f"    After prune: {n_after_prune} active neurons")
                else:
                    print(f"    After prune: {mask.sum().item()}/{mask.numel()} active")
                
                # Regrow using gradient information
                gradient = self.gradient_momentum.get(name) if self.momentum_based_regrowth else layer.weight.grad
                mask = self.regrow_structured(mask, weight, gradient, n_prune)
                
                if self.structure in ['neuron', 'row', 'column']:
                    if self.structure == 'neuron' or self.structure == 'row':
                        n_after_regrow = (mask.sum(dim=1) > 0).sum().item()
                    else:
                        n_after_regrow = (mask.sum(dim=0) > 0).sum().item()
                    print(f"    After regrow: {n_after_regrow} active neurons")
                else:
                    final_active = mask.sum().item()
                    final_sparsity = 1.0 - (final_active / mask.numel())
                    print(f"    After regrow: {final_active}/{mask.numel()} active (s={final_sparsity:.4f})")
                
                self.masks[name] = mask
                
                # Apply mask
                with torch.no_grad():
                    layer.weight.mul_(mask)
