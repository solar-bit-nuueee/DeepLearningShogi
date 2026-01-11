import torch
import torch.nn as nn
import math
import numpy as np

class SRigLScheduler:
    """
    Structured RigL (SRigL) for dlshogi - Standard PyTorch Implementation
    
    Based on "Dynamic Sparse Training with Structured Sparsity" (ICLR 2024)
    by Lasby et al. https://arxiv.org/abs/2305.02299
    
    Key Features:
    1. True dense gradient sampling every δT steps
    2. Salient weights: union of top-K drop and grow sets
    3. Neuron ablation: count(salient) < γ_sal * k'
    4. Exact constant fan-in with strict enforcement
    5. K schedule with cosine decay to 0
    6. ERK distribution with exact parameter count matching
    
    Usage in dlshogi train.py:
        srigl = SRigLScheduler(model, sparsity=0.9, update_freq=1000)
        
        # In training loop:
        for batch in dataloader:
            # 1. Sample dense gradients if needed
            srigl.before_step(batch_data)
            
            # 2. Forward + backward
            loss = train_step(batch)
            loss.backward()
            
            # 3. Apply mask to gradients
            srigl.after_backward()
            
            # 4. Optimizer step
            optimizer.step()
            
            # 5. Apply mask to weights and update topology
            srigl.after_step(step, optimizer)
    """
    
    def __init__(self, 
                 model,
                 sparsity=0.9,
                 update_freq=1000,
                 warmup_steps=0,
                 total_steps=100000,
                 alpha_init=0.3,
                 alpha_final=0.0,
                 gamma_sal=0.3,
                 delta_T=100,
                 use_erk_distribution=True,
                 min_fan_in=1,
                 prune_1x1=False,
                 device='cuda'):
        """
        Args:
            model: PyTorch model
            sparsity: Target global sparsity
            update_freq: Topology update frequency (ΔT)
            warmup_steps: Dense training before first update
            total_steps: Total training steps (used for alpha schedule)
            alpha_init: Initial update rate (K = α * active_weights)
            alpha_final: Final update rate (typically 0.0)
            gamma_sal: Neuron ablation threshold (γ_sal)
            delta_T: Dense gradient sampling interval (δT)
            use_erk_distribution: Use ERK for layer-wise sparsity
            min_fan_in: Minimum fan-in per neuron
            prune_1x1: Whether to prune 1x1 convolutions
            device: Device for computations
        """
        self.model = model
        self.sparsity = sparsity
        self.update_freq = update_freq
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.alpha_init = alpha_init
        self.alpha_final = alpha_final
        self.gamma_sal = gamma_sal
        self.delta_T = delta_T
        self.use_erk_distribution = use_erk_distribution
        self.min_fan_in = min_fan_in
        self.prune_1x1 = prune_1x1
        self.device = device
        
        # State
        self.masks = {}
        self.fan_in = {}  # k' per layer
        self.layer_targets = {} # Target total non-zero params per layer
        self.dense_gradients = {}  # True dense gradients
        self.gradient_steps = 0
        self.layers_to_prune = []
        self.initialized = False
        self.total_updates = 0
        self.next_dense_sample = delta_T
        
        # For dense gradient sampling
        self.cached_batch = None
        self.forward_fn = None
        self.loss_fn = None
        
        print(f"\n{'='*70}")
        print(f"Structured RigL (SRigL) - dlshogi Implementation")
        print(f"{'='*70}")
        print(f"Target Sparsity: {self.sparsity:.4f}")
        print(f"Update Frequency (ΔT): {self.update_freq} steps")
        print(f"Warmup Steps: {self.warmup_steps}")
        print(f"Total Steps: {self.total_steps}")
        print(f"Alpha Schedule: {self.alpha_init} → {self.alpha_final} (cosine)")
        print(f"Neuron Ablation (γ_sal): {self.gamma_sal}")
        print(f"Dense Gradient Sampling (δT): {self.delta_T} steps")
        print(f"ERK Distribution: {self.use_erk_distribution}")
        print(f"Min Fan-in: {self.min_fan_in}")
        print(f"Prune 1x1 Conv: {self.prune_1x1}")
        print(f"{'='*70}\n")
        
        self._initialize()
    
    def _initialize(self):
        """Initialize masks and identify layers to prune."""
        # Identify Conv2d and Linear layers
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                if isinstance(module, nn.Conv2d):
                    # Skip 1x1 convs if requested
                    if not self.prune_1x1 and module.kernel_size == (1, 1):
                        continue
                self.layers_to_prune.append((name, module))
        
        if len(self.layers_to_prune) == 0:
            print("[WARNING] No layers to prune!")
            return
        
        # Compute target parameters
        total_params = sum(module.weight.numel() for _, module in self.layers_to_prune)
        target_params = int(total_params * (1 - self.sparsity))
        
        # Compute ERK distribution
        if self.use_erk_distribution:
            layer_fan_in = self._compute_erk_distribution(target_params)
        else:
            layer_fan_in = {}
            for name, module in self.layers_to_prune:
                if isinstance(module, nn.Conv2d):
                    n_out = module.out_channels
                    n_in = module.in_channels * module.kernel_size[0] * module.kernel_size[1]
                elif isinstance(module, nn.Linear):
                    n_out, n_in = module.weight.shape
                fan_in = max(self.min_fan_in, int(n_in * (1 - self.sparsity)))
                layer_fan_in[name] = fan_in
        
        print("Initializing constant fan-in masks...")
        
        for name, module in self.layers_to_prune:
            weight = module.weight
            
            if isinstance(module, nn.Conv2d):
                # Conv2d: [out_channels, in_channels, kH, kW]
                n_out = module.out_channels
                n_in = module.in_channels * module.kernel_size[0] * module.kernel_size[1]
                shape = (n_out, n_in)
            elif isinstance(module, nn.Linear):
                # Linear: [out_features, in_features]
                n_out, n_in = weight.shape
                shape = (n_out, n_in)
            
            fan_in = layer_fan_in[name]
            self.fan_in[name] = fan_in
            # Store target non-zero count for this layer to preserve distribution during re-computation
            self.layer_targets[name] = n_out * fan_in 
            
            # Initialize mask
            mask = self._initialize_constant_fanin_mask(shape, fan_in)
            self.masks[name] = mask
            
            # Initialize dense gradient buffer
            if isinstance(module, nn.Conv2d):
                self.dense_gradients[name] = torch.zeros_like(weight.view(n_out, n_in))
            else:
                self.dense_gradients[name] = torch.zeros_like(weight)
            
            actual_sparsity = 1.0 - (mask.sum().item() / mask.numel())
            active_neurons = (mask.sum(dim=1) > 0).sum().item()
            
            print(f"  {name}:")
            print(f"    Shape: {n_out}x{n_in}")
            print(f"    Fan-in k': {fan_in}")
            print(f"    Active neurons: {active_neurons}/{n_out}")
            print(f"    Sparsity: {actual_sparsity:.4f}")
        
        self.initialized = True
        print("")
    
    def _compute_erk_distribution(self, target_params):
        """Compute ERK distribution with exact parameter count matching."""
        erk_scores = {}
        layer_shapes = {}
        
        for name, module in self.layers_to_prune:
            if isinstance(module, nn.Conv2d):
                n_out = module.out_channels
                n_in = module.in_channels * module.kernel_size[0] * module.kernel_size[1]
            elif isinstance(module, nn.Linear):
                n_out, n_in = module.weight.shape
            
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
        
        # Adjust for exact target
        current_total = sum(layer_shapes[name][0] * fan_in 
                           for name, fan_in in layer_fan_in.items())
        deficit = target_params - current_total
        
        if deficit != 0:
            sorted_layers = sorted(erk_scores.items(), key=lambda x: x[1], reverse=(deficit > 0))
            
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
        
        final_total = sum(layer_shapes[name][0] * fan_in 
                         for name, fan_in in layer_fan_in.items())
        print(f"ERK Distribution: target={target_params}, actual={final_total}, diff={final_total - target_params}")
        
        return layer_fan_in
    
    def _initialize_constant_fanin_mask(self, shape, fan_in):
        """Initialize mask with exact constant fan-in."""
        n_out, n_in = shape
        mask = torch.zeros(shape, device=self.device)
        
        for i in range(n_out):
            indices = torch.randperm(n_in, device=self.device)[:fan_in]
            mask[i, indices] = 1.0
        
        return mask
    
    def set_forward_loss_fn(self, forward_fn, loss_fn):
        """
        Set forward and loss functions for dense gradient sampling.
        
        Args:
            forward_fn: Function that takes batch and returns model outputs
                       Example: lambda batch: model(batch['x1'], batch['x2'])
            loss_fn: Function that computes loss from outputs and batch
                     Example: lambda outputs, batch: compute_loss(outputs, batch)
        """
        self.forward_fn = forward_fn
        self.loss_fn = loss_fn
    
    def before_step(self, batch):
        """
        Call before optimizer step. Handles dense gradient sampling.
        
        Args:
            batch: Current batch data (dict or tuple)
        """
        if not self.initialized:
            return
        
        self.gradient_steps += 1
        
        # Cache batch for dense gradient sampling
        self.cached_batch = batch
        
        # Dense gradient sampling at δT intervals
        if self.gradient_steps >= self.next_dense_sample:
            print(f"[Dense Gradient Sampling at step {self.gradient_steps}]")
            self._sample_dense_gradients()
            self.next_dense_sample = self.gradient_steps + self.delta_T
    
    def _sample_dense_gradients(self):
        """
        Sample true dense gradients by temporarily removing masks.
        """
        if self.forward_fn is None or self.loss_fn is None:
            print("[WARNING] forward_fn and loss_fn not set. Skipping dense gradient sampling.")
            return
        
        # Save current masks
        saved_masks = {name: mask.clone() for name, mask in self.masks.items()}
        
        try:
            # Temporarily set dense masks
            for name, module in self.layers_to_prune:
                if isinstance(module, nn.Conv2d):
                    n_out = module.out_channels
                    n_in = module.in_channels * module.kernel_size[0] * module.kernel_size[1]
                    self.masks[name] = torch.ones((n_out, n_in), device=self.device)
                else:
                    self.masks[name] = torch.ones_like(module.weight)
            
            # Apply dense masks to weights
            self._apply_masks_to_weights()
            
            # Zero gradients
            self.model.zero_grad()
            
            # Forward + backward with dense weights
            outputs = self.forward_fn(self.cached_batch)
            loss = self.loss_fn(outputs, self.cached_batch)
            loss.backward()
            
            # Store dense gradients
            for name, module in self.layers_to_prune:
                if module.weight.grad is not None:
                    if isinstance(module, nn.Conv2d):
                        # Reshape to 2D
                        n_out = module.out_channels
                        n_in = module.in_channels * module.kernel_size[0] * module.kernel_size[1]
                        grad_2d = module.weight.grad.view(n_out, n_in)
                        self.dense_gradients[name] = torch.abs(grad_2d.clone())
                    else:
                        self.dense_gradients[name] = torch.abs(module.weight.grad.clone())
        except Exception as e:
            print(f"[ERROR] Dense gradient sampling failed: {e}")
        finally:
            # Restore masks AND re-apply them to weights immediately
            # Important: Ensure weights are sparse before next training step
            self.masks = saved_masks
            self._apply_masks_to_weights()
            
            # Clear gradients (don't update weights with dense gradients)
            self.model.zero_grad()
    
    def after_backward(self):
        """
        Call after loss.backward(). Applies mask to gradients.
        """
        if not self.initialized:
            return
        
        for name, module in self.layers_to_prune:
            if module.weight.grad is not None:
                mask = self.masks[name]
                if isinstance(module, nn.Conv2d):
                    # Reshape mask to Conv2d shape
                    n_out, n_in_flat = mask.shape
                    n_in = module.in_channels
                    kh, kw = module.kernel_size
                    mask_4d = mask.view(n_out, n_in, kh, kw)
                    module.weight.grad.mul_(mask_4d)
                else:
                    module.weight.grad.mul_(mask)
    
    def after_step(self, global_step, optimizer=None):
        """
        Call after optimizer.step(). Applies mask to weights and performs topology updates.
        
        Args:
            global_step: Current training step
            optimizer: Optimizer (for resetting state of regrown weights)
        """
        if not self.initialized:
            return
        
        # Apply masks to weights
        self._apply_masks_to_weights()
        
        # Initial pruning after warmup
        if global_step == self.warmup_steps and self.warmup_steps > 0:
            print(f"\n=== Warmup complete at step {global_step} ===")
            self._enforce_constant_fanin()
        
        # Periodic topology updates
        if global_step > self.warmup_steps and (global_step - self.warmup_steps) % self.update_freq == 0:
            self._srigl_update(global_step, optimizer)
    
    def _apply_masks_to_weights(self):
        """Apply masks to model weights."""
        with torch.no_grad():
            for name, module in self.layers_to_prune:
                mask = self.masks[name]
                if isinstance(module, nn.Conv2d):
                    # Reshape mask to Conv2d shape
                    n_out, n_in_flat = mask.shape
                    n_in = module.in_channels
                    kh, kw = module.kernel_size
                    mask_4d = mask.view(n_out, n_in, kh, kw)
                    module.weight.mul_(mask_4d)
                else:
                    module.weight.mul_(mask)
    
    def _enforce_constant_fanin(self):
        """Enforce constant fan-in by magnitude pruning."""
        print("Enforcing constant fan-in constraint...")
        
        for name, module in self.layers_to_prune:
            mask = self.masks[name]
            weight = module.weight
            fan_in = self.fan_in[name]
            
            if isinstance(module, nn.Conv2d):
                n_out = module.out_channels
                n_in = module.in_channels * module.kernel_size[0] * module.kernel_size[1]
                weight_2d = weight.view(n_out, n_in)
            else:
                n_out, n_in = weight.shape
                weight_2d = weight
            
            new_mask = torch.zeros_like(mask)
            
            for i in range(n_out):
                row_weights = torch.abs(weight_2d[i])
                _, indices = torch.topk(row_weights, min(fan_in, n_in))
                new_mask[i, indices] = 1.0
            
            self.masks[name] = new_mask
            
            # Strict check
            fanin_per_neuron = new_mask.sum(dim=1)
            active_neurons = fanin_per_neuron > 0
            if active_neurons.any():
                std_fanin = fanin_per_neuron[active_neurons].std().item()
                assert std_fanin < 1e-6, f"Fan-in not constant: std={std_fanin}"
            
            n_active = active_neurons.sum().item()
            actual_sparsity = 1.0 - (new_mask.sum().item() / new_mask.numel())
            print(f"  {name}: k'={fan_in}, active={n_active}/{n_out}, s={actual_sparsity:.4f}")
        
        print("")
    
    def _compute_alpha(self, step):
        """Cosine decay schedule for update rate α(t).
        Decays from alpha_init to alpha_final over 75% of total training steps
        or total_updates if total_steps is not provided.
        """
        # If total_steps is set, use it for schedule
        if self.total_steps > 0:
            # End schedule at 75% of training (common RigL practice)
            end_step = self.warmup_steps + 0.75 * (self.total_steps - self.warmup_steps)
            if step >= end_step:
                return self.alpha_final
            
            progress = (step - self.warmup_steps) / (end_step - self.warmup_steps)
        else:
            # Fallback to update_freq based (original impl)
            progress = (step - self.warmup_steps) / (self.update_freq * 100)
            
        progress = min(1.0, max(0.0, progress))
        
        alpha = self.alpha_final + 0.5 * (self.alpha_init - self.alpha_final) * (1 + math.cos(math.pi * progress))
        return alpha
    
    def _reset_optimizer_state(self, optimizer, old_mask, new_mask, param):
        """Reset optimizer state (momentum, etc) for regrown weights."""
        if optimizer is None: return
        
        # grew: 0 -> 1, pruned: 1 -> 0
        grew = (old_mask == 0) & (new_mask == 1)
        pruned = (old_mask == 1) & (new_mask == 0)
        
        state = optimizer.state.get(param)
        if state is None: return
        
        # Support for SGD, AdamW, Muon
        # keys to reset: 'momentum_buffer' (SGD), 'exp_avg', 'exp_avg_sq' (Adam*)
        keys_to_reset = ['momentum_buffer', 'exp_avg', 'exp_avg_sq']
        
        # Muon specific state? Muon usually just has momentum_buffer
        
        for key in keys_to_reset:
            if key in state:
                buf = state[key]
                # Ensure buffer shape matches param (especially for Conv2d 4D vs 2D mask)
                if buf.shape != grew.shape:
                    if len(buf.shape) == 4 and len(grew.shape) == 2:
                         # Reshape mask to 4D for element-wise op
                        n_out, n_in = grew.shape
                        kh, kw = param.shape[2], param.shape[3]
                        grew_View = grew.view(n_out, n_in // (kh*kw), kh, kw)
                        pruned_View = pruned.view(n_out, n_in // (kh*kw), kh, kw)
                        
                        buf.mul_((~pruned_View).to(buf.dtype))
                        buf[grew_View.bool()] = 0.0
                else:
                    buf.mul_((~pruned).to(buf.dtype))
                    buf[grew.bool()] = 0.0
    
    def _srigl_update(self, global_step, optimizer=None):
        """
        SRigL topology update.
        """
        print(f"\n{'='*70}")
        print(f"SRigL Update #{self.total_updates + 1} at step {global_step}")
        print(f"{'='*70}")
        
        # Force dense gradient sample if stale
        if (self.gradient_steps + 1) >= self.next_dense_sample:
             print("Force sampling dense gradients before update...")
             self._sample_dense_gradients()
             self.next_dense_sample = self.gradient_steps + self.delta_T
        
        alpha = self._compute_alpha(global_step)
        print(f"Update rate α(t): {alpha:.4f}\n")
        
        for name, module in self.layers_to_prune:
            mask = self.masks[name]
            weight = module.weight
            gradient = self.dense_gradients[name]
            fan_in = self.fan_in[name]
            
            if isinstance(module, nn.Conv2d):
                n_out = module.out_channels
                n_in = module.in_channels * module.kernel_size[0] * module.kernel_size[1]
                weight_2d = weight.view(n_out, n_in)
            else:
                n_out, n_in = weight.shape
                weight_2d = weight
            
            print(f"Layer: {name}")
            
            # Compute K
            n_active = mask.sum().item()
            K = max(1, int(alpha * n_active))
            print(f"  K (connections to update): {K}")
            
            # Compute salient weights (paper Step 3)
            salient_mask = self._compute_salient_weights(mask, weight_2d, gradient, K)
            
            # Neuron ablation (paper Step 4)
            new_mask, n_ablated, active_neurons = self._ablate_neurons(
                mask, salient_mask, fan_in, self.gamma_sal
            )
            
            if n_ablated > 0:
                print(f"  Ablated neurons: {n_ablated}")
                
                # Recompute k' (paper Step 5)
                # Use layer_targets to respect ERK distribution
                new_fan_in = self._recompute_fan_in(name, new_mask)
                
                if new_fan_in != fan_in:
                    print(f"  Recomputed k': {fan_in} → {new_fan_in}")
                    self.fan_in[name] = new_fan_in
                    fan_in = new_fan_in
            
            # Redistribute connections (paper Steps 6 & 7)
            final_mask = self._redistribute_connections(new_mask, weight_2d, gradient, fan_in, K)
            
            # Reset optimizer state for regrown weights
            self._reset_optimizer_state(optimizer, mask, final_mask, module.weight)
            
            # Update mask
            self.masks[name] = final_mask
            
            # Consistency checks
            nz = int(final_mask.sum().item())
            # For strict ERK compliance, total active params should match target (within rounding/ablation limits)
            # Note: Ablation + constant fan-in might drift slightly from original target if not perfectly divisible, 
            # but usually it's close. We'll log it.
            target_nz = self.layer_targets[name]
            
            # Check constant fan-in for active neurons
            fanin_per_neuron = final_mask.sum(dim=1)
            active_fanins = fanin_per_neuron[fanin_per_neuron > 0]
            if len(active_fanins) > 0:
                min_f = active_fanins.min().item()
                max_f = active_fanins.max().item()
                assert min_f == max_f == fan_in, \
                    f"Fan-in consistency failed! Target: {fan_in}, Min: {min_f}, Max: {max_f}"
            
            n_active_neurons = (final_mask.sum(dim=1) > 0).sum().item()
            actual_sparsity = 1.0 - (nz / final_mask.numel())
            
            print(f"  Active neurons: {n_active_neurons}/{n_out}")
            print(f"  Active connections: {nz} (Target: {target_nz})")
            print(f"  Sparsity: {actual_sparsity:.4f}")
            print(f"  Fan-in: {fan_in}")
            print("")
        
        self.total_updates += 1
        print(f"{'='*70}\n")
    
    def _compute_salient_weights(self, mask, weight, gradient, K):
        """Compute salient weights (paper Step 3).

        A weight is salient if it is in the top-K of either:
        - largest-magnitude active weights (|w| among mask==1)
        - largest-magnitude gradients of pruned weights (|g| among mask==0)
        """
        device = weight.device
        
        # Drop criterion for saliency: TOP-K *largest* magnitudes among ACTIVE weights
        active_mask = mask > 0
        active_weights = torch.abs(weight) * mask
        active_indices = active_mask.flatten().nonzero(as_tuple=True)[0]
        
        if len(active_indices) > 0 and K > 0:
            active_magnitudes = active_weights.flatten()[active_indices]
            K_mag = min(K, len(active_indices))
            _, mag_idx_local = torch.topk(active_magnitudes, K_mag, largest=True)
            mag_idx = active_indices[mag_idx_local]
        else:
            mag_idx = torch.tensor([], dtype=torch.long, device=device)
        
        # Grow criterion for saliency: TOP-K *largest* gradients among PRUNED weights
        pruned_mask = mask == 0
        pruned_gradients = torch.abs(gradient) * pruned_mask
        pruned_indices = pruned_mask.flatten().nonzero(as_tuple=True)[0]
        
        if len(pruned_indices) > 0 and K > 0:
            pruned_mags = pruned_gradients.flatten()[pruned_indices]
            K_grad = min(K, len(pruned_indices))
            _, grad_idx_local = torch.topk(pruned_mags, K_grad, largest=True)
            grad_idx = pruned_indices[grad_idx_local]
        else:
            grad_idx = torch.tensor([], dtype=torch.long, device=device)
        
        # Union
        salient_flat = torch.zeros(weight.numel(), device=device)
        if len(mag_idx) > 0:
            salient_flat[mag_idx] = 1.0
        if len(grad_idx) > 0:
            salient_flat[grad_idx] = 1.0
        
        salient_mask = salient_flat.reshape(weight.shape)
        return salient_mask
    
    def _ablate_neurons(self, mask, salient_mask, fan_in, gamma_sal):
        """Neuron ablation (paper Step 4)."""
        salient_per_neuron = salient_mask.sum(dim=1)
        threshold = gamma_sal * fan_in
        active_neurons = salient_per_neuron >= threshold
        
        new_mask = mask.clone()
        new_mask[~active_neurons, :] = 0.0
        
        n_ablated = (~active_neurons).sum().item()
        return new_mask, n_ablated, active_neurons
    
    def _recompute_fan_in(self, name, mask):
        """Recompute k' after neuron ablation (paper Step 5).
        Correctly uses layer-specific target parameter counts to respect ERK.
        """
        # Retrieve target non-zero count for this layer (preserved from init)
        target_active = self.layer_targets.get(name)
        if target_active is None:
             # Fallback if somehow missing (shouldn't happen)
             n_out, n_in = mask.shape
             target_active = int(n_out * n_in * (1 - self.sparsity))
        
        n_active_neurons = (mask.sum(dim=1) > 0).sum().item()
        
        if n_active_neurons == 0:
            return self.min_fan_in
        
        new_fan_in = max(self.min_fan_in, target_active // n_active_neurons)
        return new_fan_in
    
    def _redistribute_connections(self, mask, weight, gradient, fan_in, K):
        """Redistribute connections following SRigL paper Steps 6 & 7.

        Step 6 (layer-wise): prune the K smallest-magnitude ACTIVE weights.
        Step 7 (per-neuron): regrow weights (from currently pruned positions) in
        descending order of |gradient| until each active neuron reaches fan_in.
        """
        n_out, n_in = mask.shape
        device = mask.device

        # Track which neurons are active after ablation (paper Step 4)
        active_neurons = mask.sum(dim=1) > 0

        # Step 6: prune K smallest |w| in this layer
        active_mask = mask > 0
        active_weights = torch.abs(weight) * mask
        active_indices = active_mask.flatten().nonzero(as_tuple=True)[0]

        if len(active_indices) > 0 and K > 0:
            K_prune = min(K, len(active_indices))
            active_magnitudes = active_weights.flatten()[active_indices]
            _, prune_idx_local = torch.topk(active_magnitudes, K_prune, largest=False)
            prune_idx = active_indices[prune_idx_local]

            mask_flat = mask.flatten()
            mask_flat[prune_idx] = 0.0
            mask = mask_flat.reshape(n_out, n_in)

        # Step 7: regrow per neuron up to fan_in using largest |grad|
        for i in range(n_out):
            if not active_neurons[i]:
                continue

            current = int(mask[i].sum().item())
            if current < fan_in:
                needed = fan_in - current
                inactive_idx = (mask[i] == 0).nonzero(as_tuple=True)[0]
                if inactive_idx.numel() == 0:
                    continue

                row_grads = torch.abs(gradient[i, inactive_idx])
                n_grow = min(needed, inactive_idx.numel())
                _, grow_local = torch.topk(row_grads, n_grow, largest=True)
                mask[i, inactive_idx[grow_local]] = 1.0
            elif current > fan_in:
                # Should not happen if K_prune was effective, but safe-guard
                active_idx = (mask[i] > 0).nonzero(as_tuple=True)[0]
                row_w = torch.abs(weight[i, active_idx])
                _, keep_local = torch.topk(row_w, fan_in, largest=True)
                new_row = torch.zeros((n_in,), device=device, dtype=mask.dtype)
                new_row[active_idx[keep_local]] = 1.0
                mask[i] = new_row

        return mask
